// BloomMod.cpp - staged skeleton for a "real bloom for unlit particles" mod.
//
// STATUS: UNTESTED. Written in this session WITHOUT compiling against the
// Preloader SDK and WITHOUT running on a device. Treat every GL-state detail
// as "to verify on your phone".
//
// SDK names used (all taken from the official docs, see README sources):
//   PL_REGISTER_MOD, ll::mod::NativeMod::current(), getLogger().info/warn,
//   pl::memory::HookHandle(target, detour, &original, priority), .installed(), .reset(),
//   pl::memory::HookPriority::Normal.
// Everything else is standard OpenGL ES 3.0 / EGL / libc.
//
// Build stage with:  -DBLOOM_STAGE=N   (1..5)  - see README for how to verify each.
//   1  log GL draw activity so you can find the particle program ("learn mode")
//   2  create the mask framebuffer (shares the scene's depth buffer)
//   3  draw a TEST colour into the mask and show it as a corner thumbnail
//   4  replay particle draws into the mask + dual Kawase blur; show blurred thumbnail
//   5  additive full-screen composite (the actual bloom)

#include <pl/Mod.hpp>
#include <pl/memory/Hook.hpp>

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <dlfcn.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "Shaders.h"

#ifndef BLOOM_STAGE
#define BLOOM_STAGE 1
#endif
// Diagnostics (no adb needed): a small magenta square at the bottom-left corner drawn at swap time,
// plus a status file. Set to 0 once everything works.
#ifndef BLOOM_DIAG
#define BLOOM_DIAG 1
#endif

namespace {

// ---- configuration --------------------------------------------------------
// TODO(you): fill this in after Stage 1. 0 = "learn mode, don't replay anything".
// It is a fingerprint of the shader program (hash of its uniform + attribute
// names), NOT a Minecraft offset, so it should survive game updates unless Mojang
// renames shader inputs.
constexpr uint64_t kTargetFingerprint = 0x67045df0827b190dull;  // from your status file: s_ParticleTexture program (all particles; lit/unlit split TBD)
constexpr int kBlurLevels = 4;
constexpr float kIntensity = 0.8f;

// ---- hook plumbing --------------------------------------------------------
using DrawElementsFn = void (*)(GLenum, GLsizei, GLenum, const void*);
using SwapFn = EGLBoolean (*)(EGLDisplay, EGLSurface);

DrawElementsFn gOrigDrawElements = nullptr;
SwapFn gOrigSwap = nullptr;
pl::memory::HookHandle gHookDraw;
pl::memory::HookHandle gHookSwap;
ll::mod::NativeMod* gSelf = nullptr;
uint64_t gFrame = 0;

// ---- diagnostics state ----
std::string gDataDir;
uint64_t gDrawCalls = 0, gSwapCalls = 0;
std::map<int, uint32_t> gFboCounts;   // draw FBO id -> number of draws seen
GLint gLastVp[4] = {0, 0, 0, 0};
uint64_t gDrawKinds[4] = {0, 0, 0, 0};  // 0=DrawElements 1=DrawElementsInstanced 2=DrawArrays 3=DrawArraysInstanced
using GetProcFn = void* (*)(const char*);
GetProcFn gOrigGetProc = nullptr;
pl::memory::HookHandle gHookGetProc;
std::mutex gProcMutex;
std::map<std::string, uint32_t> gProcNames;  // every name the game asked eglGetProcAddress for
bool gSwapHookOk = false, gGetProcHookOk = false;

// Per-shader statistics (keyed by program fingerprint) - helps spot the particle shader(s).
struct ProgStat {
  uint64_t draws = 0, blended = 0, noDepthWrite = 0, depthTested = 0, prevDraws = 0, recent = 0;
  GLint srcRGB = 0, dstRGB = 0;
  std::map<int, uint64_t> perFbo;
};
std::map<uint64_t, ProgStat> gProgStats;
std::map<uint64_t, std::string> gDescByFp;      // fingerprint -> uniform/attribute names (truncated)
std::map<int, std::pair<int, int>> gFboViewport;  // draw FBO id -> last viewport w,h
std::string gBlitReport;                        // result of the depth-copy experiment
bool gBlitTested = false;
std::string gLibNote;  // which .so the GL/EGL exports live in (dladdr)

// ---- Stage 1: shader fingerprinting ---------------------------------------
std::unordered_map<GLuint, uint64_t> gFingerprintByProgram;  // NOTE: GL may reuse ids; fine for a prototype
std::unordered_map<uint64_t, uint32_t> gDrawsThisWindow;      // fingerprint -> draws

uint64_t fnv1a(uint64_t h, const char* s) {
  for (; *s; ++s) { h ^= static_cast<uint8_t>(*s); h *= 1099511628211ull; }
  return h;
}

uint64_t fingerprintProgram(GLuint prog) {
  auto it = gFingerprintByProgram.find(prog);
  if (it != gFingerprintByProgram.end()) return it->second;

  uint64_t h = 1469598103934665603ull;
  std::string desc;
  GLint nU = 0, nA = 0;
  glGetProgramiv(prog, GL_ACTIVE_UNIFORMS, &nU);
  glGetProgramiv(prog, GL_ACTIVE_ATTRIBUTES, &nA);
  char name[128];
  for (GLint i = 0; i < nU; ++i) {
    GLsizei len = 0; GLint size = 0; GLenum type = 0;
    glGetActiveUniform(prog, i, sizeof(name), &len, &size, &type, name);
    h = fnv1a(h, name); desc += name; desc += ' ';
  }
  for (GLint i = 0; i < nA; ++i) {
    GLsizei len = 0; GLint size = 0; GLenum type = 0;
    glGetActiveAttrib(prog, i, sizeof(name), &len, &size, &type, name);
    h = fnv1a(h, name); desc += name; desc += ' ';
  }
  gFingerprintByProgram[prog] = h;
  gDescByFp[h] = desc.substr(0, 220);
  if (gSelf) {
    gSelf->getLogger().info("new program id={} fp={:016x} uniforms={} attribs={} :: {}",
                            prog, h, nU, nA, desc.substr(0, 300));
  }
  return h;
}

// ---- GL state guard -------------------------------------------------------
// bgfx (RenderDragon) caches GL state. If we change state behind its back and do
// not restore it EXACTLY, the game can render garbage. Extend this list if you see glitches.
struct StateGuard {
  GLint fbo, readFbo, prog, vao, activeTex, tex0, vp[4], blendSrcRGB, blendDstRGB, blendSrcA, blendDstA,
      blendEqRGB, blendEqA;
  GLboolean blend, depth, scissor, cull, stencil, colorMask[4], depthMask;
  StateGuard() {
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &fbo);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFbo);
    glGetIntegerv(GL_CURRENT_PROGRAM, &prog);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTex);
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &tex0);
    glGetIntegerv(GL_VIEWPORT, vp);
    glGetIntegerv(GL_BLEND_SRC_RGB, &blendSrcRGB);
    glGetIntegerv(GL_BLEND_DST_RGB, &blendDstRGB);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &blendSrcA);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &blendDstA);
    glGetIntegerv(GL_BLEND_EQUATION_RGB, &blendEqRGB);
    glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &blendEqA);
    blend = glIsEnabled(GL_BLEND);
    depth = glIsEnabled(GL_DEPTH_TEST);
    scissor = glIsEnabled(GL_SCISSOR_TEST);
    cull = glIsEnabled(GL_CULL_FACE);
    stencil = glIsEnabled(GL_STENCIL_TEST);
    glGetBooleanv(GL_COLOR_WRITEMASK, colorMask);
    glGetBooleanv(GL_DEPTH_WRITEMASK, &depthMask);
  }
  ~StateGuard() {
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, readFbo);
    glUseProgram(prog);
    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex0);
    glActiveTexture(activeTex);
    glViewport(vp[0], vp[1], vp[2], vp[3]);
    glBlendFuncSeparate(blendSrcRGB, blendDstRGB, blendSrcA, blendDstA);
    glBlendEquationSeparate(blendEqRGB, blendEqA);
    auto set = [](GLenum cap, GLboolean on) { on ? glEnable(cap) : glDisable(cap); };
    set(GL_BLEND, blend); set(GL_DEPTH_TEST, depth); set(GL_SCISSOR_TEST, scissor);
    set(GL_CULL_FACE, cull); set(GL_STENCIL_TEST, stencil);
    glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
    glDepthMask(depthMask);
  }
};

// ---- Stage 2: mask framebuffer -------------------------------------------
struct Target { GLuint fbo = 0, tex = 0; int w = 0, h = 0; };
struct Mask {
  Target color;                 // RGBA8 mask (prototype; RGBA16F needs an extension check - TODO)
  GLint sceneFbo = -1;          // which scene FBO we borrowed depth from
  GLint depthName = -1;
  GLuint depthRb = 0;          // our OWN depth+stencil buffer (copy of the scene depth)
  bool ready = false;
  bool warnedDefaultFb = false;
} gMask;

void makeTarget(Target& t, int w, int h, bool withFbo = true) {
  if (!t.tex) glGenTextures(1, &t.tex);
  glBindTexture(GL_TEXTURE_2D, t.tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  if (withFbo) {
    if (!t.fbo) glGenFramebuffers(1, &t.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.tex, 0);
  }
  t.w = w; t.h = h;
}

// The scene is drawn straight into the DEFAULT framebuffer (FBO 0, per your status file), and the default
// framebuffer's depth buffer can't be attached to another FBO. So the mask keeps its OWN D24S8 depth buffer and,
// once per frame (at the first particle draw), COPIES the scene depth into it with glBlitFramebuffer.
// Your device test showed D24S8 is the one format for which that blit works.
// Only the main scene view (FBO 0, viewport >= 1500 px wide) is used; other FBOs are ignored.
bool ensureMask() {
  GLint fbo = 0, vp[4];
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &fbo);
  glGetIntegerv(GL_VIEWPORT, vp);
  if (fbo != 0 || vp[2] < 1500) return false;
  if (gMask.ready && gMask.color.w == vp[2] && gMask.color.h == vp[3]) return true;

  GLint prevTex = 0, prevRb = 0, prevRead = 0, prevDraw = 0;
  glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex);
  glGetIntegerv(GL_RENDERBUFFER_BINDING, &prevRb);
  glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prevRead);
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevDraw);

  makeTarget(gMask.color, vp[2], vp[3]);  // leaves the mask FBO bound
  if (!gMask.depthRb) glGenRenderbuffers(1, &gMask.depthRb);
  glBindRenderbuffer(GL_RENDERBUFFER, gMask.depthRb);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, vp[2], vp[3]);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, gMask.depthRb);
  const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

  glBindTexture(GL_TEXTURE_2D, prevTex);
  glBindRenderbuffer(GL_RENDERBUFFER, prevRb);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, prevRead);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prevDraw);

  gMask.ready = (status == GL_FRAMEBUFFER_COMPLETE);
  if (gSelf)
    gSelf->getLogger().info("mask {}x{} (own D24S8 depth) status=0x{:x} ready={}", vp[2], vp[3], status, gMask.ready);
  return gMask.ready;
}

// Copy the scene's current depth into the mask's own depth buffer, once per frame.
uint64_t gDepthCopiedFrame = ~0ull;
void copyDepthIfNeeded() {
  if (gDepthCopiedFrame == gFrame) return;
  gDepthCopiedFrame = gFrame;
  GLint rd = 0, dr = 0;
  glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &rd);
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &dr);
  const GLboolean scissor = glIsEnabled(GL_SCISSOR_TEST);
  glDisable(GL_SCISSOR_TEST);  // scissor would clip the blit
  glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gMask.color.fbo);
  glBlitFramebuffer(0, 0, gMask.color.w, gMask.color.h, 0, 0, gMask.color.w, gMask.color.h,
                    GL_DEPTH_BUFFER_BIT, GL_NEAREST);
  if (scissor) glEnable(GL_SCISSOR_TEST);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, rd);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dr);
}

// ---- Stage 3-5: programs, blur chain, composite --------------------------
struct Prog { GLuint id = 0; GLint uTex = -1, uHalf = -1, uIntensity = -1; };
Prog gDown, gUp, gComposite;
GLuint gEmptyVao = 0;
Target gChain[kBlurLevels];
bool gGpuInit = false;

GLuint compile(GLenum type, const char* src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[512]; glGetShaderInfoLog(s, sizeof(log), nullptr, log);
    if (gSelf) gSelf->getLogger().error("shader compile failed: {}", log);
    glDeleteShader(s); return 0;
  }
  return s;
}

bool makeProg(Prog& p, const char* fs) {
  GLuint v = compile(GL_VERTEX_SHADER, bloom::shaders::kFullscreenVS);
  GLuint f = compile(GL_FRAGMENT_SHADER, fs);
  if (!v || !f) return false;
  p.id = glCreateProgram();
  glAttachShader(p.id, v); glAttachShader(p.id, f);
  glLinkProgram(p.id);
  glDeleteShader(v); glDeleteShader(f);
  GLint ok = 0; glGetProgramiv(p.id, GL_LINK_STATUS, &ok);
  if (!ok) { if (gSelf) gSelf->getLogger().error("program link failed"); return false; }
  p.uTex = glGetUniformLocation(p.id, "uTex");
  p.uHalf = glGetUniformLocation(p.id, "uHalfPixel");
  p.uIntensity = glGetUniformLocation(p.id, "uIntensity");
  return true;
}

bool initGpu() {
  if (gGpuInit) return true;
  gGpuInit = makeProg(gDown, bloom::shaders::kDownFS) && makeProg(gUp, bloom::shaders::kUpFS) &&
             makeProg(gComposite, bloom::shaders::kCompositeFS);
  if (gGpuInit) glGenVertexArrays(1, &gEmptyVao);
  return gGpuInit;
}

void drawFullscreen(const Prog& p, GLuint srcTex, const Target& dst, float halfX, float halfY, float intensity) {
  glBindFramebuffer(GL_FRAMEBUFFER, dst.fbo);
  glViewport(0, 0, dst.w, dst.h);
  glUseProgram(p.id);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, srcTex);
  glUniform1i(p.uTex, 0);
  if (p.uHalf >= 0) glUniform2f(p.uHalf, halfX, halfY);
  if (p.uIntensity >= 0) glUniform1f(p.uIntensity, intensity);
  glBindVertexArray(gEmptyVao);
  glDrawArrays(GL_TRIANGLES, 0, 3);
}

// Dual Kawase: mask -> level0 (1/2) -> level1 (1/4) ... then back up.
void blurMask() {
  for (int i = 0; i < kBlurLevels; ++i) {
    const int w = std::max(1, gMask.color.w >> (i + 1)), h = std::max(1, gMask.color.h >> (i + 1));
    if (gChain[i].w != w || gChain[i].h != h) makeTarget(gChain[i], w, h);
  }
  glDisable(GL_BLEND); glDisable(GL_DEPTH_TEST); glDisable(GL_SCISSOR_TEST);
  glDisable(GL_CULL_FACE); glDisable(GL_STENCIL_TEST);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  for (int i = 0; i < kBlurLevels; ++i) {  // down
    const GLuint src = (i == 0) ? gMask.color.tex : gChain[i - 1].tex;
    drawFullscreen(gDown, src, gChain[i], 0.5f / gChain[i].w, 0.5f / gChain[i].h, 1.f);
  }
  for (int i = kBlurLevels - 1; i >= 1; --i) {  // up
    drawFullscreen(gUp, gChain[i].tex, gChain[i - 1], 0.5f / gChain[i].w, 0.5f / gChain[i].h, 1.f);
  }
}

void clearMaskColor(float r, float g, float b, float a) {
  glBindFramebuffer(GL_FRAMEBUFFER, gMask.color.fbo);
  glDisable(GL_SCISSOR_TEST);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glClearColor(r, g, b, a);
  glClear(GL_COLOR_BUFFER_BIT);  // colour only: the depth buffer is SHARED with the game, never clear it
}

void runBloomAtSwap() {
  if (!gMask.ready || !initGpu()) return;
  StateGuard guard;
  const Target screen{0, 0, guard.vp[2], guard.vp[3]};

#if BLOOM_STAGE == 3
  clearMaskColor(1.f, 0.f, 1.f, 1.f);  // test colour: magenta
  glDisable(GL_BLEND); glDisable(GL_DEPTH_TEST); glDisable(GL_SCISSOR_TEST);
  Target thumb{0, 0, screen.w / 4, screen.h / 4};
  drawFullscreen(gComposite, gMask.color.tex, thumb, 0, 0, 1.f);
  clearMaskColor(0.f, 0.f, 0.f, 0.f);
#elif BLOOM_STAGE >= 4
  blurMask();
  glDisable(GL_DEPTH_TEST); glDisable(GL_SCISSOR_TEST); glDisable(GL_CULL_FACE); glDisable(GL_STENCIL_TEST);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
#if BLOOM_STAGE == 4
  glDisable(GL_BLEND);  // debug: show the blurred mask in a corner thumbnail
  Target thumb{0, 0, screen.w / 4, screen.h / 4};
  drawFullscreen(gComposite, gChain[0].tex, thumb, 0, 0, 1.f);
#else
  glEnable(GL_BLEND);   // the actual bloom: ADD blurred particles on top of the frame
  glBlendFunc(GL_ONE, GL_ONE);
  glBlendEquation(GL_FUNC_ADD);
  drawFullscreen(gComposite, gChain[0].tex, screen, 0, 0, kIntensity);
#endif
  clearMaskColor(0.f, 0.f, 0.f, 0.f);  // ready for next frame
#endif
}

// ---- diagnostics ---------------------------------------------------------
// Proof-of-life marker: if you see a magenta square (120x120 px) in the bottom-left corner,
// the eglSwapBuffers hook runs and can draw on the screen.
void drawMarker() {
  GLboolean sc = glIsEnabled(GL_SCISSOR_TEST);
  GLint box[4]; glGetIntegerv(GL_SCISSOR_BOX, box);
  GLfloat cc[4]; glGetFloatv(GL_COLOR_CLEAR_VALUE, cc);
  GLboolean cm[4]; glGetBooleanv(GL_COLOR_WRITEMASK, cm);
  GLint fbo = 0; glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &fbo);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  glEnable(GL_SCISSOR_TEST);
  glScissor(0, 0, 120, 120);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glClearColor(1.f, 0.f, 1.f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT);
  glClearColor(cc[0], cc[1], cc[2], cc[3]);
  glColorMask(cm[0], cm[1], cm[2], cm[3]);
  glScissor(box[0], box[1], box[2], box[3]);
  sc ? glEnable(GL_SCISSOR_TEST) : glDisable(GL_SCISSOR_TEST);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
}

void writeDiag() {
  std::string text;
  text += std::string("build=") + __DATE__ + " " + __TIME__ + "\n";  // proves WHICH binary is running
  text += "frame=" + std::to_string(gFrame) + "\n";
  text += "BLOOM_STAGE=" + std::to_string(BLOOM_STAGE) + "\n";
  text += "swap_calls=" + std::to_string(gSwapCalls) + "\n";
  text += "draw_calls=" + std::to_string(gDrawCalls) + "\n";
  text += "programs_seen=" + std::to_string(gFingerprintByProgram.size()) + "\n";
  text += "last_viewport=" + std::to_string(gLastVp[0]) + "," + std::to_string(gLastVp[1]) + "," +
          std::to_string(gLastVp[2]) + "," + std::to_string(gLastVp[3]) + "\n";
  text += "mask_ready=" + std::to_string(gMask.ready) + "\n";
  text += "scene_fbo_is_default_warning=" + std::to_string(gMask.warnedDefaultFb) + "\n";
  text += "swap_hook_installed=" + std::to_string(gSwapHookOk) + "\n";
  text += "getproc_hook_installed=" + std::to_string(gGetProcHookOk) + "\n";
  text += gLibNote;
  text += "draw_kinds elements=" + std::to_string(gDrawKinds[0]) + " elementsInst=" + std::to_string(gDrawKinds[1]) +
          " arrays=" + std::to_string(gDrawKinds[2]) + " arraysInst=" + std::to_string(gDrawKinds[3]) + "\n";
  {
    std::lock_guard<std::mutex> lk(gProcMutex);
    text += "egl_getproc_distinct_names=" + std::to_string(gProcNames.size()) + "\n";
    text += "glDraw*/glUseProgram/glBindFramebuffer names asked:";
    for (auto& [n, c] : gProcNames)
      if (n.rfind("glDraw", 0) == 0 || n == "glUseProgram" || n == "glBindFramebuffer") text += " " + n + "x" + std::to_string(c);
    text += "\nfirst 40 names:";
    int k = 0;
    for (auto& [n, c] : gProcNames) { if (k++ >= 40) break; text += " " + n; }
    text += "\n";
  }
  text += gBlitReport;
  text += "viewport per FBO:";
  for (auto& [fbo, wh] : gFboViewport) text += " " + std::to_string(fbo) + "=" + std::to_string(wh.first) + "x" + std::to_string(wh.second);
  text += "\n--- shader programs (fp draws recent120 blended noDepthWrite depthTested blend(src/dst) fbos :: names) ---\n";
  for (auto& [fp, st] : gProgStats) {
    st.recent = st.draws - st.prevDraws;
    st.prevDraws = st.draws;
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%016llx draws=%llu recent=%llu blended=%llu nodepthwrite=%llu depthtest=%llu blend=0x%x/0x%x fbos=",
                  (unsigned long long)fp, (unsigned long long)st.draws, (unsigned long long)st.recent,
                  (unsigned long long)st.blended, (unsigned long long)st.noDepthWrite,
                  (unsigned long long)st.depthTested, st.srcRGB, st.dstRGB);
    text += buf;
    for (auto& [fbo, n] : st.perFbo) text += std::to_string(fbo) + ":" + std::to_string(n) + ",";
    text += " :: " + gDescByFp[fp] + "\n";
  }
  text += "---\n";
  text += "draws_per_fbo (fbo:count):";
  for (auto& [fbo, n] : gFboCounts) text += " " + std::to_string(fbo) + ":" + std::to_string(n);
  text += "\n";
  // Try the mod's own data dir, and also a spot easy to open in a file manager (may fail: UNVERIFIED).
  for (const std::string& path : {gDataDir + "/bloom_status.txt", std::string("/storage/emulated/0/Download/bloom_status.txt")}) {
    std::ofstream f(path, std::ios::trunc);
    if (f) f << text;
  }
}

// ---- detours --------------------------------------------------------------
// One-time experiment: can the DEFAULT framebuffer's depth be copied into our own buffer?
// (Needed because the scene is drawn straight into FBO 0, whose depth buffer we cannot attach elsewhere.)
// ES 3.0 requires matching depth formats for a depth blit, so we try several and log glGetError for each.
void runDepthBlitTest(int w, int h) {
  gBlitTested = true;
  GLint rd = 0, dr = 0, orb = 0, depthBits = 0, stencilBits = 0;
  glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &rd);
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &dr);
  glGetIntegerv(GL_RENDERBUFFER_BINDING, &orb);
  glGetIntegerv(GL_DEPTH_BITS, &depthBits);
  glGetIntegerv(GL_STENCIL_BITS, &stencilBits);
  gBlitReport += "--- depth blit test (default FB -> own renderbuffer) ---\n";
  gBlitReport += "test size=" + std::to_string(w) + "x" + std::to_string(h) + " default_fb depth_bits=" +
                 std::to_string(depthBits) + " stencil_bits=" + std::to_string(stencilBits) + "\n";
  struct F { GLenum fmt; GLenum att; const char* name; };
  const F fs[] = {{GL_DEPTH_COMPONENT24, GL_DEPTH_ATTACHMENT, "D24"},
                  {GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL_ATTACHMENT, "D24S8"},
                  {GL_DEPTH_COMPONENT16, GL_DEPTH_ATTACHMENT, "D16"},
                  {GL_DEPTH_COMPONENT32F, GL_DEPTH_ATTACHMENT, "D32F"},
                  {GL_DEPTH32F_STENCIL8, GL_DEPTH_STENCIL_ATTACHMENT, "D32FS8"}};
  while (glGetError() != GL_NO_ERROR) {}  // NOTE: swallows any pending game error; fine for a one-off test
  for (const F& f : fs) {
    GLuint rb = 0, fb = 0;
    glGenRenderbuffers(1, &rb);
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorage(GL_RENDERBUFFER, f.fmt, w, h);
    const GLenum storeErr = glGetError();
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fb);
    glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, f.att, GL_RENDERBUFFER, rb);
    const GLenum none = GL_NONE;
    glDrawBuffers(1, &none);  // depth-only FBO must have draw buffer NONE to be complete
    const GLenum status = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    const GLenum blitErr = glGetError();
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s: storage_err=0x%x fbo_status=0x%x blit_err=0x%x %s\n", f.name, storeErr,
                  status, blitErr, (blitErr == GL_NO_ERROR && status == GL_FRAMEBUFFER_COMPLETE) ? "<== WORKS" : "");
    gBlitReport += buf;
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fb);
    glDeleteRenderbuffers(1, &rb);
  }
  glBindRenderbuffer(GL_RENDERBUFFER, orb);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, rd);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dr);
}

using DrawElementsInstFn = void (*)(GLenum, GLsizei, GLenum, const void*, GLsizei);
using DrawArraysFn = void (*)(GLenum, GLint, GLsizei);
using DrawArraysInstFn = void (*)(GLenum, GLint, GLsizei, GLsizei);
DrawElementsInstFn gOrigDrawElementsInst = nullptr;
DrawArraysFn gOrigDrawArrays = nullptr;
DrawArraysInstFn gOrigDrawArraysInst = nullptr;

// Called for every draw. Records stats; returns true when this draw must ALSO be replayed into the mask.
bool noteDraw(int kind) {
  ++gDrawCalls;
  ++gDrawKinds[kind];
  GLint prog = 0;
  glGetIntegerv(GL_CURRENT_PROGRAM, &prog);  // TODO(perf): track via a glUseProgram hook instead
  const uint64_t fp = fingerprintProgram(static_cast<GLuint>(prog));
  GLint dfbo = 0;
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &dfbo);
  gFboCounts[dfbo]++;
  glGetIntegerv(GL_VIEWPORT, gLastVp);
  gFboViewport[dfbo] = {gLastVp[2], gLastVp[3]};
  {
    ProgStat& st = gProgStats[fp];
    ++st.draws;
    st.perFbo[dfbo]++;
    if (glIsEnabled(GL_BLEND)) ++st.blended;
    if (glIsEnabled(GL_DEPTH_TEST)) ++st.depthTested;
    GLboolean dm = GL_TRUE; glGetBooleanv(GL_DEPTH_WRITEMASK, &dm);
    if (!dm) ++st.noDepthWrite;
    if (st.draws == 1) {
      glGetIntegerv(GL_BLEND_SRC_RGB, &st.srcRGB);
      glGetIntegerv(GL_BLEND_DST_RGB, &st.dstRGB);
    }
  }
  if (dfbo == 0 && !gBlitTested && gFrame > 300 && gLastVp[2] >= 1500) runDepthBlitTest(gLastVp[2], gLastVp[3]);
#if BLOOM_STAGE == 1
  gDrawsThisWindow[fp]++;
#endif
#if BLOOM_STAGE >= 4
  return kTargetFingerprint != 0 && fp == kTargetFingerprint && ensureMask();
#elif BLOOM_STAGE >= 2
  if (!gMask.ready) ensureMask();
  return false;
#else
  (void)fp;
  return false;
#endif
}

// Replay FIRST, then the real draw (see README: avoids the replay failing its own depth test).
template <class F>
void replayIntoMask(F&& draw) {
  copyDepthIfNeeded();  // mask gets the scene depth as it is at the first particle draw of the frame
  GLint prev = 0;
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prev);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gMask.color.fbo);
  draw();
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prev);
}

void hkDrawElements(GLenum m, GLsizei c, GLenum t, const void* i) {
  if (noteDraw(0)) replayIntoMask([&] { gOrigDrawElements(m, c, t, i); });
  gOrigDrawElements(m, c, t, i);
}
void hkDrawElementsInst(GLenum m, GLsizei c, GLenum t, const void* i, GLsizei n) {
  if (noteDraw(1)) replayIntoMask([&] { gOrigDrawElementsInst(m, c, t, i, n); });
  gOrigDrawElementsInst(m, c, t, i, n);
}
void hkDrawArrays(GLenum m, GLint f, GLsizei c) {
  if (noteDraw(2)) replayIntoMask([&] { gOrigDrawArrays(m, f, c); });
  gOrigDrawArrays(m, f, c);
}
void hkDrawArraysInst(GLenum m, GLint f, GLsizei c, GLsizei n) {
  if (noteDraw(3)) replayIntoMask([&] { gOrigDrawArraysInst(m, f, c, n); });
  gOrigDrawArraysInst(m, f, c, n);
}

// Idea: the game (bgfx) probably looks up GL functions with eglGetProcAddress and keeps the
// returned pointers, so hooking the libGLES exports misses them. Instead we hook eglGetProcAddress
// itself and hand back OUR functions for the draw calls. UNVERIFIED: requires that this mod is
// enabled before the game resolves its GL functions (the tutorial says enable() runs before game start).
void* hkGetProc(const char* name) {
  void* real = gOrigGetProc ? gOrigGetProc(name) : nullptr;
  if (!name) return real;
  {
    std::lock_guard<std::mutex> lk(gProcMutex);
    if (gProcNames.size() < 3000) ++gProcNames[name];
  }
  if (!real) return real;
  auto is = [&](const char* s) { return std::strcmp(name, s) == 0; };
  if (is("glDrawElements")) { gOrigDrawElements = reinterpret_cast<DrawElementsFn>(real); return reinterpret_cast<void*>(&hkDrawElements); }
  if (is("glDrawElementsInstanced")) { gOrigDrawElementsInst = reinterpret_cast<DrawElementsInstFn>(real); return reinterpret_cast<void*>(&hkDrawElementsInst); }
  if (is("glDrawArrays")) { gOrigDrawArrays = reinterpret_cast<DrawArraysFn>(real); return reinterpret_cast<void*>(&hkDrawArrays); }
  if (is("glDrawArraysInstanced")) { gOrigDrawArraysInst = reinterpret_cast<DrawArraysInstFn>(real); return reinterpret_cast<void*>(&hkDrawArraysInst); }
  return real;
}

EGLBoolean hkSwap(EGLDisplay d, EGLSurface s) {
  ++gFrame;
  ++gSwapCalls;
#if BLOOM_DIAG
  drawMarker();
  if (gFrame % 120 == 0) writeDiag();
#endif
#if BLOOM_STAGE == 1
  if (gFrame % 600 == 0 && gSelf) {
    for (auto& [fp, n] : gDrawsThisWindow)
      gSelf->getLogger().info("window summary: fp={:016x} draws={}", fp, n);
    gDrawsThisWindow.clear();
  }
#endif
#if BLOOM_STAGE >= 3
  runBloomAtSwap();
#endif
  return gOrigSwap(d, s);
}

}  // namespace

// ---- public entry points (called from the template's MyMod.cpp) ----------
// Declared in BloomMod.h. The template's own mod class keeps PL_REGISTER_MOD;
// it just calls bloom::install() in enable() and bloom::uninstall() in disable().
#include "BloomMod.h"

namespace bloom {

bool install(ll::mod::NativeMod& self) {
  gSelf = &self;
  gDataDir = self.getDataDir().string();
  self.getLogger().info("particle-bloom starting, BLOOM_STAGE={}", BLOOM_STAGE);
  // Hook targets: eglGetProcAddress (to intercept the game's GL lookups) and eglSwapBuffers (frame end).
  void* getProcT = dlsym(RTLD_DEFAULT, "eglGetProcAddress");
  void* swapT = dlsym(RTLD_DEFAULT, "eglSwapBuffers");
  void* drawT = dlsym(RTLD_DEFAULT, "glDrawElements");  // diagnostics only
  for (auto [label, addr] : {std::pair<const char*, void*>{"eglGetProcAddress", getProcT},
                             {"eglSwapBuffers", swapT}, {"glDrawElements", drawT}}) {
    Dl_info info{};
    gLibNote += std::string(label) + " addr=" + (addr ? "set" : "NULL") + " lib=" +
                ((addr && dladdr(addr, &info) && info.dli_fname) ? info.dli_fname : "?") + "\n";
  }
  if (!getProcT || !swapT) {
    self.getLogger().warn("EGL symbol lookup failed (getproc={}, swap={})", getProcT != nullptr, swapT != nullptr);
    return false;
  }
  gHookGetProc = pl::memory::HookHandle(getProcT, reinterpret_cast<void*>(&hkGetProc),
                                        reinterpret_cast<void**>(&gOrigGetProc),
                                        pl::memory::HookPriority::Normal);
  gHookSwap = pl::memory::HookHandle(swapT, reinterpret_cast<void*>(&hkSwap),
                                     reinterpret_cast<void**>(&gOrigSwap),
                                     pl::memory::HookPriority::Normal);
  gGetProcHookOk = gHookGetProc.installed();
  gSwapHookOk = gHookSwap.installed();
  self.getLogger().info("hooks installed: getproc={} swap={}", gGetProcHookOk, gSwapHookOk);
  return true;
}

void uninstall() {
  gHookDraw.reset();
  gHookGetProc.reset();
  gHookSwap.reset();
  // GL objects are leaked on purpose (deleting needs the render thread's context). TODO for later.
}

}  // namespace bloom
