// Shaders.h - GLSL ES 3.00 sources for the bloom passes.
// STATUS: written for this project, NOT yet compiled on a device.
// The dual Kawase filter taps follow the commonly published version
// (Marius Bjorge, ARM, SIGGRAPH 2015). I wrote them from memory and did NOT
// fetch the paper in this session, so treat the exact weights as "to verify".
#pragma once

namespace bloom::shaders {

// One oversized triangle that covers the screen. No vertex buffer needed:
// gl_VertexID (0,1,2) generates the positions. Bind an EMPTY VAO and call
// glDrawArrays(GL_TRIANGLES, 0, 3).
inline constexpr const char* kFullscreenVS = R"(#version 300 es
out vec2 vUV;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUV = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";

// Downsample: 5 taps (centre weighted x4 + 4 diagonals), result / 8.
// uHalfPixel = 0.5 / (size of the DESTINATION texture).
inline constexpr const char* kDownFS = R"(#version 300 es
precision mediump float;
uniform sampler2D uTex;
uniform vec2 uHalfPixel;
in vec2 vUV;
out vec4 oColor;
void main() {
  vec4 s = texture(uTex, vUV) * 4.0;
  s += texture(uTex, vUV - uHalfPixel);
  s += texture(uTex, vUV + uHalfPixel);
  s += texture(uTex, vUV + vec2(uHalfPixel.x, -uHalfPixel.y));
  s += texture(uTex, vUV - vec2(uHalfPixel.x, -uHalfPixel.y));
  oColor = s / 8.0;
}
)";

// Upsample: 8 taps in a ring, result / 12.
// uHalfPixel = 0.5 / (size of the SOURCE texture).
inline constexpr const char* kUpFS = R"(#version 300 es
precision mediump float;
uniform sampler2D uTex;
uniform vec2 uHalfPixel;
in vec2 vUV;
out vec4 oColor;
void main() {
  vec4 s = texture(uTex, vUV + vec2(-uHalfPixel.x * 2.0, 0.0));
  s += texture(uTex, vUV + vec2(-uHalfPixel.x,  uHalfPixel.y)) * 2.0;
  s += texture(uTex, vUV + vec2(0.0,  uHalfPixel.y * 2.0));
  s += texture(uTex, vUV + vec2( uHalfPixel.x,  uHalfPixel.y)) * 2.0;
  s += texture(uTex, vUV + vec2( uHalfPixel.x * 2.0, 0.0));
  s += texture(uTex, vUV + vec2( uHalfPixel.x, -uHalfPixel.y)) * 2.0;
  s += texture(uTex, vUV + vec2(0.0, -uHalfPixel.y * 2.0));
  s += texture(uTex, vUV + vec2(-uHalfPixel.x, -uHalfPixel.y)) * 2.0;
  oColor = s / 12.0;
}
)";

// Composite / debug-show: output rgb * intensity.
// For the real composite the caller enables blending with (GL_ONE, GL_ONE),
// i.e. the result is ADDED to what is already in the target framebuffer.
inline constexpr const char* kCompositeFS = R"(#version 300 es
precision mediump float;
uniform sampler2D uTex;
uniform float uIntensity;
in vec2 vUV;
out vec4 oColor;
void main() {
  oColor = vec4(texture(uTex, vUV).rgb * uIntensity, 1.0);
}
)";

}  // namespace bloom::shaders
