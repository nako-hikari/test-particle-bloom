#pragma once
#include <pl/Mod.hpp>

namespace bloom {
bool install(ll::mod::NativeMod& self);  // call from enable()
void uninstall();                        // call from disable()
}  // namespace bloom
