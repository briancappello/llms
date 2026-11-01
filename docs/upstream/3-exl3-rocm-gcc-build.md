# ROCm: JIT build fails with g++ — force-included rocm/compat.h pulls clang-only builtins into .cpp files

**Setup:** `dev` @ 0662fac / 1426f1c, ROCm 7.2.4, torch 2.14.1+rocm7.2, GCC 16.2.1 (default `c++`).

On the ROCm build every source gets `-include rocm/compat.h`, including the plain `.cpp` files that torch compiles with the host compiler. compat.h includes `hip/hip_bf16.h`, and ROCm 7.2's `amd_hip_bf16.h` uses clang-only builtins, so g++ fails on `avx2_target.cpp`, `dsv4_attn_hip.cpp` and others:

```
amd_hip_bf16.h:844: error: '__builtin_elementwise_fma' was not declared in this scope
amd_hip_bf16.h:350: error: static assertion failed
```

**Workaround:** `CXX=/opt/rocm/llvm/bin/clang++`.

**Possible fixes:**
- only force-include compat.h for `.cu` sources, or
- default the host compiler to ROCm's clang on HIP builds.
