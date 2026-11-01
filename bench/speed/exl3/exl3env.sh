# Source before running anything from the exllamav3 dev checkout on the R9700:  . exl3env.sh  [convert]
export HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1201 CXX=/opt/rocm/llvm/bin/clang++   # skip iGPU; g++ can't build (exllamav3#459)
export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions_exl3dev
export MALLOC_MMAP_THRESHOLD_=1048576          # convert.py RSS ratchet (exllamav3#460); free for inference

# hipBLASLt fp32 GEMMs are wrong past 2^19 rows on gfx1201 (ROCm/rocm-libraries#13243). Only convert.py hits
# that (lm_head of large-vocab models), and exllamav3 PR #458 fixes it in code. Disabling hipBLASLt globally
# costs ~18% decode speed (88 -> 72 tok/s on Cyber-Tiel 4.54bpw), so only do it for conversions on trees
# without #458:  . exl3env.sh convert
if [ "$1" = "convert" ]; then
    export ROCBLAS_USE_HIPBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0
else
    unset ROCBLAS_USE_HIPBLASLT TORCH_BLAS_PREFER_HIPBLASLT
fi
