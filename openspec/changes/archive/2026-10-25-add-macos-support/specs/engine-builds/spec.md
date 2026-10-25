# Spec Delta

## Purpose

Build llama.cpp engine variants that are isolated from one another on Linux and
macOS. Scope the pinned vLLM ROCm build to the Linux hosts it supports.

## ADDED Requirements

### Requirement: Supported backends per platform
`bin/build-engine` SHALL accept these backends:
- `metal` and `cpu` on macOS;
- `vulkan`, `hip`, `hip-wmma`, `cuda` and `cpu` on Linux.

A backend that is not supported on the current platform SHALL fail before any
clone, fetch or configure step. A `cpu` build SHALL explicitly disable Metal.

#### Scenario: Metal build on macOS
- **WHEN** `build-engine metal metal` runs on an Apple Silicon Mac with the prerequisites installed
- **THEN** `~/opt/llama.cpp-metal/bin/llama-server` is installed and `COMMIT` metadata is written

#### Scenario: HIP requested on macOS
- **WHEN** `build-engine hip hip` runs on macOS
- **THEN** it exits non-zero with an "unsupported on Darwin" error and touches no source tree

#### Scenario: CPU build on macOS
- **WHEN** `build-engine cpu cpu` runs on macOS
- **THEN** the build is configured with Metal disabled

### Requirement: Library isolation is enforced on both platforms
Every installed engine SHALL find its `libggml*` and `libllama*` libraries only
inside its own install prefix. The build SHALL fail if it finds any such library
outside the prefix:
- on Linux, using the resolved dynamic dependencies;
- on macOS, using load commands and rpaths that resolve relative to the binary.

#### Scenario: Leaked library on macOS
- **WHEN** the installed macOS `llama-server` would load `libggml` from a path outside its prefix
- **THEN** the build exits non-zero and reports an isolation failure listing each library that leaked

### Requirement: Scripts run under stock shells and tools
`bin/build-engine` SHALL run correctly under macOS's stock `/bin/bash` 3.2 and
the BSD userland, and under bash 5 with GNU coreutils. It SHALL NOT depend on
`nproc`, GNU-only `date` flags, or bash 4+ features. An empty extra-argument
list SHALL be valid.

#### Scenario: No extra cmake arguments under bash 3.2
- **WHEN** `/bin/bash bin/build-engine cpu cpu` runs on macOS with no extra arguments
- **THEN** the build does not fail with an unbound-variable error

#### Scenario: Missing prerequisites
- **WHEN** `ninja` or `cmake` is not installed
- **THEN** the script exits before cloning or configuring, naming the missing tool

### Requirement: vLLM ROCm build is Linux-only
`bin/build-vllm`, `bin/run-vllm-reranker`, `bin/run-vllm-embeddings` and their
`make` targets SHALL exit non-zero with a clear message on any host that is not
Linux. The only exception is that `build-vllm --print-config` and `--dry-run`
SHALL work on every platform.

#### Scenario: make vllm on macOS
- **WHEN** `make vllm` runs on macOS
- **THEN** it fails immediately with a message that the vLLM ROCm build is Linux-only

#### Scenario: Print config anywhere
- **WHEN** `bin/build-vllm --print-config` runs on macOS
- **THEN** it prints the pinned configuration and exits zero
