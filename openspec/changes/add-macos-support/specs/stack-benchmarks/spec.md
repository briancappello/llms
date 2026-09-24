# Spec Delta

## Purpose

Measure models as they are actually served: behind llama-swap and the engine
that serves them in production, configured exactly as the registry and the
host's engines specify (long-context agentic settings, speculative decoding
enabled), and driven over HTTP. This must work the same way on Linux and macOS.

## ADDED Requirements

### Requirement: Benchmarks run the production configuration
Every benchmark that reports throughput, context capacity, memory or
speculative acceptance SHALL serve the model through llama-swap. The command
SHALL be rendered by `llms` from the host's real settings and registry. A
variant SHALL be the production entry plus an explicit, recorded change (for
example a different `ctx`, KV cache type, `mtp` setting or engine). Benchmarks
SHALL NOT start inference servers with command lines written by hand.

#### Scenario: Variant is a recorded change to the production entry
- **WHEN** a KV sweep compares f16 and q8_0 V cache for a model
- **THEN** each variant's served command equals the production render except for the declared change, and the change is written into the results

#### Scenario: Speculative decoding stays on
- **WHEN** a production entry enables MTP or a draft model
- **THEN** its benchmark variants keep it enabled unless disabling it is the declared change

### Requirement: Isolated benchmark instance
Benchmarks SHALL run in an isolated `llms` instance with its own config
directory and listen port. The production service SHALL be stopped for the
duration through `llms stop` and restored afterwards through `llms start`. It
SHALL be restored even when the benchmark fails or is interrupted.

#### Scenario: Interrupted run restores service
- **WHEN** a benchmark is interrupted with Ctrl-C while the isolated instance is serving
- **THEN** the isolated llama-swap and its child servers are stopped, and the production service is started again

#### Scenario: Production registry untouched
- **WHEN** a benchmark run finishes
- **THEN** the production `registry.json`, `settings.json` and rendered config are byte-identical to how they were before the run

### Requirement: Measurement through the server API
Prefill and generation throughput, speculative acceptance, and token counts
SHALL be taken from the served stack's HTTP responses and endpoints:
- chat completion timings;
- `/metrics`;
- `/upstream/<model>/...`, for example `/tokenize`.

Profiling tools (`llama-bench`) and estimators (`llama-fit-params`) SHALL NOT be
used as the source of any recorded result.

#### Scenario: Depth sweep
- **WHEN** a depth sweep runs at depths 4096, 32768 and 65536
- **THEN** each result row comes from a served chat completion with a prompt trimmed to that depth through the backend's `/tokenize`

### Requirement: Portable server memory measurement
Benchmarks SHALL record the memory used by the served model's inference process
on both platforms:
- on Linux, the dGPU VRAM in use (AMD or NVIDIA);
- on macOS, the inference process's physical footprint, together with the
  Metal recommended working-set budget.

If memory cannot be read, the value SHALL be recorded as unavailable. It SHALL
NOT be recorded as zero, and it SHALL NOT fail the benchmark.

#### Scenario: Context capacity on macOS
- **WHEN** a context-capacity probe loads a model at increasing `ctx` on macOS
- **THEN** each step records the footprint of the served process and the budget, and the largest `ctx` that loads and answers a request is reported

#### Scenario: No GPU tool available
- **WHEN** neither `rocm-smi` nor `nvidia-smi` is present on a Linux host
- **THEN** the memory column is recorded as unavailable and the run continues

### Requirement: Benchmark scripts are portable
Benchmark entry points SHALL run on Linux and on macOS (including stock bash
3.2) without GNU-only tools. They SHALL find model files through the registry,
not through hard-coded HF cache globs or host-specific install paths.

#### Scenario: Run a sweep on the Mac
- **WHEN** a depth sweep runs on the Mac host for a registered model
- **THEN** it completes using the Mac's configured engine, with no reference to Vulkan, HIP or ROCm
