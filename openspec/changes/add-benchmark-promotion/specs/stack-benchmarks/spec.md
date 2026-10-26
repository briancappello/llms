## MODIFIED Requirements

### Requirement: Benchmarks run the production configuration
Every benchmark that reports throughput, context capacity, memory or
speculative acceptance SHALL serve the model through llama-swap. The command
SHALL be rendered by `llms` from the host's real settings and registry. A
variant SHALL be the production entry plus an explicit, recorded change (for
example a different `ctx`, KV cache type, `mtp` setting or engine). Benchmarks
SHALL NOT start inference servers with command lines written by hand.

One exception is allowed for exploration: a benchmark MAY measure an inference
server that the operator started outside the stack (standalone mode). Every
result from standalone mode SHALL be labelled `standalone`, SHALL record the
endpoint it measured, and SHALL NOT carry a variant patch. A standalone result
SHALL NOT be used to change production (see benchmark-promotion).

#### Scenario: Variant is a recorded change to the production entry
- **WHEN** a KV sweep compares f16 and q8_0 V cache for a model
- **THEN** each variant's served command equals the production render except for the declared change, and the change is written into the results

#### Scenario: Speculative decoding stays on
- **WHEN** a production entry enables MTP or a draft model
- **THEN** its benchmark variants keep it enabled unless disabling it is the declared change

#### Scenario: Standalone probe is labelled
- **WHEN** the KV probe measures a `llama-server` the operator started by hand, given its base URL
- **THEN** the results record mode `standalone` and that URL, with no variant patch, and the production service is neither stopped nor started

## ADDED Requirements

### Requirement: Results record the production base
Every stack-mode result SHALL record, beside the variant patch and the rendered
command, a digest of the production configuration the patch was applied to:
the model's production registry entry, the production definition of the engine
it is served by and of every engine the patch names, and the header macros. The
digest SHALL change whenever any of them changes, and SHALL NOT depend on JSON
key order or whitespace.

#### Scenario: Digest follows the production entry
- **WHEN** a sweep is run, then the model's production `ctx` is edited, then the same sweep is run again
- **THEN** the two result files record different base digests for that model

#### Scenario: Digest ignores formatting
- **WHEN** production `settings.json` is re-indented without any value changing
- **THEN** a new run records the same base digest as before
