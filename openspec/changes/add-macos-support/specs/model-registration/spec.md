# Spec Delta

## Purpose

Scaffold registry entries from GGUF and cache metadata in a way that holds
across hosts. Surface metadata that changes served behavior without the
operator knowing, and find complete models on disk that are not registered.

## ADDED Requirements

### Requirement: MTP scaffolded as intent
When `llms add` detects MTP layers in a GGUF, the new registry entry SHALL set
`"mtp": true`. It SHALL NOT add literal speculative flags to `extra_args`, so
each host's engine decides the speculative arguments.

#### Scenario: Adding an MTP model
- **WHEN** `llms add` registers a GGUF whose metadata reports `nextn_predict_layers` greater than 0
- **THEN** the entry contains `"mtp": true`, and its `extra_args` contain no `--spec-type` flag

### Requirement: Embedded sampling metadata is reported
When a GGUF contains `general.sampling.*` keys, `llms add` SHALL include them in
its output and record them in the entry under a non-rendering field. llama.cpp
applies these values unless they are overridden. `add` SHALL NOT turn them into
sampling flags.

#### Scenario: GGUF with embedded temperature
- **WHEN** `llms add` registers a GGUF that has `general.sampling.temp` = 1.0
- **THEN** the command output and the entry show `temp: 1.0` as embedded sampling, and the rendered command has no `--temp` flag

### Requirement: Discover unregistered cached models
The system SHALL list complete models in the configured HF cache that no
registry entry references:
- GGUF targets, with shard groups collapsed and projectors and drafts excluded;
- MLX snapshot directories that contain `config.json` and at least one
  `.safetensors` file.

Incomplete downloads SHALL NOT be listed. Discovery SHALL be read-only and
offline.

#### Scenario: Partial MLX download hidden
- **WHEN** a cached MLX snapshot has safetensors shards but no `config.json`
- **THEN** it does not appear in the unregistered listing

#### Scenario: Registered model hidden
- **WHEN** a GGUF in the cache is already the `path` of a registry entry
- **THEN** it does not appear in the unregistered listing
