# Spec Delta

## Purpose

Runtime controls over the running llama-swap stack and the local clients that
use it. They behave the same on Linux and macOS, whichever engine serves the
model.

## ADDED Requirements

### Requirement: Unload models
The system SHALL provide `llms unload [NAME]`. With a name, it SHALL unload only
that model. Without one, it SHALL unload all running models. The llama-swap
service stays up in both cases. Unloading an unknown name SHALL fail without
changing anything.

#### Scenario: Unload all
- **WHEN** `llms unload` runs while a model is loaded
- **THEN** `/running` reports no loaded models and llama-swap keeps answering

### Requirement: Port preflight before start
Before starting the llama-swap service, the system SHALL check whether the
configured listen address is already taken by a process other than the managed
service. If it is, start SHALL fail with an error naming the address, and the
service manager SHALL NOT be called.

#### Scenario: Port taken by another process
- **WHEN** another program is listening on `127.0.0.1:18080` and `llms start` runs
- **THEN** the command fails with a "listen address in use" error and does not call `launchctl` or `systemctl`

### Requirement: Warm-up accepts every supported reasoning field
`llms use` SHALL treat an assistant answer as valid if it has any of: string
`content`, string `reasoning_content`, string `reasoning`, or well-formed tool
calls.

#### Scenario: MLX warm-up
- **WHEN** a model served by an MLX server answers with `content: null` and a string `reasoning`
- **THEN** `llms use` reports the model ready

### Requirement: Client default model sync
`llms use NAME --sync-clients` SHALL update the pi provider's model list as
`ensure-clients` does. It SHALL also set pi's default provider and default model
to NAME, so that both clients point at the same model and neither causes model
swaps. The default model SHALL be set only when the answer to the warm-up
request is valid. Other pi settings SHALL be kept.

#### Scenario: Sync pi default
- **WHEN** `llms use kat-apex --sync-clients` succeeds
- **THEN** pi's settings name the llama-swap provider and `kat-apex` as the default, and other keys are unchanged

#### Scenario: Failed warm-up
- **WHEN** the warm-up request fails
- **THEN** pi's settings are not modified
