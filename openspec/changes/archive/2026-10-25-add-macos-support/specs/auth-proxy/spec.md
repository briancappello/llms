# Spec Delta

## Purpose

Let the optional LAN authentication proxy find its API key file in the same
documented place on Linux and macOS, consistent with the rest of `llms`'
configuration.

## ADDED Requirements

### Requirement: Key file location
Unless `LSA_KEYS_FILE` is set, the proxy SHALL read keys from
`$XDG_CONFIG_HOME/llama-swap/api-keys`. If `XDG_CONFIG_HOME` is unset, it SHALL
read `$HOME/.config/llama-swap/api-keys`. This SHALL hold on both Linux and
macOS.

#### Scenario: macOS default
- **WHEN** the proxy starts on macOS with neither `LSA_KEYS_FILE` nor `XDG_CONFIG_HOME` set
- **THEN** it reads `~/.config/llama-swap/api-keys`, not `~/Library/Application Support/llama-swap/api-keys`

#### Scenario: Explicit override
- **WHEN** `LSA_KEYS_FILE` is set
- **THEN** that file is used on every platform

### Requirement: Credentials are never forwarded
The proxy SHALL accept a Bearer token or `x-api-key`. It SHALL reject requests
without a valid key and SHALL remove credentials before forwarding a request
upstream. This behavior SHALL be the same on both platforms.

#### Scenario: Valid key forwarded without credentials
- **WHEN** a request carries a valid `Authorization: Bearer` key
- **THEN** the upstream request contains neither `Authorization` nor `x-api-key`
