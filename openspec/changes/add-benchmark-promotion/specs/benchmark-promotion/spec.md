## Purpose

Turn a benchmark conclusion into the production configuration. The variant that
was measured best is applied to production exactly as it was measured, guarded
against drift, rolled back if it does not serve, and logged.

## ADDED Requirements

### Requirement: Promote a measured variant
The operator SHALL be able to promote one variant from a stack-mode results file
(TSV or kv-probe JSON) by naming the file and the variant. Promotion SHALL apply
that variant's recorded patch to the production registry entries and engines,
using the same merge rules the benchmark used to build the variant. It SHALL
then re-render the llama-swap config and restart the production service.
Promoting the `production` variant (an empty patch) SHALL change no files and
SHALL still be logged.

#### Scenario: KV cache winner becomes production
- **WHEN** a KV sweep found the `q8V` variant best and the operator promotes `q8V` from that results file
- **THEN** the production registry entry carries the q8_0 V cache arguments, the rendered config serves them, and the service is restarted

#### Scenario: Engine patch is promoted
- **WHEN** the promoted variant patches an engine (for example its `mtp_args`)
- **THEN** that engine in production `settings.json` is changed and every other key in the file is kept

#### Scenario: Engine shared with other models
- **WHEN** the promoted patch changes an engine that other registered models also use, so their rendered commands change too
- **THEN** promotion lists those models and fails unless the operator passes an explicit flag accepting them

#### Scenario: Patch reaches beyond the measured model
- **WHEN** the recorded patch changes a registry entry other than the model the rows measured
- **THEN** promotion fails and changes nothing

#### Scenario: Unknown variant
- **WHEN** the named variant has no rows in the results file
- **THEN** promotion fails, names the variants the file does contain, and changes nothing

### Requirement: Promotion refuses drifted or unprovable results
Before writing anything, promotion SHALL verify that production still matches the
measurement:
- the digest of the current production base equals the base digest recorded
  with the variant;
- after the patch is applied, the command production would render for the model
  equals the recorded rendered command.

If either check fails, if the result rows for the variant disagree on patch,
base digest or command, if the results carry no base digest, or if the result is
from standalone mode, promotion SHALL fail with the reason and change nothing.

#### Scenario: Production edited after the run
- **WHEN** the model's production entry was edited after the sweep was recorded
- **THEN** promotion fails, reports that the base digest differs, and production files are unchanged

#### Scenario: Standalone result
- **WHEN** the operator tries to promote from a kv-probe result recorded in standalone mode
- **THEN** promotion fails and tells the operator to re-measure the change as a stack variant

#### Scenario: Results from before base digests
- **WHEN** the results file has no base digest
- **THEN** promotion fails and asks for a re-run

### Requirement: Promotion is verified and rolled back on failure
After restarting, promotion SHALL send a warm-up request to the promoted model
through the production service and require a valid answer. If the write, the
restart or the warm-up fails, promotion SHALL restore the previous registry,
settings and rendered config byte for byte, restart the service again, and
report the failure. The service SHALL only be restarted if it was active
before promotion; an inactive service SHALL be left stopped and the warm-up
skipped, which the log records.

#### Scenario: Promoted config fails to load
- **WHEN** the promoted variant renders a command the server rejects
- **THEN** the warm-up fails, the previous files are restored byte-identical, the service is restarted on them, and promotion exits non-zero

### Requirement: Promotion log
Every successful promotion SHALL append one row to `results/promotions.tsv` in
the repository, with: UTC time, host name, model, variant, results file, the
patch, the base digest, and the rendered command. A failed promotion SHALL NOT
append a row.

#### Scenario: Log row written
- **WHEN** a promotion succeeds
- **THEN** `results/promotions.tsv` gains one row naming the model, variant and results file it came from
