# Model serving: locked configuration and operations

Companion to `README.md`. Everything about *which* models are served, with *what* flags, and how to switch
between them. For building llama.cpp see `~/dev/my-llama/RUNBOOK.md`. Measured data behind
every claim here is in `results/`.

## Source of truth

```
config/registry.json      <-- EDIT HERE, this is the only source
        |  llm render (or: make render)
        v
~/.config/llama-swap/config.yaml           <-- GENERATED. Do not edit.
        |  systemd --user llama-swap.service  (binds 127.0.0.1:8080)
        v
http://127.0.0.1:8080/v1    every model advertised by its real name
```

`config.yaml` is regenerated wholesale from `registry.json` plus
`config.header.yaml`. Any edit made directly to `config.yaml` is lost on the next
`llm render (or: make render)`.

**Model selection is on-demand by name.** Every chat model is listed in
`/v1/models` under its real id (`cold-fusion`, `bigbang-v1`, ...); a client picks
one by setting `model` in a normal OpenAI request and llama-swap loads/swaps to
it. Single user, so a request triggering a swap is fine -- there is no `taichi`
virtual id and no profiles. The default model is preloaded at startup via
`hooks.on_startup.preload`. `globalTTL: 0` means nothing unloads on a timer.

## Network access

llama-swap binds **127.0.0.1:8080 only** and has **no auth** -- local tools (pi,
opencode, `llm`, benches) use it directly and keyless.

LAN/remote access goes through **`llama-swap-auth`** (`proxy/main.go`, a
std-lib Go reverse proxy) on **`0.0.0.0:4096`**, which requires a bearer key and
forwards to `127.0.0.1:8080`. Port 4096 is already open in firewalld; 8080 is
not, so localhost is the only keyless path. Why a proxy and not llama-swap's own
`apiKeys`: those are **global** (they apply to localhost too, with no exemption),
which would force every local tool to carry a key.

```
local  ->  http://127.0.0.1:8080/v1                         (keyless)
remote ->  http://taichi:4096/v1   Authorization: Bearer <key>
```

- Keys: one per line in `~/.config/llama-swap/api-keys` (mode 0600, **not** in the
  repo). Add/rotate a key then `systemctl --user restart llama-swap-auth`.
  Generate: `python3 -c "import secrets;print('sk-'+secrets.token_urlsafe(36))"`.
- `GET /health` is exempt (for LAN monitors); everything else needs the key.
- Service: `~/.config/systemd/user/llama-swap-auth.service` (`After=llama-swap`).
  Rebuild the binary with `make proxy`.
- Remote clients: point any OpenAI client at `http://taichi:4096/v1`, send the
  key, request a model by name. For opencode set `LLAMASWAP_URL=http://taichi:4096`
  and `LLAMASWAP_API_KEY=<key>`; for pi set the provider `baseUrl`/`apiKey`.

## llama.cpp: one source of truth, built from source

There is exactly one source tree and one build pipeline. Do not install distro
packages (`llama.cpp-hip`, etc.) or drop stray binaries in `~/.local/bin` -- they
shadow `$PATH` and cause version skew.

```
~/src/llama.cpp                      shallow clone of ggml-org/master (THE source)
        |  ~/dev/my-llama/build-backends.sh   (pulls latest, one pinned commit)
        v
~/opt/llama.cpp-vulkan   (RADV, gfx1201)   <-- SERVING backend (config.header.yaml macro)
~/opt/llama.cpp-hip      (ROCm/HIP)         built + isolation-checked, kept for ad-hoc use
~/opt/llama.cpp-hip-wmma (ROCm + rocWMMA)   built, but no gain over hip here
~/opt/llama.cpp-COMMIT                       records the built commit for all three
```

`build-backends.sh` builds all three from the *same* commit with
`-DCMAKE_INSTALL_RPATH='$ORIGIN/../lib'` and asserts each binary's ggml/llama libs
resolve to its own prefix (isolation check), so a backend comparison is real and
not version skew. `~/.local/bin/{llama-server,llama-cli,...}` are **symlinks** into
the serving backend (`$ORIGIN` resolves through the symlink, so `../lib` still
works); re-point them to switch the PATH default.

**Backend choice is measured, not assumed** (`results/speed/backend-sweep.md`, all
three at commit 169e4a7 on cold-fusion): **vulkan wins** -- TG ~27-31% faster than
hip at every depth, and PP faster at any depth >=32k (hip only leads PP at a
trivial 4k). hip-wmma is within noise of hip. Re-run `bench/speed/backend-sweep.sh`
(with `llama-swap` stopped) after a major llama.cpp bump to re-confirm.

## Inventory

| name | ctx | KV | file | notes |
|---|---|---|---|---|
| `fable-fusion` | 262144 | f16 K / **q8_0 V** | IQ4_NL 17.0G | default (preloaded at startup); vision (mmproj) |
| `kat-apex` | 262144 | f16 / f16 | UD-Q5_K_XL 25.5G | grafted MTP head |
| `kat-q5km` | 262144 | f16 / f16 | Q5_K_M 23.3G | no MTP; roomiest |
| `qwopus-mtp` | 262144 | f16 / f16 | Q5_K_M 23.6G | native MTP; fastest TG |
| `ornith-mxfp4` | 262144 | f16 / f16 | MXFP4 20.5G | best PP |
| `ornith-q5kxl` | 262144 | f16 / f16 | UD-Q5_K_XL 24.7G | quality comparison |

## Locked sampling settings, and why

Sampling is **per-model**, because the vendors genuinely disagree. Copying one
model's settings to another is a real error, not a harmless default.

### fable-fusion (Qwen3.6-27B derivative)

```
--cache-type-v q8_0 --spec-type draft-mtp --reasoning-preserve
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
```

This is Qwen3.6's documented **"thinking mode for general tasks"** preset. It is
identical to their "precise coding tasks" preset except temperature (1.0 vs 0.6),
and it matches the config Qwen used for their own SWE-bench Verified runs
(temp 1.0, top_p 0.95).

- `presence_penalty` and `repeat_penalty` stay at llama.cpp's defaults (0.0 and
  1.0), which already match Qwen's spec. DavidAU additionally warns that raising
  rep-pen degrades MTP, so leave it off.
- `--min-p 0.0` is **not** redundant. llama.cpp defaults min_p to **0.05**, which
  silently truncates the tail on every token. Qwen specifies 0.0.
- `--cache-type-v q8_0` is what makes 262144 reachable at all -- see below.

### kat-apex / kat-q5km (KAT-Coder-V2.5-Dev)

```
--reasoning-preserve
--temp 1.0 --top-p 0.95 --top-k 20
--presence-penalty 1.5 --repeat-last-n -1
(kat-apex only: --spec-type draft-mtp)
```

Kwaipilot's published spec. Note the **direct conflict with fable**: KAT wants
`presence_penalty 1.5`, Qwen wants `0.0`. KAT's RL training specifically added
penalties for repeated content and excessive parallel tool calls, so the penalty
is load-bearing for that model and harmful to assume for others.

`--repeat-last-n -1` is required for correctness: llama.cpp defaults the penalty
window to **64 tokens**, whereas OpenAI's `presence_penalty` semantics apply over
the whole context. Without it, `--presence-penalty 1.5` is nearly a no-op.

Suspected cost: ~20% TG, since the penalty sampler then scans the full context per
token. Unconfirmed -- isolating it is a ~15 minute test.

### qwopus-mtp

llama.cpp defaults, deliberately. No vendor sampling spec was published for this
merge. Do not copy KAT's or Qwen's settings onto it without evidence.

## `--reasoning-preserve`: check this on every new reasoning model

Both KAT and fable ship chat templates whose **default path strips `<think>`
blocks from every assistant turn before the last user query**. Both models were
explicitly trained to use historical reasoning. Neither had it enabled.

llama-server tells you at startup and it is easy to miss:

```
init: chat template supports preserving reasoning, consider enabling it via --reasoning-preserve
```

Vendors state it reduces total token consumption by avoiding redundant
re-reasoning, so it relieves context pressure as well as improving quality. When
adding any new model, grep its template:

```bash
# see bench/quality/swebench/ for the full gguf-py probe pattern
uv run --with numpy python -c "...GGUFReader..." | grep preserve_thinking
```

## VRAM and context ceilings

KV cost per token is set by **cached layers x KV heads x head_dim**, and is
completely independent of weight quantisation. Changing the weight quant changes
the *base* footprint, and therefore the max context that fits, but never the
KV bytes per token.

| model | cached layers | KV heads | head_dim | KiB/token (f16) |
|---|---|---|---|---|
| KAT / Qwopus / Ornith (A3B hybrid) | 10 (+1 MTP) | 2 | 256 | 21.0 (24.0 with MTP) |
| fable (Qwen3.6-27B hybrid) | 16 (+1 MTP) | 4 | 256 | 65.0 (68.4 with MTP) |

Both are hybrid architectures -- most layers are linear-attention (Gated DeltaNet)
and cache nothing. fable is expensive because of 4 KV heads at head_dim 256, not
because of its 64 layers.

**Consequence:** fable at f16/f16 cannot reach 262144 (needs ~34.7 GB of 32.6 GB).
`f16 K + q8_0 V` halves the V half and gets there at 31.8-32.1 GB. Measured cost is
**9-17% PP above 64k depth, ~0% TG** (`results/speed/fable-kv.tsv`). K is left
at f16 because every query is scored against it; V is only the weighted sum
afterwards and is the more forgiving half to quantise.

**Headroom is allocated up front.** llama.cpp sizes the KV cache and compute graph
at load time for worst-case `n_ctx`/`n_batch`, so VRAM does *not* grow as context
fills. Verified on kat-apex: peak grew **33 MiB** from load to 246k depth, with no
fragmentation drift over repeated deep requests
(`results/context/soak-kat-apex-262144.tsv`).

So a few hundred MiB of headroom is genuinely usable. The remaining risk is purely
**external**: GPU0 is the VGA device, so anything else allocating there can OOM a
run. Keep it dedicated.

## Operations

### After a reboot

```bash
llm status            # is llama-swap up? what's loaded?
```

The default model is preloaded automatically (`hooks.on_startup.preload`), and
any request for another model swaps to it on demand -- no manual load needed.
`llm use <name>` only pre-warms.

Two caveats:

- **`Linger=no`**, so the user service starts on *login*, not at boot. For headless
  start: `sudo loginctl enable-linger brian`.
- `fable-fusion` is flagged `default: true`, so it is the one preloaded at
  startup. Move the `default` flag in `registry.json` to change that.

### Switching

Just request a different model by name -- llama-swap swaps to it. To pre-warm:

```bash
llm ls                # inventory (marks what's loaded)
llm use qwopus-mtp    # optional: load it now (~11s)
llm status            # confirm + VRAM
```

### Changing settings

```bash
$EDITOR config/registry.json    # edit extra_args / ctx
llm render (or: make render)
systemctl --user restart llama-swap
llm use <name>                                    # pre-warm to load it now
pgrep -a llama-server                             # ALWAYS verify flags took effect
```

That last line matters. `extra_args` are appended *after* the `${common}` macro,
so a later flag overrides an earlier one (this is how `--cache-type-v q8_0` beats
the macro's `f16`). It works, but it is order-dependent and worth confirming.

## The one unavoidable duplication

**Request parameters override server flags.** Any OpenAI-compatible client that
sends `temperature` wins over `--temp`. So benchmark harnesses must repeat the
sampling spec:

```bash
SAMPLING_ARGS="-c model.model_kwargs.temperature=1.0 \
               -c model.model_kwargs.top_p=0.95 \
               -c model.model_kwargs.presence_penalty=1.5"
```

`top_k` and `min_p` are not OpenAI fields, so they only ever come from the server
side. **If you change sampling in `registry.json`, change it in the harness too**,
or the benchmark silently measures a different configuration than the one you
serve. This exact mismatch invalidated the first KAT SWE-bench run.

## Running the benchmarks

All under `bench/`. Results land in `results/`.

```bash
# throughput vs depth, model x spec matrix
bench/speed/depth-sweep.sh          # 3 models x {none,draft-mtp} x 3 depths x 3 reps
bench/speed/kv-sweep.sh            # f16 vs q8_0-V arms for fable

# capability gates before spending hours on a model
bench/gates/gates.sh              # load / toolcall(22) / reasoning detection
bench/gates/mtp-engage.sh             # does --spec-type draft-mtp actually draft?

# context ceilings
bench/context/ctx-probe.sh          # KiB/token via VRAM delta, then verify+stretch
bench/context/soak-ctx.sh # peak VRAM at depth, fragmentation probe

# quality
bench/quality/toolcall.py  --url http://127.0.0.1:8080/v1 --model cold-fusion
bench/quality/niah.py      --url http://127.0.0.1:8080/v1 --tokenize http://127.0.0.1:8080/tokenize
```

### SWE-bench

```bash
llm use kat-apex
bench/swebench/run-ordered.sh kat-apex 300 8080   # <model> <deadline-min> <port>
# then score:
cd bench/quality/sweresults/kat-apex
../../.venv/bin/python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path preds.json --max_workers 3 \
    --run_id pair-kat --cache_level env
```

`subset60.txt` is a seeded (20260727) stratified sample of 59 instances by
SWE-bench Verified `difficulty`. Keep it fixed so runs stay paired.

## Traps discovered the hard way

1. **`pkill -f llama-server` kills the calling shell**, because the pattern matches
   the shell's own command line. Use `pkill -x llama-server` (process name only).
   Scripts are safe; ad-hoc commands are not.
2. **`mini-extra swebench -c ...` drops the default config.** You must pass
   `-c swebench.yaml` first or every instance dies with a `ValidationError` on
   missing `system_template`.
3. **`MSWEA_DOCKER_EXECUTABLE=podman`** -- the docker environment shells out to a
   literal `docker` binary and ignores `DOCKER_HOST`.
4. **`MSWEA_COST_TRACKING=ignore_errors`** -- litellm raises on unknown local model
   names rather than reporting zero cost.
5. **`mini-extra --filter` processes in dataset order**, not the order of your
   sample file. Truncating at a deadline gives a repo-clustered subset. Use
   `run-ordered.sh`, which drives one instance per invocation.
6. **`n_layer_nextn` is only logged for `BAILINGMOE2`.** Grepping the load log to
   detect MTP always fails on `qwen35`/`qwen35moe`. Confirm MTP behaviourally
   (`draft_n > 0`) or by inspecting GGUF tensors.
7. **`llama.cpp` computes `prompt_per_second` over tokens actually evaluated**, not
   `usage.prompt_tokens`. Prompt caching therefore does not inflate reported PP --
   an earlier concern about `bench-quants.tsv` being optimistic was wrong.
