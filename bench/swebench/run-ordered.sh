#!/usr/bin/env bash
# Run a model over subset60.txt ONE INSTANCE AT A TIME, in the file's shuffled
# order, until a wall-clock deadline.
#
# WHY NOT batch mode: `mini-extra swebench --filter` processes instances in
# DATASET order, which for SWE-bench Verified is alphabetical by instance_id.
# Truncating that at a time limit yields a repo-clustered sample (all astropy,
# then all django, ...) rather than a random one -- astropy+django would be the
# entire sample and sympy/sphinx would never be reached. Driving one instance
# per invocation lets us honour subset60.txt's seeded shuffle, so stopping at
# any point leaves an unbiased random subsample of the stratified draw.
#
# Cost: ~10s of dataset-load overhead per instance. Worth it.
#
# Already-completed instances are skipped by mini-swe-agent itself, so this is
# resumable and safe to re-run.
#
# Usage: run-ordered.sh <model-tag> <deadline-minutes> [port]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:?model tag}"
BUDGET_MIN="${2:?deadline in minutes}"
PORT="${3:-8099}"
SUBSET="${SUBSET:-$HERE/subset60.txt}"
# mini-swe-agent + swebench harness live in a venv outside git (500MB).
# Create with: make swebench-venv
VENV="${VENV:-$HERE/.venv}"
OUT="${OUT:-$HERE/../../results/swebench/$MODEL}"
ORDER="$OUT/processed-order.txt"

export MSWEA_DOCKER_EXECUTABLE=podman
export MSWEA_COST_TRACKING=ignore_errors
export DOCKER_HOST="unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/podman/podman.sock"
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY="local"

# SAMPLING: request params OVERRIDE llama-server flags, so the vendor spec has
# to be repeated here even though it is already in registry.json. This is the
# one genuinely unavoidable duplication -- an OpenAI-compatible client always
# wins over server defaults. If you change sampling in registry.json, change it
# here too, or the run silently uses different settings from the served model.
#
# Default below is Kwaipilot's published SWE-bench Verified config for KAT.
# For a model with no vendor spec, pass SAMPLING_ARGS="" to inherit the
# server-side defaults from registry.json instead.
SAMPLING_ARGS="${SAMPLING_ARGS-\
-c model.model_kwargs.temperature=1.0 \
-c model.model_kwargs.top_p=0.95 \
-c model.model_kwargs.presence_penalty=1.5}"

curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "no server on :$PORT" >&2; exit 1; }
mkdir -p "$OUT"
DEADLINE=$(( $(date +%s) + BUDGET_MIN * 60 ))

echo "=== $MODEL | deadline ${BUDGET_MIN} min | $(date)"
i=0
while read -r I; do
    [[ -z "$I" ]] && continue
    i=$((i+1))
    now=$(date +%s)
    if (( now >= DEADLINE )); then
        echo "--- deadline reached at instance $i ($I); stopping"
        break
    fi
    if [[ -d "$OUT/$I" ]]; then
        echo "[$i] $I  (already done, skip)"
        echo "$I" >> "$ORDER"
        continue
    fi
    left=$(( (DEADLINE - now) / 60 ))
    printf '[%2d] %-40s %3d min left ... ' "$i" "$I" "$left"
    s=$(date +%s)
    timeout 2700 "$VENV/bin/mini-extra" swebench \
        --subset verified --split test \
        --filter "^${I}$" --workers 1 \
        --model "openai/${MODEL}" \
        --environment-class docker \
        -c swebench.yaml \
        -c model.model_kwargs.api_base="$OPENAI_API_BASE" \
        ${SAMPLING_ARGS} \
        -c model.model_kwargs.max_tokens="${MAX_TOKENS:-8192}" \
        -o "$OUT" >/dev/null 2>&1
    printf 'done in %d min\n' $(( ($(date +%s) - s) / 60 ))
    echo "$I" >> "$ORDER"
    podman ps -q --filter "name=minisweagent-" | xargs -r podman rm -f >/dev/null 2>&1
done < "$SUBSET"

echo "=== agent phase complete: $(ls -d "$OUT"/*/ 2>/dev/null | wc -l) instances | $(date)"
