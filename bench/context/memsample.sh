#!/usr/bin/env bash
# Sample VRAM and RAM every N seconds (default 30) into a TSV, for depth_bench.py's BENCH_MEMLOG.
#   memsample.sh <out.tsv> [interval_s] [pid_pattern]
# Columns: epoch, vram_used_gib (largest card = the dGPU; card1 is the iGPU), ram_used_gib (MemTotal-MemAvailable),
#          tabby_pss_gib (PSS: shared memfd arena counted once) of every process whose cmdline matches pid_pattern
#          (default: Tabby's python, not a gdb parent) and its children, swap_used_gib (system), tabby_swap_gib, time
out="${1:?usage: $0 <out.tsv> [interval_s] [pid_pattern]}"; iv="${2:-30}"; pat="${3:-^$HOME/dev/exllamav3/.venv/bin/python main\\.py}"
[ -s "$out" ] || printf 'epoch\tvram_used_gib\tram_used_gib\ttabby_pss_gib\tswap_used_gib\ttabby_swap_gib\ttime\n' > "$out"
while :; do
  vram=$(sort -n /sys/class/drm/card*/device/mem_info_vram_used | tail -1)
  read -r ram swap < <(awk '/^MemTotal/{t=$2}/^MemAvailable/{a=$2}/^SwapTotal/{st=$2}/^SwapFree/{sf=$2}END{print (t-a)*1024, (st-sf)*1024}' /proc/meminfo)
  rss=0; tsw=0
  for q in $(for p in $(pgrep -f -- "$pat"); do echo "$p"; pgrep -P "$p"; done | sort -u); do
    read -r r w < <(awk '/^Pss:/{p=$2*1024}/^SwapPss:/{s=$2*1024}END{print p+0, s+0}' /proc/$q/smaps_rollup 2>/dev/null)
    rss=$((rss + ${r:-0})); tsw=$((tsw + ${w:-0}))
  done
  awk -v e="$(date +%s.%N)" -v v="$vram" -v r="$ram" -v t="$rss" -v s="$swap" -v w="$tsw" -v d="$(date +%T)" \
    'BEGIN{g=2^30; printf "%.1f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%s\n", e, v/g, r/g, t/g, s/g, w/g, d}' >> "$out"
  sleep "$iv"
done
