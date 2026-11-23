#!/usr/bin/env bash
# Answer each prompt set with the capture server and move its capture files into CAPROOT/<set>.
# Usage: run_capture.sh CAPROOT MODEL PORT CONC set1 [set2 ...]   (sets = data/<set>.jsonl)
set -euo pipefail
CAP=$1; M=$2; PORT=$3; CONC=$4; shift 4
cd "$(dirname "$0")"
flush() { curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
          -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":4}" > /dev/null; sleep 3; }
for S in "$@"; do
  echo "$(date +%T) set $S"
  python3 generate.py "data/$S.jsonl" "data/${S}_gen.jsonl" --url "http://127.0.0.1:$PORT/v1" --model "$M" --conc "$CONC"
  flush; mkdir -p "$CAP/$S"
  find "$CAP/live" -maxdepth 1 -name 'chatcmpl-gen*.pt' -exec mv -t "$CAP/$S" {} +
  rm -f "$CAP/live"/chatcmpl-[!g]*.pt
  echo "$(date +%T) set $S: $(ls "$CAP/$S" | wc -l) captures, $(du -sh "$CAP/$S" | cut -f1)"
done
echo "$(date +%T) ALL DONE"
