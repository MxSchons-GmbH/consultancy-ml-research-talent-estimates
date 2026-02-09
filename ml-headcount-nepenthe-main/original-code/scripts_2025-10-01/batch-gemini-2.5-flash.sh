#!/bin/sh
set -o errexit   # abort on nonzero exitstatus
set -o nounset   # abort on unbound variable
set -o pipefail  # don't hide errors within pipes

INPUT_FILE=data/Profiles/2025-08-15-comparator.jsonl.gz
PROMPT_FILE=evaluation_prompt_01.md
MODEL=gemini-2.5-flash
LOG_FILE=logs/2025-08-15-batch-gemini-2.5-flash-115k.log

## Gemini Batch run
elixir evaluate_profiles.exs -i "$INPUT_FILE" -p "$PROMPT_FILE" -m "$MODEL" -b 10 --temperature 0.0 --start-from 109250 --mode batch -vv > "$LOG_FILE"