#!/bin/sh
set -o errexit   # abort on nonzero exitstatus
set -o nounset   # abort on unbound variable
set -o pipefail  # don't hide errors within pipes

INPUT_FILE=data/Profiles/2025-08-15-comparator.jsonl.gz
PROMPT_FILE=evaluation_prompt_01.md
MODEL=gpt-5-mini-2025-08-07
LOG_FILE=logs/2025-08-15-batch-gpt-5-mini-115k.log

## OpenAI Batch run
elixir evaluate_profiles.exs -i "$INPUT_FILE" -p "$PROMPT_FILE" -m "$MODEL" -b 10 --start-from 1000 --mode batch -vv > "$LOG_FILE"