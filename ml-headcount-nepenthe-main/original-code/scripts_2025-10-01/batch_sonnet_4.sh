#!/bin/sh
set -o errexit   # abort on nonzero exitstatus
set -o nounset   # abort on unbound variable
set -o pipefail  # don't hide errors within pipes

INPUT_FILE=data/Profiles/2025-08-15-comparator.jsonl.gz
PROMPT_FILE=evaluation_prompt_01.md
MODEL=claude-sonnet-4-20250514
LOG_FILE=logs/2025-08-15-batch-sonnet-4-115k.log


## Anthropic Batch run
### prompt 01
elixir evaluate_profiles.exs -i "$INPUT_FILE" -p "$PROMPT_FILE" --temperature 0.0 -m "$MODEL" -b 10 --start-from 1000 --mode batch -vv > "$LOG_FILE"