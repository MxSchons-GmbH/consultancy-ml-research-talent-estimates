#!/bin/sh
set -o errexit   # abort on nonzero exitstatus
set -o nounset   # abort on unbound variable
set -o pipefail  # don't hide errors within pipes

INPUT_FILE=data/Profiles/validation/validation_input_586.tsv
MODEL=gpt-5-mini-2025-08-07

## OpenAI
# gpt-5-mini-2025-08-07
### prompt 01
elixir evaluate_profiles.exs -p evaluation_prompt_01.md --tsv "$INPUT_FILE" -m "$MODEL" -o tmp/586-prompt-01-OpenAI-01.tsv -b 10 -vv > logs/586-01-gpt-5-mini.log
elixir evaluate_profiles.exs -p evaluation_prompt_01.md --tsv tmp/586-prompt-01-OpenAI-01.tsv -m "$MODEL" -o tmp/586-prompt-01-OpenAI-02.tsv -b 10 -vv >> logs/586-01-gpt-5-mini.log
elixir evaluate_profiles.exs -p evaluation_prompt_01.md --tsv tmp/586-prompt-01-OpenAI-02.tsv -m "$MODEL" -o 586-prompt-01-OpenAI.tsv -b 10 -vv >> logs/586-01-gpt-5-mini.log

### prompt 02
elixir evaluate_profiles.exs -p evaluation_prompt_02.md --tsv "$INPUT_FILE" -m "$MODEL" -o tmp/586-prompt-02-OpenAI-01.tsv -b 10 -vv > logs/586-02-gpt-5-mini.log
elixir evaluate_profiles.exs -p evaluation_prompt_02.md --tsv tmp/586-prompt-02-OpenAI-01.tsv -m "$MODEL" -o tmp/586-prompt-02-OpenAI-02.tsv -b 10 -vv >> logs/586-02-gpt-5-mini.log
elixir evaluate_profiles.exs -p evaluation_prompt_02.md --tsv tmp/586-prompt-02-OpenAI-02.tsv -m "$MODEL" -o 586-prompt-02-OpenAI.tsv -b 10 -vv >> logs/586-02-gpt-5-mini.log