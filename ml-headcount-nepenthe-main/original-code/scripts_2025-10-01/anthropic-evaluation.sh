#!/bin/sh
set -o errexit   # abort on nonzero exitstatus
set -o nounset   # abort on unbound variable
set -o pipefail  # don't hide errors within pipes

INPUT_FILE=data/Profiles/validation/validation_input_586.tsv

## Anthropic
### prompt 01
#elixir evaluate_profiles.exs -p evaluation_prompt_01.md --tsv "$INPUT_FILE" --temperature 1.0 -m claude-sonnet-4-20250514 -o tmp/t1.0-586-prompt-01-Anthropic-01.tsv -b 10 -vv > logs/t1.0-586-01-sonnet-4.log
elixir evaluate_profiles.exs -p evaluation_prompt_01.md --tsv tmp/t1.0-586-prompt-01-Anthropic-01.tsv --temperature 1.0 -m claude-sonnet-4-20250514 -o tmp/t1.0-586-prompt-01-Anthropic-02.tsv -b 10 -vv >> logs/t1.0-586-01-sonnet-4.log
elixir evaluate_profiles.exs -p evaluation_prompt_01.md --tsv tmp/t1.0-586-prompt-01-Anthropic-02.tsv --temperature 1.0 -m claude-sonnet-4-20250514 -o t1.0-586-prompt-01-Anthropic.tsv -b 10 -vv >> logs/t1.0-586-01-sonnet-4.log

### prompt 02
elixir evaluate_profiles.exs -p evaluation_prompt_02.md --tsv "$INPUT_FILE" --temperature 1.0 -m claude-sonnet-4-20250514 -o tmp/t1.0-586-prompt-02-Anthropic-01.tsv -b 10 -vv > logs/t1.0-586-02-sonnet-4.log
elixir evaluate_profiles.exs -p evaluation_prompt_02.md --tsv tmp/t1.0-586-prompt-02-Anthropic-01.tsv --temperature 1.0 -m claude-sonnet-4-20250514 -o tmp/t1.0-586-prompt-02-Anthropic-02.tsv -b 10 -vv >> logs/t1.0-586-02-sonnet-4.log
elixir evaluate_profiles.exs -p evaluation_prompt_02.md --tsv tmp/t1.0-586-prompt-02-Anthropic-02.tsv --temperature 1.0 -m claude-sonnet-4-20250514 -o t1.0-586-prompt-02-Anthropic.tsv -b 10 -vv >> logs/t1.0-586-02-sonnet-4.log