#!/bin/sh
set -o errexit   # abort on nonzero exitstatus
set -o nounset   # abort on unbound variable
set -o pipefail  # don't hide errors within pipes

INPUT_FILE=data/Profiles/validation/validation_input_586.tsv

## Gemini t=1.0
### prompt 01
elixir evaluate_profiles.exs -p evaluation_prompt_01.md --tsv "$INPUT_FILE" --temperature 1.0 -m gemini-2.5-flash -o tmp/t1.0-586-prompt-01-Gemini-01.tsv -b 10 -vvv > logs/t1.0-586-01-gemini-2.5-flash.log
elixir evaluate_profiles.exs -p evaluation_prompt_01.md --tsv tmp/t1.0-586-prompt-01-Gemini-01.tsv --temperature 1.0 -m gemini-2.5-flash -o tmp/t1.0-586-prompt-01-Gemini-02.tsv -b 10 -vvv >> logs/t1.0-586-01-gemini-2.5-flash.log
elixir evaluate_profiles.exs -p evaluation_prompt_01.md --tsv tmp/t1.0-586-prompt-01-Gemini-02.tsv --temperature 1.0 -m gemini-2.5-flash -o t1.0-586-prompt-01-Gemini.tsv -b 10 -vvv >> logs/t1.0-586-01-gemini-2.5-flash.log

### prompt 02
elixir evaluate_profiles.exs -p evaluation_prompt_02.md --tsv "$INPUT_FILE" --temperature 1.0 -m gemini-2.5-flash -o tmp/t1.0-586-prompt-02-Gemini-01.tsv -b 10 -vvv > logs/t1.0-586-02-gemini-2.5.log
elixir evaluate_profiles.exs -p evaluation_prompt_02.md --tsv tmp/t1.0-586-prompt-02-Gemini-01.tsv --temperature 1.0 -m gemini-2.5-flash -o tmp/t1.0-586-prompt-02-Gemini-02.tsv -b 10 -vvv >> logs/t1.0-586-02-gemini-2.5-flash.log
elixir evaluate_profiles.exs -p evaluation_prompt_02.md --tsv tmp/t1.0-586-prompt-02-Gemini-02.tsv --temperature 1.0 -m gemini-2.5-flash -o t1.0-586-prompt-02-Gemini.tsv -b 10 -vvv >> logs/t1.0-586-02-gemini-2.5-flash.log