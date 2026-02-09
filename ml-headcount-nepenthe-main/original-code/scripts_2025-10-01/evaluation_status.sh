#!/bin/sh
set -o errexit   # abort on nonzero exitstatus
set -o nounset   # abort on unbound variable
set -o pipefail  # don't hide errors within pipes

# Script to check evaluation status and generate resume commands
# Usage: ./evaluation_status.sh [model]

MODEL="${1:-gemini-2.5-flash-lite}"
PROGRESS_FILE="data/out/.evaluation_progress_${MODEL//\./_}.txt"
INPUT_FILE="data/Profiles/snap_me1d8qy32cqipopsau.jsonl.gz"

echo "=== Evaluation Status for ${MODEL} ===" >&2
echo "" >&2

if [ -f "${PROGRESS_FILE}" ]; then
    echo "Recent progress:" >&2
    tail -5 "${PROGRESS_FILE}" >&2
    echo "" >&2
    
    # Get last processed count
    last_processed=$(tail -1 "${PROGRESS_FILE}" | grep -o 'Profiles processed: [0-9]*' | grep -o '[0-9]*' || echo "0")
    
    # Get total profiles
    total_profiles=$(gunzip -c "${INPUT_FILE}" | wc -l | tr -d ' ')
    
    remaining=$((total_profiles - last_processed))
    
    echo "Progress Summary:" >&2
    echo "- Total profiles: ${total_profiles}" >&2
    echo "- Processed: ${last_processed}" >&2
    echo "- Remaining: ${remaining}" >&2
    echo "" >&2
    
    if [ ${remaining} -gt 0 ]; then
        echo "To resume evaluation:" >&2
        echo "./evaluate_profiles_rate_limited.sh ${MODEL} \"\" ${last_processed}" >&2
    else
        echo "Evaluation is complete!" >&2
    fi
else
    echo "No progress file found for ${MODEL}" >&2
    echo "To start evaluation:" >&2
    echo "./evaluate_profiles_rate_limited.sh ${MODEL}" >&2
fi

echo "" >&2
echo "Available result files:" >&2
ls -la data/profile_evaluations_${MODEL//\./_}_*.jsonl 2>/dev/null | head -5 || echo "No result files found"