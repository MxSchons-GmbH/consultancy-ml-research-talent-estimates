#!/bin/sh
set -o errexit   # abort on nonzero exitstatus
set -o nounset   # abort on unbound variable
set -o pipefail  # don't hide errors within pipes

# Optimized version that pre-processes profiles into a lookup file
# Usage: ./concat_batch_results_optimized.sh <output_file> <input_file1> <input_file2> ...

if [ $# -lt 2 ]; then
    echo "Usage: $0 <output_file> <input_file1> [input_file2] ..." >&2
    echo "Example: $0 all_results.json data/out/sonnet-4/batch_results_*.jsonl" >&2
    exit 1
fi

# Check if profiles file exists
profiles_file="data/Profiles/2025-08-15-comparator.jsonl"
if [ ! -f "${profiles_file}" ]; then
    echo "Error: Profiles file ${profiles_file} not found" >&2
    exit 1
fi

output_file="$1"
shift

# Check if output file already exists and warn user
if [ -f "${output_file}" ]; then
    echo "Warning: Output file ${output_file} already exists and will be overwritten" >&2
fi

# Create temporary files
temp_file=$(mktemp)
enriched_temp_file=$(mktemp)
seen_ids_file=$(mktemp)
lookup_file="profiles_lookup.txt"
trap 'rm -f "${temp_file}" "${enriched_temp_file}" "${seen_ids_file}"' EXIT

total_results=0
duplicates_skipped=0

echo "Extracting results from batch files..." >&2
for input_file in "$@"; do
    if [ ! -f "${input_file}" ]; then
        echo "Warning: File ${input_file} not found, skipping..." >&2
        continue
    fi
    
    echo "Processing ${input_file}..." >&2
    
    # Clean control characters and fix broken JSON with embedded newlines
    cleaned_file=$(mktemp)
    
    # More robust cleaning: join lines that break JSON strings and remove control chars
    # Read entire file and replace embedded newlines in quoted strings
    python3 -c "
import json
import sys
import re

with open('${input_file}', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Remove control characters except newlines and tabs
content = re.sub(r'[\x00-\x08\x0B-\x1F\x7F]', '', content)

# Fix broken JSON strings by joining lines that are clearly part of a linkedin_id
# Pattern: linkedin_id followed by quote, newline, more text, quote
content = re.sub(r'\"([^\"]*)\n([^\"]*)\",', r'\"\1\2\",', content)

with open('${cleaned_file}', 'w', encoding='utf-8') as f:
    f.write(content)
" 2>/dev/null || {
    # Fallback to simpler approach if Python fails
    tr -d '\000-\010\013-\037' < "${input_file}" > "${cleaned_file}"
}
    
    # Extract results from each line and append to temp file
    while IFS= read -r line; do
        if [ -n "${line}" ]; then
            # Extract each result object from the results array
            if echo "${line}" | jq -c '.results[]' >> "${temp_file}" 2>/dev/null; then
                line_count=$(echo "${line}" | jq '.results | length' 2>/dev/null || echo "0")
                total_results=$((total_results + line_count))
            else
                echo "Warning: Failed to parse line in ${input_file}, skipping..." >&2
            fi
        fi
    done < "${cleaned_file}"
    
    rm -f "${cleaned_file}"
done

# Check if we need to rebuild the lookup file
if [ ! -f "${lookup_file}" ] || [ "${profiles_file}" -nt "${lookup_file}" ]; then
    echo "Building optimized lookup table (this may take a few minutes)..." >&2
    # Create a simple key-value lookup file
    jq -r '
      select(.linkedin_id != null) |
      if .current_company_company_id != null then
        "\(.linkedin_id) \(.current_company_company_id)"
      elif .current_company != null and .current_company.company_id != null then
        "\(.linkedin_id) \(.current_company.company_id)"
      else
        empty
      end
    ' "${profiles_file}" > "${lookup_file}"
    echo "Lookup table built with $(wc -l < "${lookup_file}") entries" >&2
else
    echo "Using existing lookup table..." >&2
fi

echo "Enriching results with company data..." >&2

# Process each result and add company_id (with deduplication)
while IFS= read -r result_line; do
    if [ -n "${result_line}" ]; then
        linkedin_id=$(echo "${result_line}" | jq -r '.linkedin_id')
        
        # Skip if we've already seen this linkedin_id
        if rg -q "^${linkedin_id}$" "${seen_ids_file}" 2>/dev/null; then
            duplicates_skipped=$((duplicates_skipped + 1))
            continue
        fi
        
        # Mark this linkedin_id as seen
        echo "${linkedin_id}" >> "${seen_ids_file}"
        
        # Look up company_id directly in the lookup file
        company_id=$(rg "^${linkedin_id} " "${lookup_file}" 2>/dev/null | head -1 | cut -d' ' -f2- || echo "")
        
        # Add company_id to the result object
        if [ -n "${company_id}" ]; then
            echo "${result_line}" | jq -c --arg company_id "${company_id}" '. + {current_company_company_id: $company_id}' >> "${enriched_temp_file}"
        else
            echo "${result_line}" | jq -c '. + {current_company_company_id: null}' >> "${enriched_temp_file}"
        fi
    fi
done < "${temp_file}"

# Build final JSON array from enriched results
echo "[" > "${output_file}"
if [ -s "${enriched_temp_file}" ]; then
    # Add all results with proper comma separation
    sed '$!s/$/,/' "${enriched_temp_file}" >> "${output_file}"
fi
echo "]" >> "${output_file}"

# Clean up temporary files (removed unused file references)

echo "Successfully concatenated results to ${output_file}" >&2
echo "Total results processed: ${total_results}" >&2
echo "Duplicates skipped: ${duplicates_skipped}" >&2
echo "Unique results in output: $((total_results - duplicates_skipped))" >&2
echo "Lookup table saved as ${lookup_file} for future runs" >&2