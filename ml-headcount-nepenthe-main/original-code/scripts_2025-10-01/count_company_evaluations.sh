#!/bin/sh
set -o errexit   # abort on nonzero exitstatus
set -o nounset   # abort on unbound variable
set -o pipefail  # don't hide errors within pipes

# Script to count ACCEPT and REJECT evaluations per company
# Usage: ./count_company_evaluations.sh <input_json> <output_json>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_json> <output_json>" >&2
    echo "Example: $0 results.json company_stats.json" >&2
    exit 1
fi

input_file="$1"
output_file="$2"

# Check if input file exists
if [ ! -f "${input_file}" ]; then
    echo "Error: Input file ${input_file} not found" >&2
    exit 1
fi

# Create temporary files
temp_file=$(mktemp)
counts_file=$(mktemp)
trap 'rm -f "${temp_file}" "${counts_file}"' EXIT

echo "Processing evaluations..." >&2

# Clean control characters and extract data
tr -d '\000-\010\013-\037' < "${input_file}" | jq -r '.[] | 
  select(.current_company_company_id != null) | 
  "\(.current_company_company_id)|\(.evaluation)"' > "${temp_file}" 2>/dev/null || {
    echo "Error: Failed to parse input JSON" >&2
    exit 1
}

total_entries=$(wc -l < "${temp_file}")
echo "Found ${total_entries} entries with company IDs" >&2

# Count accepts and rejects per company using awk
echo "Counting accepts and rejects per company..." >&2
awk -F'|' '
{
    company = $1
    evaluation = $2
    
    if (evaluation == "ACCEPT") {
        accepts[company]++
    } else if (evaluation == "REJECT") {
        rejects[company]++
    }
    companies[company] = 1
}
END {
    for (company in companies) {
        accept_count = (accepts[company] ? accepts[company] : 0)
        reject_count = (rejects[company] ? rejects[company] : 0)
        print company "|" accept_count "|" reject_count
    }
}
' "${temp_file}" | sort > "${counts_file}"

# Convert to JSON array
echo "Building JSON output..." >&2
echo "[" > "${output_file}"

first=true
while IFS='|' read -r company accepts rejects; do
    if [ "${first}" = true ]; then
        first=false
    else
        echo "," >> "${output_file}"
    fi
    
    # Escape company ID for JSON (handle quotes and backslashes)
    escaped_company=$(echo "${company}" | sed 's/\\/\\\\/g; s/"/\\"/g')
    
    printf '  {"company_id":"%s","accepts":%d,"rejects":%d}' \
        "${escaped_company}" "${accepts}" "${rejects}" >> "${output_file}"
done < "${counts_file}"

echo "" >> "${output_file}"
echo "]" >> "${output_file}"

# Calculate statistics
total_companies=$(wc -l < "${counts_file}")
total_accepts=$(awk -F'|' '{sum += $2} END {print sum}' "${counts_file}")
total_rejects=$(awk -F'|' '{sum += $3} END {print sum}' "${counts_file}")

echo "" >&2
echo "=== Summary ===" >&2
echo "Total companies: ${total_companies}" >&2
echo "Total accepts: ${total_accepts}" >&2
echo "Total rejects: ${total_rejects}" >&2
echo "" >&2
echo "Output written to: ${output_file}" >&2

# Verify the output is valid JSON
if jq empty "${output_file}" 2>/dev/null; then
    echo "✓ Output is valid JSON" >&2
    
    # Show top companies by accepts
    echo "" >&2
    echo "Top 5 companies by accepts:" >&2
    jq -r 'sort_by(.accepts) | reverse | .[0:5] | .[] | "  \(.company_id): \(.accepts) accepts, \(.rejects) rejects"' "${output_file}" >&2
    
    echo "" >&2
    echo "Top 5 companies by total evaluations:" >&2
    jq -r 'map(. + {total: (.accepts + .rejects)}) | sort_by(.total) | reverse | .[0:5] | .[] | "  \(.company_id): \(.total) total (\(.accepts) accepts, \(.rejects) rejects)"' "${output_file}" >&2
else
    echo "✗ Warning: Output may not be valid JSON!" >&2
    exit 1
fi