#!/bin/sh
set -o errexit   # abort on nonzero exitstatus
set -o nounset   # abort on unbound variable
set -o pipefail  # don't hide errors within pipes

# Fast version that only processes null entries
# Usage: ./fix_null_company_ids_fast.sh <input_json> <output_json>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_json> <output_json>" >&2
    echo "Example: $0 results.json results_fixed.json" >&2
    exit 1
fi

input_file="$1"
output_file="$2"

# Check if input file exists
if [ ! -f "${input_file}" ]; then
    echo "Error: Input file ${input_file} not found" >&2
    exit 1
fi

# Check if profiles lookup exists
profiles_lookup="profiles_lookup.txt"
if [ ! -f "${profiles_lookup}" ]; then
    echo "Error: Profiles lookup file ${profiles_lookup} not found" >&2
    echo "Please run concat_batch_results_optimized.sh first to generate the lookup file" >&2
    exit 1
fi

# Create temporary files
temp_nulls=$(mktemp)
temp_fixed=$(mktemp)
temp_output=$(mktemp)
trap 'rm -f "${temp_nulls}" "${temp_fixed}" "${temp_output}"' EXIT

echo "Analyzing null company IDs..." >&2

# Statistics counters
total_nulls=0
fixed_trimming=0
fixed_lowercase=0
fixed_urldecode=0
still_missing=0

# First pass: extract only the null entries for processing
echo "Extracting null entries..." >&2
tr -d '\000-\010\013-\037' < "${input_file}" | \
jq -c '.[] | select(.current_company_company_id == null)' > "${temp_nulls}" 2>/dev/null

total_nulls=$(wc -l < "${temp_nulls}")
echo "Found ${total_nulls} null company IDs to process" >&2

# Process each null entry
while IFS= read -r line; do
    if [ -n "${line}" ]; then
        linkedin_id=$(echo "${line}" | jq -r '.linkedin_id')
        
        # Try to fix the LinkedIn ID and lookup
        fixed_id=""
        found_company=""
        fix_reason=""
        
        # First, try trimming spaces
        trimmed_id=$(echo "${linkedin_id}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        if [ "${trimmed_id}" != "${linkedin_id}" ]; then
            # ID had spaces, try lookup with trimmed version (case-insensitive)
            found_company=$(grep -i "^${trimmed_id} " "${profiles_lookup}" 2>/dev/null | head -1 | cut -d' ' -f2- || echo "")
            if [ -n "${found_company}" ]; then
                fixed_id="${trimmed_id}"
                fix_reason="trimmed"
                fixed_trimming=$((fixed_trimming + 1))
            fi
        fi
        
        # Also try lowercase version even without trimming
        if [ -z "${fixed_id}" ]; then
            lowercase_id=$(echo "${trimmed_id}" | tr '[:upper:]' '[:lower:]')
            if [ "${lowercase_id}" != "${trimmed_id}" ]; then
                # ID has uppercase letters, try lowercase lookup
                found_company=$(grep "^${lowercase_id} " "${profiles_lookup}" 2>/dev/null | head -1 | cut -d' ' -f2- || echo "")
                if [ -n "${found_company}" ]; then
                    fixed_id="${lowercase_id}"
                    fix_reason="lowercased"
                    fixed_lowercase=$((fixed_lowercase + 1))
                fi
            fi
        fi
        
        # If not found, try URL decoding
        if [ -z "${fixed_id}" ]; then
            # Check if ID contains URL encoding
            if echo "${linkedin_id}" | grep -q '%'; then
                # URL decode using Python
                decoded_id=$(echo "${linkedin_id}" | python3 -c "import urllib.parse, sys; print(urllib.parse.unquote(sys.stdin.read().strip()))" 2>/dev/null || echo "${linkedin_id}")
                
                # Also trim the decoded ID
                decoded_id=$(echo "${decoded_id}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
                
                if [ "${decoded_id}" != "${linkedin_id}" ]; then
                    found_company=$(grep "^${decoded_id} " "${profiles_lookup}" 2>/dev/null | head -1 | cut -d' ' -f2- || echo "")
                    if [ -n "${found_company}" ]; then
                        fixed_id="${decoded_id}"
                        fix_reason="url-decoded"
                        fixed_urldecode=$((fixed_urldecode + 1))
                    fi
                fi
            fi
        fi
        
        # If we found a fix, save it
        if [ -n "${fixed_id}" ] && [ -n "${found_company}" ]; then
            # Store the LinkedIn ID and company for later replacement
            echo "${linkedin_id}|${found_company}" >> "${temp_fixed}"
            echo "Fixed: '${linkedin_id}' -> '${fixed_id}' (${fix_reason}) -> company: ${found_company}" >&2
        else
            still_missing=$((still_missing + 1))
            echo "Not in profiles: '${linkedin_id}'" >&2
        fi
    fi
done < "${temp_nulls}"

echo "Building fixed output..." >&2

# Second pass: rebuild the complete file with fixes applied
tr -d '\000-\010\013-\037' < "${input_file}" | \
jq -c '.[]' | \
while IFS= read -r line; do
    if [ -n "${line}" ]; then
        linkedin_id=$(echo "${line}" | jq -r '.linkedin_id')
        current_company=$(echo "${line}" | jq -r '.current_company_company_id')
        
        if [ "${current_company}" = "null" ]; then
            # Check if we have a fix for this ID
            fixed_company=$(grep "^${linkedin_id}|" "${temp_fixed}" 2>/dev/null | cut -d'|' -f2- | head -1 || echo "")
            if [ -n "${fixed_company}" ]; then
                # Apply the fix
                echo "${line}" | jq -c --arg company_id "${fixed_company}" '. + {current_company_company_id: $company_id}'
            else
                # Keep as null
                echo "${line}"
            fi
        else
            # Already has a company, keep as is
            echo "${line}"
        fi
    fi
done > "${temp_output}"

# Build final JSON array
echo "[" > "${output_file}"
if [ -s "${temp_output}" ]; then
    sed '$!s/$/,/' "${temp_output}" >> "${output_file}"
fi
echo "]" >> "${output_file}"

# Report statistics
echo "" >&2
echo "=== Fix Summary ===" >&2
echo "Total null company IDs found: ${total_nulls}" >&2
echo "Fixed by trimming spaces: ${fixed_trimming}" >&2
echo "Fixed by lowercasing: ${fixed_lowercase}" >&2
echo "Fixed by URL decoding: ${fixed_urldecode}" >&2
echo "Still missing (not in profiles): ${still_missing}" >&2
echo "" >&2
echo "Output written to: ${output_file}" >&2

# Verify the output is valid JSON
if jq empty "${output_file}" 2>/dev/null; then
    echo "✓ Output is valid JSON" >&2
else
    echo "✗ Warning: Output may not be valid JSON!" >&2
    exit 1
fi