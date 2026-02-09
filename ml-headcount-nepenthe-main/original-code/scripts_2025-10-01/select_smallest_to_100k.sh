#!/bin/sh
set -o errexit   # abort on nonzero exitstatus
set -o nounset   # abort on unbound variable
set -o pipefail  # don't hide errors within pipes

# Script to select smallest companies by LinkedIn employees until sum reaches 100,000
# Input: data/companies_employees.tsv
# Output: data/smallest_companies_100k.tsv

INPUT_FILE="data/companies_employees.tsv"
OUTPUT_FILE="data/smallest_companies_100k.tsv"
TEMP_SORTED="/tmp/sorted_ascending.tsv"

# Check if input file exists
if [ ! -f "${INPUT_FILE}" ]; then
    echo "Error: File '${INPUT_FILE}' not found" >&2
    exit 1
fi

# Sort by employees_in_linkedin ascending (smallest first), excluding header
tail -n +2 "${INPUT_FILE}" | sort -t$'\t' -k4 -n > "${TEMP_SORTED}"

# Process with awk to accumulate companies until reaching 100,000 total
{
    # Print header
    head -n 1 "${INPUT_FILE}"
    
    # Process sorted data and accumulate until 100k
    awk -F'\t' '
    BEGIN {
        total = 0
        target = 100000
        companies = 0
    }
    {
        count = $4
        
        if (total + count <= target) {
            print $0
            total += count
            companies++
        } else if (total < target) {
            print $0
            total += count
            companies++
            exit
        }
    }
    END {
        print "Selected " companies " companies with " total " total LinkedIn profiles" > "/dev/stderr"
    }
    ' "${TEMP_SORTED}"
} > "${OUTPUT_FILE}" 2>&1

# Clean up temp file
rm -f "${TEMP_SORTED}"

# Display summary
echo "Smallest companies (up to 100k total profiles) written to: ${OUTPUT_FILE}"
echo ""

# Count results
SELECTED_COUNT=$(tail -n +2 "${OUTPUT_FILE}" | wc -l | tr -d ' ')
TOTAL_PROFILES=$(tail -n +2 "${OUTPUT_FILE}" | awk -F'\t' '{sum+=$4} END {print sum}')

echo "Summary:"
echo "--------"
echo "Number of companies selected: ${SELECTED_COUNT}"
echo "Total LinkedIn profiles: ${TOTAL_PROFILES}"
echo ""
echo "Sample of selected companies (first 10):"
echo "-----------------------------------------"
head -n 11 "${OUTPUT_FILE}" | awk -F'\t' '{printf "%-40s %-25s %s\n", $1, $3, $4}'