#!/bin/sh
set -o errexit   # abort on nonzero exitstatus
set -o nounset   # abort on unbound variable
set -o pipefail  # don't hide errors within pipes

# Script to sort companies by employees_in_linkedin count
# Input: data/companies_employees.tsv
# Output: data/companies_sorted_by_linkedin.tsv

INPUT_FILE="data/companies_employees.tsv"
OUTPUT_FILE="data/companies_sorted_by_linkedin.tsv"

# Check if input file exists
if [ ! -f "${INPUT_FILE}" ]; then
    echo "Error: File '${INPUT_FILE}' not found" >&2
    exit 1
fi

# Sort by the 4th column (employees_in_linkedin) numerically in descending order
# Keep header at the top
{
    head -n 1 "${INPUT_FILE}"
    tail -n +2 "${INPUT_FILE}" | sort -t$'\t' -k4 -rn
} > "${OUTPUT_FILE}"

# Display summary
echo "Sorted companies written to: ${OUTPUT_FILE}"
echo ""
echo "Top 10 companies by LinkedIn employees:"
echo "----------------------------------------"
head -n 1 "${OUTPUT_FILE}" | awk -F'\t' '{printf "%-40s %s\n", $1, $4}'
tail -n +2 "${OUTPUT_FILE}" | head -n 10 | awk -F'\t' '{printf "%-40s %s\n", $1, $4}'