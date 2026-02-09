#!/bin/sh
set -o errexit   # abort on nonzero exitstatus
set -o nounset   # abort on unbound variable
set -o pipefail  # don't hide errors within pipes

# Script to filter companies by size from companies_employees.tsv
# Excludes companies with "10,001+ employees"
# Input: data/companies_employees.tsv
# Output: data/filtered_companies_by_size.tsv

INPUT_FILE="data/companies_employees.tsv"
OUTPUT_FILE="data/filtered_companies_by_size.tsv"

# Check if input file exists
if [ ! -f "${INPUT_FILE}" ]; then
    echo "Error: File '${INPUT_FILE}' not found" >&2
    exit 1
fi

# Filter companies excluding those with "10,001+ employees"
# Keep the header and filter data rows
{
    head -n 1 "${INPUT_FILE}"
    tail -n +2 "${INPUT_FILE}" | grep -v "10,001+ employees" || true
} > "${OUTPUT_FILE}"

# Count results
TOTAL_COMPANIES=$(tail -n +2 "${INPUT_FILE}" | wc -l | tr -d ' ')
FILTERED_COMPANIES=$(tail -n +2 "${OUTPUT_FILE}" | wc -l | tr -d ' ')
EXCLUDED_COMPANIES=$((TOTAL_COMPANIES - FILTERED_COMPANIES))

echo "Filtered companies written to: ${OUTPUT_FILE}"
echo "Total companies in source: ${TOTAL_COMPANIES}"
echo "Companies smaller than 10,001+ employees: ${FILTERED_COMPANIES}"
echo "Companies excluded (10,001+ employees): ${EXCLUDED_COMPANIES}"