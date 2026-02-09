#!/bin/sh

# Script to extract employee data from BrightData company NDJSON file
# Input: filtered_company_ids.txt (list of company IDs)
# Data source: BrightData Companies/company_data.ndjson
# Output: companies_employees.tsv

# Check if required files exist
COMPANY_IDS_FILE="data/filtered_company_ids.txt"
NDJSON_FILE="data/BrightData Companies/company_data.ndjson"
OUTPUT_FILE="data/companies_employees.tsv"

if [ ! -f "$COMPANY_IDS_FILE" ]; then
    echo "Error: File '$COMPANY_IDS_FILE' not found" >&2
    exit 1
fi

if [ ! -f "$NDJSON_FILE" ]; then
    echo "Error: File '$NDJSON_FILE' not found" >&2
    exit 1
fi

# Write TSV header
echo -e "id\temployees\tcompany_size\temployees_in_linkedin" > "$OUTPUT_FILE"

# Process each company ID
while IFS= read -r company_id; do
    # Skip empty lines
    [ -z "$company_id" ] && continue
    
    # Search for the company in the NDJSON file and extract fields
    # Using jq to parse JSON and extract the fields we need
    grep -F "\"id\":\"$company_id\"" "$NDJSON_FILE" | head -1 | \
    jq -r 'select(.id == "'"$company_id"'") | 
           [.id, 
            (if .employees then (.employees | length) else 0 end), 
            .company_size // "", 
            .employees_in_linkedin // 0] | 
           @tsv' >> "$OUTPUT_FILE" 2>/dev/null
    
done < "$COMPANY_IDS_FILE"

# Count results
TOTAL=$(tail -n +2 "$OUTPUT_FILE" | wc -l | tr -d ' ')

echo "Employee data extracted to: $OUTPUT_FILE"
echo "Total companies processed: $TOTAL"