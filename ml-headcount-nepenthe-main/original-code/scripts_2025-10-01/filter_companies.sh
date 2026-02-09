#!/bin/sh

# Script to filter LinkedIn URLs from TSV file where Max or Red gave rating of 1
# Input file: 2025-08-05_systematic_search_all - add claude + personal rating.tsv
# Output files: 
#   - filtered_linkedin_urls.txt (companies)
#   - company_person_profiles.txt (personal profiles)
# Columns: Max rating is column 4, Red rating is column 5, LinkedIn URL is column 25

# Check if file exists
INPUT_FILE="data/2025-08-05_systematic_search_all - Only Proper ML Consultancies.tsv"
COMPANY_FILE="data/filtered_company_ids.txt"
PROFILE_FILE="data/company_person_profiles.txt"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found in current directory" >&2
    exit 1
fi

# Process the file and extract just the public IDs
# Extract all URLs with ratings of 1
ALL_URLS=$(tail -n +2 "$INPUT_FILE" | \
awk -F'\t' '($4 == "1" || $5 == "1") && $25 ~ /linkedin\.com/ {print $25}' | \
grep -v '^$')

# Function to decode URL-encoded strings
urldecode() {
    printf '%b\n' "$(echo "$1" | sed 's/+/ /g; s/%\([0-9A-Fa-f][0-9A-Fa-f]\)/\\x\1/g')"
}

# Extract personal profile IDs (from /in/) and decode them
echo "$ALL_URLS" | \
sed -n 's|.*/in/\([^/?#]*\).*|\1|p' | \
while read -r id; do
    urldecode "$id"
done | \
sort -u > "$PROFILE_FILE"

# Extract company/org IDs (from /company/, /company-beta/, /showcase/, /school/, etc.) and decode them
echo "$ALL_URLS" | \
sed -n -e 's|.*/company/\([^/?#]*\).*|\1|p' \
       -e 's|.*/company-beta/\([^/?#]*\).*|\1|p' \
       -e 's|.*/showcase/\([^/?#]*\).*|\1|p' \
       -e 's|.*/school/\([^/?#]*\).*|\1|p' \
       -e 's|.*/edu/school?id=\([^/?#&]*\).*|\1|p' \
       -e 's|.*/organization-guest/company/\([^/?#]*\).*|\1|p' | \
grep -v '^$' | \
while read -r id; do
    urldecode "$id"
done | \
sort -u > "$COMPANY_FILE"

# Count results for summary
COMPANY_TOTAL=$(wc -l < "$COMPANY_FILE" | tr -d ' ')
PROFILE_TOTAL=$(wc -l < "$PROFILE_FILE" | tr -d ' ')

echo "Company LinkedIn URLs written to: $COMPANY_FILE"
echo "Total company URLs with rating 1 from Max or Red: $COMPANY_TOTAL"
echo ""
echo "Personal profile URLs written to: $PROFILE_FILE"
echo "Total personal profiles with rating 1 from Max or Red: $PROFILE_TOTAL"