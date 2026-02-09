#!/bin/sh
set -o errexit
set -o nounset
set -o pipefail

input_file="$1"
output_file="${2:-${input_file%.jsonl.gz}_filtered.csv}"

# Print CSV header
echo "id,ml_match,broad_match,strict_no_match" > "${output_file}"

# Process JSONL: extract relevant fields with jq, then filter with awk
zcat "${input_file}" | jq -c '{
    id: (.linkedin_id // .id // "unknown"),
    text: [
        (.about // ""),
        (.summary // ""),
        (.experience[]?.description // ""),
        (.education[]?.description // "")
    ] | join(" ")
}' | awk -F'"text":"' '
BEGIN {
    IGNORECASE = 1
}
{
    ml = 0
    broad = 0
    strict = 1

    # Extract id
    if (match($0, /"id":"([^"]+)"/)) {
        id = substr($0, RSTART+6, RLENGTH-7)
    } else {
        id = "unknown"
    }

    # Extract text field (everything after "text":")
    if (NF > 1) {
        text = $2
        # Remove trailing "}
        sub(/"}$/, "", text)

        # ML filter
        if (match(text, /(machine[\s-]?learning|[^a-zA-Z]ML[^a-zA-Z]|deep[\s-]?learning|reinforcement[\s-]?learning|[^a-zA-Z]RL[^a-zA-Z])/)) ml = 1

        # Broad filter
        if (match(text, /(augmented generation|agent reinforcement|mats scholar|[^a-zA-Z]mats[^a-zA-Z]|research scientist|evals|interpretability|feature engineering|research intern|candidate|graduate research assistant|science institute|staff research scientist|doctor)/)) broad = 1

        # Strict_no filter (NOT logic)
        if (match(text, /(certificate|programmer|council|companies|capital|proven track record|pilot|money|specialist|chief|udemy|track record|customer|management|today|cross functional|administrator|excellence|commerce|linkedin|leader|incident|tier|brand|investment|[^a-zA-Z]hr[^a-zA-Z]|sites|offerings|prior|centers|advising|certified information|key responsibilities|master data|anti|deadlines|physiology|carbon|impacts|certified machine|qualification)/)) strict = 0
    }

    print id "," ml "," broad "," strict
}
' >> "${output_file}"

echo "Filtered results written to: ${output_file}"