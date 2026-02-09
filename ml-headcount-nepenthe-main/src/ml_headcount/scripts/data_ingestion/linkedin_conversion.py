
import pandas as pd
import json

def convert_linkedin_json_to_csv(data):
    """Convert LinkedIn JSON data to CSV format
    
    Args:
        data (dict or list): LinkedIn JSON data (can be dict with 'profiles' key or list of profiles)
        
    Returns:
        pd.DataFrame: DataFrame with public_identifier, profile_summary, company_id, and company_name columns
    """
    # 3. Parse JSON (standard or NDJSON)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            lines = [line for line in data.splitlines() if line.strip()]
            data = [json.loads(line) for line in lines]

    # 4. Drill down to your list of profile records
    if isinstance(data, dict) and "profiles" in data and isinstance(data["profiles"], list):
        records = data["profiles"]
    elif isinstance(data, dict):
        records = list(data.values())
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Unexpected JSON structure")

    rows = []
    for entry in records:
        public_id = entry.get('public_identifier', '')
        # Don't add LinkedIn URL prefix - keep it simple
        
        # Extract company information from current_company
        current_company = entry.get('current_company') or {}
        if isinstance(current_company, dict):
            company_id = current_company.get('company_id', '')
            company_name = current_company.get('name', '')
        else:
            company_id = ''
            company_name = ''

        # ensure profile_data is a dict
        pdict = entry.get('profile_data') or {}
        if not isinstance(pdict, dict):
            pdict = {}

        # --- experiences: company, title, description ---
        exp_pieces = []
        for exp in pdict.get('experiences') or []:
            if not isinstance(exp, dict):
                continue
            comp  = (exp.get('company')       or '').strip()
            title = (exp.get('title')         or '').strip()
            desc  = (exp.get('description')   or '').strip()
            if comp or title or desc:
                exp_pieces.append(f"{comp} ({title}): {desc}")
        exp_str = " | ".join(exp_pieces)

        # --- certifications: authority, name ---
        cert_pieces = []
        for cert in pdict.get('certifications') or []:
            if not isinstance(cert, dict):
                continue
            auth = (cert.get('authority') or '').strip()
            name = (cert.get('name')      or '').strip()
            if auth or name:
                cert_pieces.append(f"{auth}: {name}")
        cert_str = " | ".join(cert_pieces)

        # --- education: field_of_study, degree_name, school ---
        edu_pieces = []
        for edu in pdict.get('education') or []:
            if not isinstance(edu, dict):
                continue
            field  = (edu.get('field_of_study') or '').strip()
            degree = (edu.get('degree_name')    or '').strip()
            school = (edu.get('school')         or '').strip()
            if field or degree or school:
                edu_pieces.append(f"{field} – {degree} @ {school}")
        edu_str = " | ".join(edu_pieces)

        # --- summary: raw string ---
        summary_str = (pdict.get('summary') or '').strip()

        # combine into one column
        profile_summary = " || ".join([
            f"Experiences: {exp_str}",
            f"Certifications: {cert_str}",
            f"Education: {edu_str}",
            f"Summary: {summary_str}"
        ])

        rows.append({
            'public_identifier': public_id,
            'profile_summary': profile_summary,
            'company_id': company_id,
            'company_name': company_name
        })

    # 6. Create DataFrame
    df = pd.DataFrame(rows)
    print(f"Converted {len(df)} LinkedIn profiles to DataFrame")
    
    return df
