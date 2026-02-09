## Input/Output Files Analysis

### **External Input Files (Must be provided by user):**

1. **LinkedIn JSON Export** 
   - **Format**: JSON file with LinkedIn profile data
   - **Structure**: Contains `profiles` array with profile records
   - **Required fields**: `public_identifier`, `profile_data` (with `experiences`, `certifications`, `education`, `summary`)
   - **Used by**: `convert_linkedin_json_to_csv()`, `extract_linkedin_titles_to_csv()`

2. **CV Data File**
   - **Format**: TSV or CSV
   - **Required columns**: `cv_text`, `category`
   - **Purpose**: Contains CV texts with binary classification (yes/no, 0/1)
   - **Used by**: `run_keybert_keyword_extraction()`

3. **Validation CVs File**
   - **Format**: TSV (`validation_cvs.tsv`)
   - **Required columns**: `cv_text`, `category`
   - **Purpose**: Validation dataset for testing keyword filters
   - **Used by**: `score_validation_cvs_with_filters()`

4. **Affiliation Data File**
   - **Format**: CSV (`affiliation_counts.csv`)
   - **Required columns**: `Affiliation`, `Count`
   - **Purpose**: Contains organization affiliations from ML conferences/arXiv
   - **Used by**: `filter_academic_affiliations()`

5. **Company Database File**
   - **Format**: TSV (`company_database.tsv`)
   - **Required columns**: `Headquarters Location`, `Total Headcount`, plus ML estimation columns
   - **Purpose**: Main organizational data with ML headcount estimates
   - **Used by**: `process_headquarters_with_subregions()`, `debias_ml_headcount_estimates()`

### **Intermediate Files (Created by pipeline functions):**

6. **LinkedIn Profile CSV** (`output.csv`)
   - **Columns**: `public_identifier`, `profile_summary`
   - **Created by**: `convert_linkedin_json_to_csv()`

7. **LinkedIn Titles CSV** (`all_titles.csv`, `last_titles.csv`)
   - **Columns**: Job titles extracted from LinkedIn profiles
   - **Created by**: `extract_linkedin_titles_to_csv()`

8. **Keyword Frequencies CSV** (`keyword_frequencies.csv`)
   - **Columns**: Keywords with frequency counts
   - **Created by**: `extract_cv_keyword_frequencies()`

9. **Cleaned Affiliations CSV** (`cleaned_affiliations.csv`)
   - **Columns**: `Affiliation` (non-academic only)
   - **Created by**: `filter_academic_affiliations()`

10. **CV Keyword Clusters JSON** (`cv_keyword_clusters.json`)
    - **Content**: Complete keyword and cluster data
    - **Created by**: `run_keybert_keyword_extraction()`

11. **Discriminative Keywords CSV** (`cv_discriminative_keywords.csv`)
    - **Columns**: Keywords with discriminative scores
    - **Created by**: `run_keybert_keyword_extraction()`

12. **Keywords Lists JSON** (`keywords.json`)
    - **Content**: Keywords categorized into strict/broad and yes/no lists as arrays
    - **Created by**: `split_keywords_into_strict_broad_lists()`

13. **Validation CVs Scored CSV** (`validation_cvs_scored.csv`)
    - **Columns**: CVs with eight different scoring methods
    - **Created by**: `score_validation_cvs_with_filters()`

14. **Company Database with Subregions TSV** (`company_database_with_subregions.tsv`)
    - **Columns**: Original columns plus `Country`, `Subregion`
    - **Created by**: `process_headquarters_with_subregions()`

15. **Cleaned LLM Estimates TSV** (`company_database_with_subregions_cleaned_llm_estimates.tsv`)
    - **Columns**: Original columns with outlier LLM estimates removed
    - **Created by**: `clean_llm_estimates_outliers()`

### **Final Output Files:**

16. **All Organizations Debiased TSV** (`all_orgs_debiased.tsv`)
    - **Columns**: Original columns plus debiased ML estimates and confidence intervals
    - **Created by**: `debias_ml_headcount_estimates()`

17. **Filtered Organization Files**:
    - `orgs_ML.tsv` - Organizations with max headcount and population
    - `orgs_talent_dense.tsv` - Talent-dense organizations
    - `orgs_stage5_work_trial.tsv` - Stage 5 work trial organizations
    - `orgs_stage5_work_trial_recommended.tsv` - Recommended stage 5 organizations
    - `orgs_enterprise_500ml_0p5pct.tsv` - Enterprise organizations
    - `orgs_midscale_50ml_1pct.tsv` - Mid-scale organizations
    - `orgs_boutique_10ml_5pct.tsv` - Boutique organizations

18. **Summary and Analysis Files**:
    - `combined_summary.csv` - Combined summary statistics
    - `organizations_list.csv` - Final formatted organizational summary
    - Geographic analysis HTML files in `analysis_outputs/` folder

## ✅ **Found Files (Matching Requirements)**

### External Input Files
- **CV Data File**: `data/raw_data_cvs/2025-08-05_CV_rated.tsv` ✓
  - Contains `cv_text` and `category` columns as required
- **Validation CVs File**: `data/raw_data_cvs/Validation Set/2025-08-06_validation_set.csv` ✓
  - Contains CVs for validation testing
- **Affiliation Data File**: `data/raw_data_search/2025-08-04_arxiv_kaggle_all_affiliations_cleaned.csv` ✓
  - Contains `affiliation` and `count` columns
- **LinkedIn JSON Export**: `data/raw_data_cvs/2025-08-07 85k profiles.jsonl` ✓
  - 1.1GB file with LinkedIn profiles in JSONL format
  - Contains required fields: `linkedin_id`, `experience`, `certifications`, `education`, `summary`

### Intermediate Files
- **Keyword Frequencies**: `data/outputs_keyword_search_CVs/cv_keywords_FIXED_frequency_analysis (1).csv` ✓
- **CV Keyword Clusters**: `data/outputs_keyword_search_CVs/cv_cluster_frequency_analysis.csv` ✓

## ⚠️ **Partial Matches**

- **Company Database**: `data/raw_data_cvs/2025-08-07 included_companies.tsv` 
  - Contains company info but may be missing required ML estimation columns
  - Needs verification against required schema

## ❌ **Missing Files**

### External Input Files
- None (all external inputs found)

### Intermediate Files (15+ missing)
- LinkedIn Profile CSV (`output.csv`)
- LinkedIn Titles CSV (`all_titles.csv`, `last_titles.csv`)
- Cleaned Affiliations CSV (`cleaned_affiliations.csv`)
- CV Keyword Clusters JSON (`cv_keyword_clusters.json`)
- Discriminative Keywords CSV (`cv_discriminative_keywords.csv`)
- Keywords Lists CSV (`keywords.csv`)
- Validation CVs Scored CSV (`validation_cvs_scored.csv`)
- Company Database with Subregions TSV
- Cleaned LLM Estimates TSV

### Final Output Files (All missing)
- All Organizations Debiased TSV
- All 7 filtered organization files (`orgs_ML.tsv`, `orgs_talent_dense.tsv`, etc.)
- Summary and analysis files (`combined_summary.csv`, `organizations_list.csv`)

## 🔧 **Required Preprocessing**

1. **Company Database**: Verify schema matches requirements, may need additional ML estimation columns
2. **Pipeline Execution**: Run the full pipeline to generate missing intermediate and final output files

## 📝 **Note on LinkedIn Data**

The LinkedIn profile data JSON file (`2025-08-07 85k profiles.jsonl`) is **NOT required** for generating the final organizational outputs. The functions that process LinkedIn data (`convert_linkedin_json_to_csv()` and `extract_linkedin_titles_to_csv()`) create standalone CSV files that are not consumed by any other functions in the main pipeline. The LinkedIn data processing appears to be a separate analytical track that was likely used for exploratory analysis or supplementary insights.

## 📊 **Status Summary**
- **External Inputs**: 5/5 found ✅ (though LinkedIn data is optional)
- **Intermediate Files**: ~2/15+ found ⚠️
- **Final Outputs**: 0/9 found ❌

The project has all required input data but needs pipeline execution to generate the complete set of intermediate and final output files.