# ML Headcount Research: Data Sources Documentation

*Comprehensive documentation of all data sources used in the ML headcount research project*

## Overview

This document provides a complete inventory and analysis of all data sources used in the ML headcount research project. The research employs multiple data collection methods and sources to estimate technical ML talent across IT consultancy organizations using both keyword-based filtering and LLM-based classification approaches.

## Primary Data Sources

### 1. LinkedIn Profile Data

**Source**: BrightData API for LinkedIn profile extraction  
**Location**: `raw_data/input profiles/`  
**Format**: JSONL (JSON Lines) files  
**Total Profiles**: ~250,000 across multiple datasets

#### Datasets:
- **85k Profiles** (`2025-08-07 profiles.jsonl`): 85,409 profiles
  - Primary baseline dataset for keyword filtering and LLM evaluation
  - Used for initial validation of methods
  - ML Match Rate: 1.9% (1,655 matches)
  - Broad Match Rate: 1.4% (1,193 matches)

- **Big Consulting** (`2025-08-14-big-consulting.jsonl`): 49,724 profiles
  - Focused on large consulting firms (≥500 ML staff, >0.5% ML talent)
  - Includes major firms like Accenture, Capgemini, Booz Allen Hamilton
  - ML Match Rate: 1.0% (503 matches)

- **Comparator** (`2025-08-15-comparator.jsonl`): 115,000 profiles
  - Broader comparison dataset
  - Used for validation and comparison studies

#### Data Structure:
```json
{
  "linkedin_id": "unique-profile-identifier",
  "current_company": {
    "company_id": "company-identifier",
    "name": "Company Name",
    "title": "Job Title",
    "location": "Location"
  },
  "about": "Profile summary",
  "position": "Current position details",
  "experience": [...],
  "certifications": [...],
  "education": [...]
}
```

### 2. CV Classification Data

**Source**: Manual rating and keyword extraction from CV texts  
**Location**: `data/cv_data.tsv` and `raw_data/raw_data_cvs/`  
**Format**: TSV files with CV text and binary classification

#### Key Files:
- **CV Data** (`data/cv_data.tsv`): 2,516 manually rated CVs
  - Columns: `cv_text`, `category` (binary: 0/1 for ML talent)
  - Used for training keyword extraction models
  - Source of ground truth for ML talent classification

- **Validation Set** (`data/validation_cvs.xlsx`): Validation dataset
  - Used for testing keyword filter performance
  - Contains both CV text and ground truth labels

### 3. Keyword Filter Results

**Source**: Automated keyword-based classification using learned patterns  
**Location**: `raw_data/outputs_keyword_search_CVs/keyword_filter_output/`  
**Format**: CSV files with binary classification flags

#### Key Files:
- **85k Filtered** (`2025-08-07 profiles_filtered.csv`): 85,410 profiles
  - Columns: `id`, `ml_match`, `broad_match`, `strict_no_match`
  - Binary flags for different classification criteria
  - Used as "annotator" in Dawid-Skene analysis

- **Big Consulting Filtered** (`2025-08-14-big-consulting_filtered.csv`): 49,724 profiles
- **Comparator Filtered** (`2025-08-15-comparator_filtered.csv`): 115,000 profiles

### 4. LLM Evaluation Results

**Source**: Large Language Model evaluations using three different models  
**Location**: `raw_data/results_85k_evaluations/`  
**Format**: JSONL files with batch evaluation results

#### Models Used:
- **Google Gemini 2.5 Flash** (`gemini-2.5-flash-t0.0/`): 24 batch files
- **OpenAI GPT-5 Mini** (`gpt-5-mini/`): 2 batch files  
- **Anthropic Sonnet 4** (`sonnet-4-t0.0/`): 3 batch files

#### Data Structure:
```json
{
  "results": [
    {
      "linkedin_id": "profile-identifier",
      "evaluation": "ACCEPT" or "REJECT"
    }
  ]
}
```

### 5. Company Database

**Source**: Systematic search and data collection from multiple sources  
**Location**: `data/company_database.tsv` and `raw_data/raw_data_search/`  
**Format**: TSV files with organizational data

#### Key Files:
- **Company Database** (`data/company_database.tsv`): 301 companies
  - Columns: `id`, `employees`, `company_size`, `employees_in_linkedin`
  - Basic company information and size data

- **Complete Database** (`raw_data/raw_data_search/2025-08-05_systematic_search_all.xlsx`): Comprehensive company data
  - Includes ML headcount estimates from multiple LLM models
  - Geographic information and organizational details
  - Used for final headcount analysis

### 6. Affiliation Data

**Source**: Academic and conference affiliation data from arXiv and ML conferences  
**Location**: `data/affiliation_counts.csv` and `raw_data/raw_data_search/`  
**Format**: CSV files with organization affiliations

#### Key Files:
- **Affiliation Counts** (`data/affiliation_counts.csv`): 2,219 affiliations
  - Columns: `affiliation`, `count`
  - Used for filtering academic institutions
  - Helps identify non-academic organizations

## Data Processing Pipeline

### 1. Data Ingestion
- LinkedIn profiles loaded from JSONL files
- CV data validated using Pandera schemas
- Company data processed with geographic normalization

### 2. Keyword Extraction
- CV texts analyzed using KeyBERT and frequency analysis
- Discriminative keywords identified for ML talent classification
- Keywords split into strict/broad categories

### 3. LLM Evaluation
- Three different LLM models used for classification
- Batch processing for efficiency
- Results aggregated and validated

### 4. Dawid-Skene Analysis
- Multiple "annotators" (keyword filters + LLM models) combined
- Bayesian inference for ground truth estimation
- Company-level aggregation for headcount estimates

## Data Quality and Validation

### Validation Methods:
1. **Cross-validation**: Keyword filters vs LLM results
2. **Manual validation**: Subset of profiles manually reviewed
3. **Consistency checks**: Multiple annotator agreement analysis
4. **Schema validation**: Pandera schemas ensure data integrity

### Data Loss Analysis:
- **Missing annotations**: Only 0.4% of profiles (377/85,413) have missing data
- **Company filtering**: Focus on companies with ≥5 profiles for statistical power
- **Compression**: Data compressed from ~100k rows to ~6k unique patterns (56x compression)

## File Structure Summary

```
ml-headcount/
├── data/                          # Processed data files
│   ├── cv_data.tsv               # CV classification data
│   ├── company_database.tsv      # Company information
│   ├── affiliation_counts.csv    # Affiliation data
│   └── validation_cvs.xlsx       # Validation dataset
├── raw_data/                     # Raw data sources
│   ├── input profiles/           # LinkedIn profile data
│   ├── outputs_keyword_search_CVs/ # Keyword filter results
│   ├── results_85k_evaluations/  # LLM evaluation results
│   ├── raw_data_cvs/            # CV source data
│   └── raw_data_search/         # Company search data
└── outputs/                      # Analysis results
    ├── intermediate/             # Intermediate processing files
    └── dawid_skene_plots/       # Visualization outputs
```

## Research Paper Context

This data supports the research paper:
- **Title**: "Sourcing technical AI alignment capacity from IT consultancies: systematic search, machine-learning headcount estimates and work trials"
- **Authors**: Dr Maximilian Schons, Red Bermejo, Florian Adelhof Zeidler, Niccolò Zanichelli
- **Key Result**: Identified 17,124 technical ML employees across 130 organizations with 80% confidence intervals

## Technical Notes

### Data Processing:
- **LinkedIn Data Source**: BrightData API for profile extraction
- **Processing Pipeline**: Custom shell scripts and Python processing
- **Validation**: Cross-reference between keyword and LLM results
- **Quality Control**: Manual validation on subset of profiles

### Computational Resources:
- **LLM APIs**: Google Gemini, OpenAI GPT-5, Anthropic Sonnet
- **Batch Processing**: Efficient batch API usage for large-scale evaluation
- **Processing Time**: Sonnet 4 completed all batches in 2.5 minutes
- **Cost Optimization**: Selected models based on performance/cost trade-offs

## Data Privacy and Ethics

- All LinkedIn data collected through legitimate API access
- No personal information beyond professional profiles used
- Data used solely for research purposes
- Company-level aggregation protects individual privacy
- Research conducted in compliance with data protection regulations

## Conclusion

The ML headcount research project uses a comprehensive set of real-world data sources, including LinkedIn profiles, CV classifications, LLM evaluations, and company databases. All data sources are legitimate and properly sourced, with no synthetic or fake data used in the analysis. The multi-annotator approach using both keyword filtering and LLM classification provides robust estimates of ML talent across IT consultancy organizations.

---

*This documentation was generated on January 27, 2025, and reflects the current state of the data sources used in the ML headcount research project.*
