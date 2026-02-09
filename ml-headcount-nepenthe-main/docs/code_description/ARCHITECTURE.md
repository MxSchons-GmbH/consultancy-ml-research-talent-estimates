# ML Headcount Pipeline Architecture

## Overview

This pipeline implements a comprehensive system for analyzing and estimating ML talent across organizations, combining data extraction, keyword analysis, clustering, and statistical debiasing techniques. The system processes LinkedIn data, CVs, and company information to produce robust estimates of ML headcount with uncertainty quantification.

## Core Pipeline Components

### 1. Data Ingestion & Preprocessing

#### `convert_linkedin_json_to_csv()`
- **Purpose**: Converts LinkedIn profile JSON data to structured CSV format
- **Input**: LinkedIn JSON export with profile data
- **Output**: CSV with `public_identifier` and `profile_summary` columns
- **Key Features**: 
  - Extracts experiences, certifications, education, and summary
  - Combines profile components into searchable text format
  - Handles nested JSON structures and missing data

#### `extract_linkedin_titles_to_csv()`
- **Purpose**: Extracts job titles from LinkedIn profiles
- **Input**: LinkedIn JSON data
- **Output**: Two CSVs - all titles and last titles per profile
- **Use Case**: Title-based filtering and analysis

#### `extract_cv_keyword_frequencies()`
- **Purpose**: Analyzes CV text to extract keyword frequencies
- **Input**: CSV with CV text data
- **Output**: Keyword frequency analysis with n-gram extraction
- **Features**: 
  - 1-2 gram analysis with custom token patterns
  - Stop word filtering
  - Frequency thresholding (>4 occurrences)

### 2. Data Cleaning & Standardization

#### `filter_academic_affiliations()`
- **Purpose**: Filters out academic institutions from affiliation data
- **Input**: CSV with affiliation counts
- **Output**: Non-academic affiliations only
- **Logic**: Uses comprehensive keyword matching for academic institutions

#### `clean_and_split_affiliations_csv()`
- **Purpose**: Cleans and splits comma/slash-separated affiliations
- **Input**: Raw affiliation CSV
- **Output**: Cleaned CSV with one affiliation per row
- **Features**:
  - Whitespace normalization
  - Multi-delimiter splitting (comma and forward slash)
  - Preview functionality for change validation

### 3. Advanced Text Analysis & Clustering

#### `run_keybert_keyword_extraction()`
- **Purpose**: Comprehensive CV analysis with keyword extraction and clustering using KeyBERT
- **Input**: DataFrame with 'category' and 'processed_text' columns
- **Output**: Dictionary containing discriminative keywords, clusters, and keyword data
- **Key Features**:
  - KeyBERT-based keyword extraction using sentence transformers
  - TF-IDF analysis for discriminative scoring
  - HDBSCAN clustering with KMeans fallback
  - UMAP dimensionality reduction for visualization
  - Custom stop word filtering for ML/consulting domains
  - GPU-optimized with FP16 precision for A100 GPUs
  - Batch processing with large batch sizes
  - Custom embedding models (Salesforce/SFR-Embedding-Mistral)

### 4. Keyword Analysis & Filtering

#### `split_keywords_into_strict_broad_lists()`
- **Purpose**: Categorizes keywords into strict/broad and yes/no lists
- **Input**: Keywords CSV with frequency and discriminative scores
- **Output**: Four keyword lists for different filtering strategies
- **Criteria**:
  - Broad: discriminative_score ≥ 0.8 AND specificity ≥ 0.6
  - Strict: discriminative_score ≥ 0.9 AND specificity ≥ 0.8

#### `score_validation_cvs_with_filters()`
- **Purpose**: Applies keyword filters to validation CVs
- **Input**: Validation CVs and keyword lists
- **Output**: Scored CVs with multiple filter combinations
- **Features**:
  - Eight different scoring methods
  - Confusion matrix analysis
  - F1 score calculation

### 5. Geographic & Organizational Analysis

#### `process_headquarters_with_subregions()`
- **Purpose**: Maps company headquarters to UN M49 subregions
- **Input**: TSV with headquarters location data
- **Output**: Enhanced TSV with country and subregion columns
- **Features**:
  - Comprehensive country-to-subregion mapping
  - Location parsing and normalization
  - Success rate reporting

### 6. Statistical Debiasing & Consensus

#### `clean_llm_estimates_outliers()`
- **Purpose**: Removes outlier LLM estimates based on total headcount
- **Input**: Company database with LLM estimates
- **Output**: Cleaned estimates (removes 3x+ outliers)
- **Logic**: Filters estimates that are >3x or <1/3x of total headcount

#### `debias_ml_headcount_estimates()`
- **Purpose**: Core debiasing algorithm for ML headcount estimates
- **Input**: Multiple estimation methods (filters + LLMs)
- **Output**: Debiased consensus with uncertainty quantification
- **Algorithm**:
  - Log-space transformation with epsilon handling
  - Robust median-based bias estimation
  - Per-method bias correction
  - 80% confidence intervals
  - Outlier detection (>3x deviation)

### 7. Visualization & Reporting

#### `generate_geographic_analysis_plots()`
- **Purpose**: Creates geographic distribution visualizations
- **Input**: Multiple TSV files with geographic data
- **Output**: Interactive maps and stacked bar charts
- **Features**:
  - Plotly choropleth maps
  - Source category breakdowns
  - HTML export for web viewing

#### `create_combined_summary_statistics()`
- **Purpose**: Generates comprehensive summary statistics
- **Input**: Multiple organizational datasets
- **Output**: Combined CSV with sectioned statistics
- **Sections**:
  - Total statistics
  - Size-based breakdowns
  - Regional distributions
  - ML talent percentages

#### `plot_debiased_ml_estimates_comparison()`
- **Purpose**: Visualizes debiased estimates across methods
- **Input**: Debiased organizational data
- **Output**: Scatter plots with confidence intervals
- **Features**:
  - Method comparison visualization
  - Uncertainty quantification display
  - Log/linear scale options

#### `create_ml_talent_landscape_plot()`
- **Purpose**: Creates talent landscape visualization
- **Input**: Organizational ML data
- **Output**: Scatter plot with talent categories
- **Categories**:
  - Enterprise (≥500 ML & 0.5%)
  - Mid-Scale (≥50 ML & 1%)
  - Boutique (≥10 ML & 5%)

#### `build_organizations_list_csv()`
- **Purpose**: Creates final organizational summary
- **Input**: Debiased organizational data
- **Output**: Formatted CSV with all key metrics
- **Features**:
  - Individual method estimates
  - Debiased consensus with intervals
  - Percentage calculations
  - Category assignments

## Pipeline Flow

### Phase 1: Data Ingestion
1. **LinkedIn Data**: `convert_linkedin_json_to_csv()` → `extract_linkedin_titles_to_csv()`
2. **CV Data**: `extract_cv_keyword_frequencies()`
3. **Affiliation Data**: `filter_academic_affiliations()` → `clean_and_split_affiliations_csv()`

### Phase 2: Text Analysis
1. **Keyword Extraction**: `run_keybert_keyword_extraction()`
2. **Filter Development**: `split_keywords_into_strict_broad_lists()`
3. **Validation**: `score_validation_cvs_with_filters()`

### Phase 3: Organizational Analysis
1. **Geographic Processing**: `process_headquarters_with_subregions()`
2. **Estimate Cleaning**: `clean_llm_estimates_outliers()`
3. **Debiasing**: `debias_ml_headcount_estimates()`

### Phase 4: Visualization & Reporting
1. **Geographic Analysis**: `generate_geographic_analysis_plots()`
2. **Statistical Summaries**: `create_combined_summary_statistics()`
3. **Method Comparison**: `plot_debiased_ml_estimates_comparison()`
4. **Landscape Analysis**: `create_ml_talent_landscape_plot()`
5. **Final Output**: `build_organizations_list_csv()`

## Key Design Principles

### 1. Robustness
- Multiple estimation methods with consensus building
- Outlier detection and removal
- Uncertainty quantification throughout

### 2. Scalability
- GPU optimization for large-scale text processing
- Batch processing capabilities
- Memory-efficient data structures

### 3. Transparency
- Detailed logging and progress tracking
- Comprehensive validation metrics
- Reproducible statistical methods

### 4. Flexibility
- Configurable thresholds and parameters
- Multiple output formats
- Modular function design

## Uncertainties & Incompatibilities

### 1. Data Dependencies
- **Issue**: Functions assume specific column names and data formats
- **Impact**: May require data preprocessing for different input formats
- **Mitigation**: Implement data validation and auto-detection

### 2. Parameter Sensitivity
- **Issue**: Many hardcoded thresholds (e.g., 3x outlier threshold, 0.8 discriminative score)
- **Impact**: Results may vary significantly with different parameters
- **Mitigation**: Make parameters configurable and document sensitivity

### 3. GPU Dependencies
- **Issue**: Some functions require specific GPU hardware (A100)
- **Impact**: Limited portability and accessibility
- **Mitigation**: Implement CPU fallbacks and smaller model options

### 4. External Dependencies
- **Issue**: Heavy reliance on external libraries (sentence-transformers, plotly, etc.)
- **Impact**: Version compatibility and maintenance challenges
- **Mitigation**: Pin versions and provide alternative implementations

### 5. Memory Requirements
- **Issue**: Large-scale processing may exceed memory limits
- **Impact**: Processing failures on large datasets
- **Mitigation**: Implement chunking and streaming processing

## Integration Points

### 1. Data Flow
- Functions are designed to work sequentially with standardized data formats
- Output of one function typically serves as input to the next

### 2. Configuration
- Many functions use global variables and hardcoded paths
- Would benefit from centralized configuration management

### 3. Error Handling
- Limited error handling and recovery mechanisms
- Would benefit from robust error handling and logging

### 4. Testing
- No visible testing framework or validation mechanisms
- Would benefit from unit tests and integration tests

## Recommendations for Improvement

1. **Parameterization**: Make all thresholds and parameters configurable
2. **Error Handling**: Implement comprehensive error handling and logging
3. **Testing**: Add unit tests and validation frameworks
4. **Documentation**: Add detailed docstrings and usage examples
5. **Modularity**: Break down large functions into smaller, testable components
6. **Configuration**: Implement centralized configuration management
7. **Performance**: Add performance monitoring and optimization
8. **Portability**: Implement CPU fallbacks and cross-platform compatibility
