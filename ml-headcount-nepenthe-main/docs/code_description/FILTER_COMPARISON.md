# Filter Comparison: Paper's 3 Filters vs. Hamilton CV Analysis Pipeline

## Overview

The paper uses **3 simple binary keyword filters** applied to LinkedIn profiles, while the Hamilton `cv_analysis` pipeline uses a **sophisticated ML-based keyword extraction and scoring system** applied to manually-rated CVs.

---

## Paper's Approach: 3 Binary Filters on LinkedIn Profiles

### Purpose
Screen 85k+ LinkedIn profiles to identify ML talent for Dawid-Skene inference

### Implementation
Shell script (`filter_profiles.sh`) with regex matching on raw LinkedIn text

### The 3 Filters

#### 1. **`ml_match`** (ML Selection - Base Filter)
- **Keywords**: 6 terms
  - `machine learning`, `machine-learning`, `ML`
  - `deep learning`, `deep-learning`
  - `reinforcement learning`, `reinforcement-learning`, `RL`
- **Logic**: Binary (0/1) - Does profile mention ANY ML term?
- **Purpose**: Broad net to capture anyone with ML keywords

#### 2. **`broad_match`** (Broad_yes - Research Indicators)
- **Keywords**: 14 terms
  - `augmented generation`, `agent reinforcement`, `mats scholar`, `mats`
  - `research scientist`, `evals`, `interpretability`, `feature engineering`
  - `research intern`, `candidate`, `graduate research assistant`
  - `science institute`, `staff research scientist`, `doctor`
- **Logic**: Binary (0/1) - Does profile mention ANY research term?
- **Purpose**: Identify research-oriented or AI safety profiles

#### 3. **`strict_no_match`** (Strict_no - Business Indicators)
- **Keywords**: 44 terms
  - `certificate`, `programmer`, `council`, `companies`, `capital`
  - `proven track record`, `pilot`, `money`, `specialist`, `chief`
  - `udemy`, `customer`, `management`, `today`, `cross functional`
  - `administrator`, `excellence`, `commerce`, `linkedin`, `leader`
  - `incident`, `tier`, `brand`, `investment`, `hr`, `sites`
  - `offerings`, `prior`, `centers`, `advising`, `certified information`
  - `key responsibilities`, `master data`, `anti`, `deadlines`
  - `physiology`, `carbon`, `impacts`, `certified machine`, `qualification`
- **Logic**: Binary (0/1) - Does profile contain ANY business term? (inverted logic)
- **Purpose**: Filter out generic business profiles and managers

### Output
3 binary columns per profile: `ml_match`, `broad_match`, `strict_no_match`

### Usage in Paper
These 3 filters act as "annotators" in the Dawid-Skene model, combined with LLM evaluations to estimate ground truth ML talent classification.

---

## Hamilton Pipeline's Approach: ML-Based Keyword Extraction & Scoring

### Purpose
Learn discriminative keywords from manually-rated CVs, then apply sophisticated scoring to validation CVs

### Implementation
Multi-stage Hamilton pipeline with KeyBERT, TF-IDF, clustering, and multiple scoring methods

### The Pipeline Stages

#### Stage 1: **Keyword Extraction** (`keyword_extraction_results`)
- **Input**: 585 manually-rated CVs (category 0 vs 1)
- **Method**: KeyBERT with Salesforce SFR-Embedding-Mistral model
- **Parameters**:
  - `batch_size`: 1024
  - `max_seq_length`: 256
  - `top_n`: 30 keywords per CV
  - `ngram_range`: (1, 4) - captures 1-4 word phrases
  - `min_score`: 0.1
  - `max_keywords`: 100
- **Output**: ~16,476 keyword extractions across all CVs

#### Stage 2: **Discriminative Keyword Scoring** (`discriminative_keywords`)
- **Input**: Keyword extraction results + preprocessed CV text
- **Method**: TF-IDF analysis to compute discriminative scores
- **Parameters**:
  - `tfidf_ngram_range`: (1, 4)
  - `tfidf_max_features`: 5000
  - `tfidf_min_df`: 2
  - `tfidf_max_df`: 0.85
- **Scoring Formula**: 
  ```
  discriminative_score = category_frequency / total_frequency
  ```
- **Output**: 200 discriminative keywords (100 per category) with scores 0-1

#### Stage 3: **Keyword List Generation** (`keyword_lists`)
- **Input**: Discriminative keywords CSV
- **Method**: Threshold-based filtering (`split_keywords_into_strict_broad_lists`)
- **Thresholds**:
  - **Frequency**: `total_raw_frequency >= 5`
  - **Broad**: `discriminative_score >= 0.8` AND `raw_category_specificity >= 0.7`
  - **Strict**: `discriminative_score >= 0.9` AND `raw_category_specificity >= 0.8`
- **Output**: 4 keyword lists
  - `strict_yes`: 68 keywords (high-confidence AI safety/research)
  - `strict_no`: 100 keywords (high-confidence industry/business)
  - `broad_yes`: 32 keywords (moderate AI safety/research)
  - `broad_no`: 0 keywords (no moderate industry indicators)

#### Stage 4: **Validation Scoring** (`validation_cvs_scored`)
- **Input**: Validation CVs + 4 keyword lists
- **Method**: 8 different scoring combinations (`score_validation_cvs_with_filters`)
- **Base Requirement**: ALL scoring methods require `COMMON_BASE` match:
  ```python
  COMMON_BASE = ["machine learning", "machine‐learning", "ML", 
                 "deep learning", "deep-learning", 
                 "reinforcement learning", "reinforcement-learning", "RL"]
  ```
- **8 Scoring Methods**:
  1. `strict_yes_strict_no`: strict_yes ∧ ¬strict_no ∧ COMMON_BASE
  2. `strict_yes_broad_no`: strict_yes ∧ ¬broad_no ∧ COMMON_BASE
  3. `broad_yes_broad_no`: broad_yes ∧ ¬broad_no ∧ COMMON_BASE
  4. `broad_yes_strict_no`: broad_yes ∧ ¬strict_no ∧ COMMON_BASE
  5. `strict_yes_only`: strict_yes ∧ COMMON_BASE (ignore NO terms)
  6. `broad_yes_only`: broad_yes ∧ COMMON_BASE (ignore NO terms)
  7. `strict_no_only`: ¬strict_no ∧ COMMON_BASE (ignore YES terms)
  8. `broad_no_only`: ¬broad_no ∧ COMMON_BASE (ignore YES terms)

- **Output**: Validation CVs with 8 binary scoring columns

---

## Key Differences

| Aspect | Paper's 3 Filters | Hamilton CV Analysis |
|--------|-------------------|---------------------|
| **Data Source** | LinkedIn profiles (85k+) | Manually-rated CVs (585) |
| **Keywords** | 64 manually curated (6+14+44) | 200 ML-extracted (100+100) |
| **Method** | Simple regex matching | KeyBERT + TF-IDF + scoring |
| **Filters** | 3 binary filters | 4 keyword lists → 8 scoring methods |
| **Base Requirement** | None (filters independent) | COMMON_BASE (ML terms) required for all |
| **Scoring** | Binary (0/1) per filter | 8 different scoring combinations |
| **Purpose** | Screen profiles for Dawid-Skene | Validate keyword filtering methods |
| **Output** | 3 columns: ml_match, broad_match, strict_no_match | 8 columns: various YES/NO combinations |
| **Validation** | Used as Dawid-Skene annotators | Compared against ground truth labels |

---

## Correspondence Between the Two

### Direct Correspondences

1. **`ml_match` ↔ `COMMON_BASE`**
   - Paper's `ml_match` filter = Hamilton's `COMMON_BASE` requirement
   - Both use the same 6 ML keywords
   - Difference: Paper uses it as a standalone filter; Hamilton uses it as a base requirement for all scoring

2. **`broad_match` ↔ `broad_yes` list**
   - Paper's 14 keywords are a **manually curated subset** of Hamilton's 32 `broad_yes` keywords
   - Both identify research/AI safety indicators
   - Hamilton's list is algorithmically generated and larger

3. **`strict_no_match` ↔ `strict_no` list**
   - Paper's 44 keywords are a **manually curated subset** of Hamilton's 100 `strict_no` keywords
   - Both filter out business/management profiles
   - Hamilton's list is algorithmically generated and larger

### Conceptual Mapping

The paper's 3 filters can be approximated by Hamilton scoring methods:

| Paper Filter Combination | Hamilton Scoring Method | Notes |
|--------------------------|-------------------------|-------|
| `ml_match=1` | Any method with COMMON_BASE=1 | All Hamilton methods require this |
| `ml_match=1 AND broad_match=1` | `broad_yes_only=1` | Closest match |
| `ml_match=1 AND strict_no_match=0` | `strict_no_only=1` | Inverted logic (0→1) |
| `ml_match=1 AND broad_match=1 AND strict_no_match=0` | `broad_yes_strict_no=1` | Best candidate filter |

### Why the Differences?

1. **Paper's filters are production-ready**:
   - Manually curated for interpretability
   - Smaller keyword lists (64 vs 200)
   - Simple binary logic for shell script implementation
   - Used directly on LinkedIn profiles at scale

2. **Hamilton's pipeline is research/validation**:
   - Algorithmically generated for comprehensiveness
   - Larger keyword lists (200 total)
   - Multiple scoring methods to compare performance
   - Used on manually-rated CVs to validate methods

3. **Different stages of the workflow**:
   - **Paper's filters**: Applied to 85k LinkedIn profiles → Dawid-Skene inference
   - **Hamilton pipeline**: Applied to 585 CVs → Learn discriminative keywords → Validate on separate validation set

---

## Conclusion

The Hamilton `cv_analysis` pipeline is **NOT a direct implementation** of the paper's 3 filters. Instead:

- **Paper's filters**: Simple, production-ready screening tool for LinkedIn profiles
- **Hamilton pipeline**: Sophisticated ML-based keyword learning and validation system

The Hamilton pipeline **generates the knowledge** that informed the paper's filter design. The paper then used a **manually curated subset** of the algorithmically-generated keywords for the final production filtering.

To replicate the paper's exact filtering approach, you would need to:
1. Use the exact 64 keywords from the paper (6+14+44)
2. Apply simple regex matching (like `filter_profiles.sh`)
3. Generate 3 binary columns: `ml_match`, `broad_match`, `strict_no_match`
4. Use these as Dawid-Skene annotators alongside LLM evaluations

The current Hamilton pipeline serves a different purpose: **learning and validating** the discriminative keywords that inform filter design.
