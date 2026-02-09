"""
KeyBERT keyword extraction, clustering, and visualization functions

This module contains separated functions for:
1. Keyword extraction using KeyBERT
2. Keyword clustering using HDBSCAN/KMeans
3. Visualization of results

Each function can be used independently in the Hamilton pipeline.
"""

import pandas as pd
import numpy as np
import torch
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS as SKLEARN_STOP_WORDS
from sklearn.cluster import KMeans
import hdbscan
import umap
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CUSTOM STOPWORDS
# ============================================================================
# These stopwords help filter out generic business terms and company names

# Define custom stopwords
CUSTOM_STOPWORDS = {
    # outsourcing / consulting firms
    "accenture","deloitte","capgemini","wipro","cognizant","pwc","ey",
    "tata","consultancy","services","bpo",
    # AI companies
    "google", "deepmind", "openai", "apollo", "far", "palisade", "anthropic",
    "quebec", "berkeley", "edx", "rwth", "aachen",
    "barclays", "kpmg", "cisco", "uber", "dxc", "oracle", "microsoft", "amazon", "meta", "nvidia",
    # generic business titles typical for consultancies
    "consultant","consulting","manager","director","partner","senior",
    "managing","lead","leadership","enterprise","business","operations",
    "strategy","pricing","project","projects",
    "professional","solutions","success","account","client","global",
    "venture","investor","financial","industry",
    "industries","vendor","supplier","networking",
    # geography shortcuts
    "north","america","americas","uk","apac","regional","sector","virginia", "toronto", "london", "singapore",
    # noise / boiler‑plate
    "board","human","impact","region","experiences", "certifications", "education", "senior", "member", "years", "experience"
}

def initialize_keybert_model(
    *,
    model_name: str,
    device: str,
    batch_size: int,
    max_seq_length: int
) -> Tuple[SentenceTransformer, KeyBERT]:
    """
    Initialize KeyBERT model with optimized settings.
    
    Args:
        model_name: Name of the sentence transformer model
        device: Device to run on ('cuda' or 'cpu')
        batch_size: Batch size for embeddings
        max_seq_length: Maximum sequence length
        
    Returns:
        Tuple of (sentence_model, kw_model)
    """
    logger.info(f"Initializing KeyBERT model: {model_name}")
    
    # Optimize for GPU if available
    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    
    # Initialize sentence transformer
    sentence_model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=True,
    ).eval()
    
    # Convert to half precision for GPU efficiency
    if device == "cuda":
        sentence_model = sentence_model.half()
    
    sentence_model.max_seq_length = max_seq_length
    
    # Initialize KeyBERT
    kw_model = KeyBERT(model=sentence_model)
    kw_model.model_kwargs = {
        "batch_size": batch_size,
        "convert_to_tensor": False,
        "device": device,
    }
    
    # Monkey-patch encode method for consistent settings
    def _encode_with_defaults(texts, **kwargs):
        kwargs.setdefault("batch_size", batch_size)
        kwargs.setdefault("convert_to_tensor", False)
        kwargs.setdefault("device", device)
        with torch.inference_mode():
            return sentence_model._encode_orig(texts, **kwargs)
    
    sentence_model._encode_orig = sentence_model.encode
    sentence_model.encode = _encode_with_defaults
    
    logger.info(f"✅ Model '{model_name}' loaded; batch size = {batch_size}")
    return sentence_model, kw_model

def extract_keywords_keybert(
    df: pd.DataFrame,
    sentence_model: SentenceTransformer,
    kw_model: KeyBERT,
    top_n: int = 30,
    ngram_range: Tuple[int, int] = (1, 4)
) -> Dict[str, Counter]:
    """
    Extract keywords using KeyBERT for each category.
    
    Args:
        df: DataFrame with 'category' and 'processed_text' columns
        sentence_model: Initialized sentence transformer model
        kw_model: Initialized KeyBERT model
        top_n: Number of top keywords to extract per document
        ngram_range: Range of n-grams to extract
        
    Returns:
        Dictionary mapping categories to keyword counters
    """
    logger.info("Extracting KeyBERT keywords by category")
    
    # Combine stopwords
    combined_stop = SPACY_STOP_WORDS.union(SKLEARN_STOP_WORDS).union(CUSTOM_STOPWORDS)
    
    def extract_keywords_for_texts(texts: List[str]) -> List[Tuple[str, float]]:
        """Extract keywords and scores for a list of texts."""
        all_keywords = []
        for txt in tqdm(texts, desc="KeyBERT", leave=False):
            try:
                kws = kw_model.extract_keywords(
                    txt,
                    keyphrase_ngram_range=ngram_range,
                    stop_words=list(combined_stop),
                    top_n=top_n,
                    use_mmr=True,
                    diversity=0.5,
                )
                # Store both keyword and score
                all_keywords.extend([(k, s) for k, s in kws])
            except Exception as e:
                logger.warning(f"Failed to extract keywords from text: {e}")
                continue
        return all_keywords
    
    category_keywords = {}
    for cat in tqdm(df["category"].unique(), desc="Categories"):
        texts = df[df["category"] == cat]["processed_text"].tolist()
        keybert_results = extract_keywords_for_texts(texts)
        
        # Aggregate (keyword, score) tuples by keyword
        # Use weighted average for scores when same keyword appears multiple times
        keyword_data = {}  # {keyword: {'total_score': float, 'count': int, 'scores': list}}
        
        for keyword, score in keybert_results:
            if keyword not in keyword_data:
                keyword_data[keyword] = {'total_score': 0.0, 'count': 0, 'scores': []}
            keyword_data[keyword]['total_score'] += score
            keyword_data[keyword]['count'] += 1
            keyword_data[keyword]['scores'].append(score)
        
        # Calculate weighted average scores and store in category_keywords
        category_keywords[cat] = {}
        for keyword, data in keyword_data.items():
            # Use weighted average: sum of scores / count
            avg_score = data['total_score'] / data['count']
            frequency = data['count']
            category_keywords[cat][keyword] = {'frequency': frequency, 'score': avg_score}
        
        # Debug: Show top keywords for this category
        if category_keywords[cat]:
            # Sort by frequency for display
            top_kws = sorted(category_keywords[cat].items(), key=lambda x: x[1]['frequency'], reverse=True)[:10]
            debug_kws = [(kw, data['frequency'], f"score={data['score']:.3f}") for kw, data in top_kws]
            logger.info(f"Top keywords for '{cat}': {debug_kws}")
    
    return category_keywords

def calculate_tfidf_scores(
    df: pd.DataFrame,
    *,
    ngram_range: Tuple[int, int],
    max_features: int,
    min_df: int,
    max_df: float
) -> Dict[str, Dict[str, float]]:
    """
    Calculate TF-IDF scores for keywords by category.
    
    Args:
        df: DataFrame with 'category' and 'processed_text' columns
        ngram_range: Range of n-grams to extract
        max_features: Maximum number of features
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        
    Returns:
        Dictionary mapping categories to keyword TF-IDF scores
    """
    logger.info("Calculating TF-IDF discriminative scores")
    
    # Combine stopwords
    combined_stop = SPACY_STOP_WORDS.union(SKLEARN_STOP_WORDS).union(CUSTOM_STOPWORDS)
    
    # Fit TF-IDF vectorizer
    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words=list(combined_stop)
    )
    
    X = vec.fit_transform(tqdm(df["processed_text"], desc="Vectorising"))
    feats = vec.get_feature_names_out()
    
    # Calculate mean TF-IDF scores per category
    scores = {}
    for cat in df["category"].unique():
        mask = df["category"] == cat
        mean_vals = X[mask.to_numpy()].mean(axis=0).A1
        scores[cat] = dict(zip(feats, mean_vals))
    
    return scores

def calculate_discriminative_keywords(
    category_keywords: Dict[str, Counter],
    tfidf_scores: Dict[str, Dict[str, float]],
    *,
    min_score: float,
    max_keywords: int
) -> Dict[str, Dict[str, float]]:
    """
    Calculate discriminative keywords based on frequency and TF-IDF scores.
    
    Args:
        category_keywords: Dictionary mapping categories to keyword counters
        tfidf_scores: Dictionary mapping categories to TF-IDF scores
        min_score: Minimum discriminative score threshold
        max_keywords: Maximum number of keywords per category
        
    Returns:
        Dictionary mapping categories to discriminative keyword scores
    """
    logger.info("Scoring discriminative keywords")
    
    def discriminative_score(kw: str, cat: str, cats: List[str]) -> float:
        """Calculate discriminative score for a keyword in a category."""
        # Frequency part
        freq_cat = category_keywords[cat][kw]
        total_freq = sum(category_keywords[c].get(kw, 0) for c in cats)
        freq_score = freq_cat / total_freq if total_freq else 0
        
        # TF-IDF part
        tfidf_cat = tfidf_scores[cat].get(kw, 0)
        tfidf_other = np.mean([tfidf_scores[c].get(kw, 0) for c in cats if c != cat])
        tfidf_score = tfidf_cat / (tfidf_cat + tfidf_other) if (tfidf_cat + tfidf_other) else 0
        
        return 0.5 * freq_score + 0.5 * tfidf_score
    
    all_cats = list(category_keywords.keys())
    discriminative_keywords = {}
    
    for cat in tqdm(all_cats, desc="Categories"):
        kw_scores = {}
        for kw in category_keywords[cat]:
            score = discriminative_score(kw, cat, all_cats)
            if score > min_score:
                kw_scores[kw] = score
        
        # Sort by score and limit to max_keywords
        sorted_kws = sorted(kw_scores.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
        discriminative_keywords[cat] = dict(sorted_kws)
        
        logger.info(f"Top discriminative for '{cat}': {list(discriminative_keywords[cat].items())[:5]}")
    
    return discriminative_keywords

def cluster_keywords(
    discriminative_keywords: Dict[str, Dict[str, float]],
    sentence_model: SentenceTransformer,
    *,
    min_cluster_size: int,
    n_clusters: Optional[int]
) -> Dict[str, Dict[int, List[Tuple[str, float]]]]:
    """
    Cluster keywords using sentence embeddings.
    
    Args:
        discriminative_keywords: Dictionary mapping categories to keyword scores
        sentence_model: Initialized sentence transformer model
        min_cluster_size: Minimum cluster size for HDBSCAN
        n_clusters: Number of clusters for KMeans fallback
        
    Returns:
        Dictionary mapping categories to cluster assignments
    """
    logger.info("Clustering keywords")
    
    def cluster_keywords_for_category(keywords: List[str]) -> Dict[str, int]:
        """Cluster keywords for a single category."""
        if len(keywords) < min_cluster_size:
            return {kw: 0 for kw in keywords}
        
        # Get embeddings
        embeddings = sentence_model.encode(keywords)
        
        # Try HDBSCAN first
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(embeddings)
        
        # Fallback to KMeans if too many outliers
        if (labels == -1).sum() > 0.5 * len(keywords):
            nc = n_clusters or max(2, min(len(keywords) // 5, 40))
            labels = KMeans(n_clusters=nc, random_state=42, n_init=10).fit_predict(embeddings)
        
        return dict(zip(keywords, labels))
    
    category_clusters = {}
    
    for cat, kw_scores in tqdm(discriminative_keywords.items(), desc="Categories"):
        kw_list = list(kw_scores.keys())
        if len(kw_list) > 5:
            label_map = cluster_keywords_for_category(kw_list)
            clusters = defaultdict(list)
            
            for kw, cid in label_map.items():
                if cid != -1:  # ignore outliers
                    clusters[cid].append((kw, kw_scores[kw]))
            
            category_clusters[cat] = clusters
            logger.info(f"{cat}: {len(clusters)} clusters")
        else:
            category_clusters[cat] = {}
    
    return category_clusters

def create_keyword_cluster_data(
    discriminative_keywords: Dict[str, Dict[str, float]],
    category_clusters: Dict[str, Dict[int, List[Tuple[str, float]]]]
) -> Dict[str, Any]:
    """
    Create structured keyword cluster data for export.
    
    Args:
        discriminative_keywords: Dictionary mapping categories to keyword scores
        category_clusters: Dictionary mapping categories to cluster assignments
        
    Returns:
        Structured keyword cluster data
    """
    logger.info("Creating keyword cluster data structure")
    
    keyword_cluster_data = {}
    
    for category in discriminative_keywords.keys():
        # Convert discriminative keyword scores to plain floats
        kw_dict = {
            kw: float(score)
            for kw, score in discriminative_keywords[category].items()
        }
        
        # Build clusters
        clusters_dict = {}
        for cluster_id, cluster_kws in category_clusters.get(category, {}).items():
            kw_list = [kw for kw, _ in cluster_kws]
            score_list = [float(score) for _, score in cluster_kws]
            avg = float(np.mean(score_list))
            rep = max(cluster_kws, key=lambda x: x[1])[0]
            
            clusters_dict[str(int(cluster_id))] = {
                "keywords": kw_list,
                "scores": score_list,
                "avg_score": avg,
                "representative": rep
            }
        
        keyword_cluster_data[category] = {
            "keywords": kw_dict,
            "clusters": clusters_dict
        }
    
    return keyword_cluster_data

def create_visualizations(
    discriminative_keywords: Dict[str, Dict[str, float]],
    category_clusters: Dict[str, Dict[int, List[Tuple[str, float]]]],
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Create visualizations for keyword analysis results.
    
    Args:
        discriminative_keywords: Dictionary mapping categories to keyword scores
        category_clusters: Dictionary mapping categories to cluster assignments
        output_dir: Directory to save visualization files
        
    Returns:
        Dictionary containing visualization data and file paths
    """
    logger.info("Creating keyword analysis visualizations")
    
    all_cats = list(discriminative_keywords.keys())
    visualization_data = {}
    
    # 1. Word Clouds
    try:
        fig, axes = plt.subplots(1, len(all_cats), figsize=(15, 5))
        if len(all_cats) == 1:
            axes = [axes]
        
        for idx, category in enumerate(all_cats):
            word_freq = discriminative_keywords[category]
            if word_freq:
                wordcloud = WordCloud(
                    width=400,
                    height=300,
                    background_color='white',
                    colormap='viridis',
                    relative_scaling=0.5
                ).generate_from_frequencies(word_freq)
                
                axes[idx].imshow(wordcloud, interpolation='bilinear')
                axes[idx].axis('off')
                axes[idx].set_title(f"'{category}' Keywords", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        wordcloud_path = f"{output_dir}/keyword_wordclouds.png"
        plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_data["wordclouds"] = wordcloud_path
        logger.info(f"Word clouds saved to {wordcloud_path}")
        
    except Exception as e:
        logger.warning(f"Failed to create word clouds: {e}")
    
    # 2. Discriminative Keywords Comparison
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_n = 15
        y_offset = 0
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_cats)))
        
        for idx, category in enumerate(all_cats):
            keywords = list(discriminative_keywords[category].items())[:top_n]
            if keywords:
                kw_names, kw_scores = zip(*keywords)
                y_positions = np.arange(len(kw_names)) + y_offset
                
                bars = ax.barh(y_positions, kw_scores, label=category, color=colors[idx], alpha=0.8)
                
                # Add category label
                ax.text(-0.1, y_offset + len(kw_names)/2, category, fontsize=12, fontweight='bold',
                        ha='right', va='center', rotation=0)
                
                # Add keyword labels
                for i, (pos, name) in enumerate(zip(y_positions, kw_names)):
                    ax.text(0.01, pos, name, fontsize=10, va='center')
                
                y_offset += len(kw_names) + 1
        
        ax.set_xlabel('Discriminative Score', fontsize=12)
        ax.set_title('Top Discriminative Keywords by Category', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1.1)
        ax.set_ylim(-1, y_offset)
        ax.set_yticks([])
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        comparison_path = f"{output_dir}/discriminative_keywords_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_data["comparison"] = comparison_path
        logger.info(f"Keywords comparison saved to {comparison_path}")
        
    except Exception as e:
        logger.warning(f"Failed to create keywords comparison: {e}")
    
    # 3. Cluster Summary Statistics
    try:
        fig = make_subplots(
            rows=1, cols=len(all_cats),
            subplot_titles=[f"'{cat}' Clusters" for cat in all_cats],
            specs=[[{'type': 'bar'} for _ in all_cats]]
        )
        
        for idx, category in enumerate(all_cats):
            if category in category_clusters:
                cluster_sizes = []
                cluster_avg_scores = []
                cluster_names = []
                
                for cluster_id, cluster_kws in category_clusters[category].items():
                    cluster_sizes.append(len(cluster_kws))
                    avg_score = np.mean([score for _, score in cluster_kws])
                    cluster_avg_scores.append(avg_score)
                    cluster_names.append(f"C{cluster_id}")
                
                # Sort by average score
                sorted_indices = np.argsort(cluster_avg_scores)[::-1]
                
                fig.add_trace(
                    go.Bar(
                        x=[cluster_names[i] for i in sorted_indices],
                        y=[cluster_avg_scores[i] for i in sorted_indices],
                        text=[f"Size: {cluster_sizes[i]}" for i in sorted_indices],
                        textposition='auto',
                        name=category,
                        marker_color=px.colors.qualitative.Set3[idx % len(px.colors.qualitative.Set3)]
                    ),
                    row=1, col=idx+1
                )
        
        fig.update_xaxes(title_text="Cluster ID", row=1, col=1)
        fig.update_yaxes(title_text="Avg Discriminative Score", row=1, col=1)
        fig.update_layout(height=400, showlegend=False, title_text="Cluster Quality by Category")
        
        cluster_path = f"{output_dir}/cluster_summary.html"
        fig.write_html(cluster_path)
        visualization_data["cluster_summary"] = cluster_path
        logger.info(f"Cluster summary saved to {cluster_path}")
        
    except Exception as e:
        logger.warning(f"Failed to create cluster summary: {e}")
    
    return visualization_data

def run_keyword_extraction_only(
    df: pd.DataFrame,
    *,
    model_name: str,
    device: str,
    batch_size: int,
    max_seq_length: int,
    top_n: int,
    ngram_range: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Run only keyword extraction without clustering.
    
    Args:
        df: DataFrame with 'category' and 'processed_text' columns
        model_name: Name of the sentence transformer model
        device: Device to run on ('cuda' or 'cpu')
        batch_size: Batch size for embeddings
        max_seq_length: Maximum sequence length
        top_n: Number of top keywords to extract per document
        ngram_range: Range of n-grams to extract
        
    Returns:
        Dictionary containing keyword extraction results
    """
    logger.info("Running keyword extraction only")
    logger.info(f"Using device: {device}")
    
    # CUDA optimizations
    if device == "cuda" and torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Clear cache before starting
        torch.cuda.empty_cache()
    
    # Initialize models
    sentence_model, kw_model = initialize_keybert_model(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_seq_length=max_seq_length
    )
    
    # Extract keywords
    category_keywords = extract_keywords_keybert(
        df, sentence_model, kw_model, top_n, ngram_range
    )
    
    # Convert to DataFrame format for consistency with Modal function
    results_data = []
    for category, keywords in category_keywords.items():
        for keyword, data in keywords.items():
            results_data.append({
                'category': category,
                'keyword': keyword,
                'frequency': data['frequency'],
                'score': data['score'],
                'ngram_length': len(keyword.split())
            })
    
    results_df = pd.DataFrame(results_data)
    
    # Handle empty results
    if results_df.empty:
        logger.warning("No keywords found! Creating empty DataFrame with expected columns.")
        results_df = pd.DataFrame(columns=['category', 'keyword', 'frequency', 'score', 'ngram_length'])
    else:
        # Sort by category and frequency
        results_df = results_df.sort_values(['category', 'frequency'], ascending=[True, False])
    
    logger.info(f"Returning DataFrame with {len(results_df)} keyword entries")
    
    # Clean up GPU memory
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cleared")
    
    return results_df

def run_keyword_clustering(
    discriminative_keywords: Dict[str, Dict[str, float]],
    sentence_model: SentenceTransformer,
    min_cluster_size: int = 3,
    n_clusters: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run keyword clustering on pre-extracted keywords.
    
    Args:
        discriminative_keywords: Dictionary mapping categories to keyword scores
        sentence_model: Initialized sentence transformer model
        min_cluster_size: Minimum cluster size for HDBSCAN
        n_clusters: Number of clusters for KMeans fallback
        
    Returns:
        Dictionary containing clustering results
    """
    logger.info("Running keyword clustering")
    
    # Cluster keywords
    category_clusters = cluster_keywords(
        discriminative_keywords, sentence_model, min_cluster_size=min_cluster_size, n_clusters=n_clusters
    )
    
    # Create structured data
    keyword_cluster_data = create_keyword_cluster_data(
        discriminative_keywords, category_clusters
    )
    
    return {
        "category_clusters": category_clusters,
        "keyword_cluster_data": keyword_cluster_data
    }

def run_keyword_visualization(
    discriminative_keywords: Dict[str, Dict[str, float]],
    category_clusters: Dict[str, Dict[int, List[Tuple[str, float]]]],
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Run visualization of keyword analysis results.
    
    Args:
        discriminative_keywords: Dictionary mapping categories to keyword scores
        category_clusters: Dictionary mapping categories to cluster assignments
        output_dir: Directory to save visualization files
        
    Returns:
        Dictionary containing visualization results
    """
    logger.info("Running keyword visualization")
    
    # Create visualizations
    visualization_data = create_visualizations(
        discriminative_keywords, category_clusters, output_dir
    )
    
    return {
        "visualization_data": visualization_data,
        "visualization_files": list(visualization_data.values())
    }
