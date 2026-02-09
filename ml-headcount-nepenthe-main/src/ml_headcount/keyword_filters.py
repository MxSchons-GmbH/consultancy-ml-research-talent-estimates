"""
Keyword filter functions based on the paper's 3 binary filters.

This module implements the exact filtering logic from filter_profiles.sh
as Python functions for use in the Hamilton pipeline.
"""

import re
from typing import Optional


def ml_match(text: Optional[str]) -> bool:
    """
    ML Selection - Base Filter
    
    Returns True if the text contains ANY of the core ML keywords:
    - machine learning, machine-learning, ML
    - deep learning, deep-learning
    - reinforcement learning, reinforcement-learning, RL
    
    Args:
        text: CV text content to search
        
    Returns:
        True if text contains any ML term, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    # Case-insensitive search
    text_lower = text.lower()
    
    # Check for ML keywords - word boundaries added programmatically
    keywords = [
        r'machine[\s-]?learning',
        'ml',
        r'deep[\s-]?learning',
        r'reinforcement[\s-]?learning',
        'rl'
    ]
    
    for keyword in keywords:
        pattern = rf'\b{keyword}\b'
        if re.search(pattern, text_lower):
            return True
    
    return False


def broad_yes_match(text: Optional[str]) -> bool:
    """
    Broad_yes - Research Indicators Filter
    
    Returns True if the text contains ANY of the research/AI safety keywords:
    - augmented generation, agent reinforcement, mats scholar, mats
    - research scientist, evals, interpretability, feature engineering
    - research intern, candidate, graduate research assistant
    - science institute, staff research scientist, doctor
    
    Args:
        text: CV text content to search
        
    Returns:
        True if text contains any research term, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    # Case-insensitive search
    text_lower = text.lower()
    
    # Check for research keywords - word boundaries added programmatically
    keywords = [
        'augmented generation',
        'agent reinforcement',
        'mats scholar',
        'mats',
        'research scientist',
        'evals',
        'interpretability',
        'feature engineering',
        'research intern',
        'candidate',
        'graduate research assistant',
        'science institute',
        'staff research scientist',
        'doctor'
    ]
    
    for keyword in keywords:
        pattern = rf'\b{keyword}\b'
        if re.search(pattern, text_lower):
            return True
    
    return False


def strict_no_match(text: Optional[str]) -> bool:
    """
    Strict_no - Business Indicators Filter (INVERTED LOGIC)
    
    Returns False if the text contains ANY of the business/generic keywords.
    Returns True if the text does NOT contain any of these keywords.
    
    Business keywords include:
    - certificate, programmer, council, companies, capital
    - proven track record, pilot, money, specialist, chief
    - udemy, customer, management, today, cross functional
    - administrator, excellence, commerce, linkedin, leader
    - incident, tier, brand, investment, hr, sites
    - offerings, prior, centers, advising, certified information
    - key responsibilities, master data, anti, deadlines
    - physiology, carbon, impacts, certified machine, qualification
    
    Args:
        text: CV text content to search
        
    Returns:
        True if text does NOT contain business terms (PASS filter)
        False if text contains any business term (REJECT)
    """
    if not text or not isinstance(text, str):
        return True  # If no text, default to passing (no business indicators found)
    
    # Case-insensitive search
    text_lower = text.lower()
    
    # Check for business/generic keywords - word boundaries added programmatically
    keywords = [
        'certificate',
        'programmer',
        'council',
        'companies',
        'capital',
        'proven track record',
        'pilot',
        'money',
        'specialist',
        'chief',
        'udemy',
        'track record',
        'customer',
        'management',
        'today',
        'cross functional',
        'administrator',
        'excellence',
        'commerce',
        'linkedin',
        'leader',
        'incident',
        'tier',
        'brand',
        'investment',
        'hr',
        'sites',
        'offerings',
        'prior',
        'centers',
        'advising',
        'certified information',
        'key responsibilities',
        'master data',
        'anti',
        'deadlines',
        'physiology',
        'carbon',
        'impacts',
        'certified machine',
        'qualification'
    ]
    
    for keyword in keywords:
        pattern = rf'\b{keyword}\b'
        if re.search(pattern, text_lower):
            return False  # Found business term -> REJECT (return False)
    
    return True  # No business terms found -> PASS (return True)


def filter_broad_yes(text: Optional[str]) -> bool:
    """
    Compound Filter: ML Selection AND Broad_yes
    
    Returns True if text matches BOTH:
    - ml_match (contains ML keywords)
    - broad_match (contains research indicator keywords)
    
    Args:
        text: CV text content to filter
        
    Returns:
        True if both filters pass, False otherwise
    """
    return ml_match(text) and broad_yes_match(text)


def filter_strict_no(text: Optional[str]) -> bool:
    """
    Compound Filter: ML Selection AND Strict_no
    
    Returns True if text matches BOTH:
    - ml_match (contains ML keywords)
    - strict_no_match (does NOT contain business indicator keywords)
    
    Args:
        text: CV text content to filter
        
    Returns:
        True if both filters pass, False otherwise
    """
    return ml_match(text) and strict_no_match(text)


def filter_broad_yes_strict_no(text: Optional[str]) -> bool:
    """
    Compound Filter: ML Selection AND Broad_yes AND Strict_no
    
    Returns True if text matches ALL THREE:
    - ml_match (contains ML keywords)
    - broad_match (contains research indicator keywords)
    - strict_no_match (does NOT contain business indicator keywords)
    
    Args:
        text: CV text content to filter
        
    Returns:
        True if all three filters pass, False otherwise
    """
    return ml_match(text) and broad_yes_match(text) and strict_no_match(text)


def apply_compound_filters(text: Optional[str]) -> dict:
    """
    Apply all three compound paper filters to a text and return results.
    
    These are the three filter combinations used in the paper as Dawid-Skene
    annotator inputs:
    - filter_broad_yes: ML Selection AND Broad_yes
    - filter_strict_no: ML Selection AND Strict_no  
    - filter_broad_yes_strict_no: ML Selection AND Broad_yes AND Strict_no
    
    Args:
        text: CV text content to filter
        
    Returns:
        Dictionary with compound filter results:
        {
            'filter_broad_yes': bool,
            'filter_strict_no': bool,
            'filter_broad_yes_strict_no': bool
        }
    """
    return {
        'filter_broad_yes': filter_broad_yes(text),
        'filter_strict_no': filter_strict_no(text),
        'filter_broad_yes_strict_no': filter_broad_yes_strict_no(text)
    }


def apply_primitive_filters(text: Optional[str]) -> dict:
    """
    Apply the three primitive paper filters to a text and return results.
    
    These are the component filters that can be combined:
    - ml_match: Base ML keywords
    - broad_match: Research indicator keywords
    - strict_no_match: Absence of business indicator keywords
    
    Args:
        text: CV text content to filter
        
    Returns:
        Dictionary with primitive filter results:
        {
            'ml_match': bool,
            'broad_match': bool, 
            'strict_no_match': bool
        }
    """
    return {
        'ml_match': ml_match(text),
        'broad_match': broad_yes_match(text),
        'strict_no_match': strict_no_match(text)
    }

