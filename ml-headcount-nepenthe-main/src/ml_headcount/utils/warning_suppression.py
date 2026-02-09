"""
Warning suppression utilities for the ML Headcount Pipeline.
"""

import warnings
import logging

def suppress_upstream_warnings():
    """
    Suppress common upstream warnings that clutter the output.
    """
    # Suppress pandas warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
    warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
    
    # Suppress numpy warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='numpy')
    warnings.filterwarnings('ignore', category=UserWarning, module='numpy')
    
    # Suppress matplotlib warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    # Suppress sklearn warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    
    # Suppress hamilton warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='hamilton')
    
    # Suppress plotly warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='plotly')
    
    # Suppress keybert warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='keybert')
    
    # Suppress sentence transformers warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='sentence_transformers')
    
    # Set logging level to reduce verbosity
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('plotly').setLevel(logging.WARNING)
    logging.getLogger('keybert').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
