"""
Matplotlib font utilities for proper Chinese character rendering.

This module provides utilities to set up matplotlib for proper rendering of Chinese characters
and other CJK (Chinese, Japanese, Korean) text across all plotting functions.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import logging

logger = logging.getLogger(__name__)

def setup_chinese_font() -> str:
    """
    Set up matplotlib to properly render Chinese characters and other CJK text.
    
    This function:
    1. Searches for available CJK fonts on the system
    2. Configures matplotlib rcParams to use the best available CJK font
    3. Updates the font fallback chain to prioritize CJK fonts
    
    Returns:
        str: The name of the selected font, or None if setup failed
    """
    try:
        # List of CJK fonts to try, in order of preference
        chinese_fonts = [
            'Noto Sans CJK JP',      # Available on this system
            'Noto Serif CJK JP',     # Available on this system
            'Noto Sans CJK SC',      # Simplified Chinese
            'Noto Sans CJK TC',      # Traditional Chinese
            'Noto Sans CJK KR',      # Korean
            'Source Han Sans SC',    # Adobe's CJK font
            'Source Han Serif SC',   # Adobe's CJK serif font
            'SimHei',                # Windows
            'PingFang SC',           # macOS
            'Hiragino Sans GB',      # macOS
            'WenQuanYi Micro Hei',   # Linux
            'Arial Unicode MS',      # macOS
            'DejaVu Sans',           # Fallback
        ]
        
        selected_font = None
        for font_name in chinese_fonts:
            try:
                # Check if font is available
                font_path = fm.findfont(fm.FontProperties(family=font_name))
                if (font_name.lower() in font_path.lower() or 
                    'cjk' in font_path.lower() or 
                    'han' in font_path.lower() or
                    'noto' in font_path.lower()):
                    selected_font = font_name
                    logger.info(f"Found CJK font: {font_name}")
                    break
            except Exception:
                continue
        
        if selected_font:
            # Properly configure matplotlib rcParams for CJK character support
            plt.rcParams['font.family'] = selected_font
            
            # Update the sans-serif font list to prioritize the CJK font
            current_sans_serif = plt.rcParams['font.sans-serif'].copy()
            if selected_font not in current_sans_serif:
                plt.rcParams['font.sans-serif'] = [selected_font] + current_sans_serif
            
            # Also update serif fonts if we found a serif CJK font
            if 'serif' in selected_font.lower():
                current_serif = plt.rcParams['font.serif'].copy()
                if selected_font not in current_serif:
                    plt.rcParams['font.serif'] = [selected_font] + current_serif
            
            logger.info(f"Configured matplotlib to use CJK font: {selected_font}")
            return selected_font
        else:
            # Fallback: use a font that supports Unicode
            plt.rcParams['font.family'] = 'DejaVu Sans'
            logger.warning("No CJK font found, using DejaVu Sans as fallback")
            return 'DejaVu Sans'
        
    except Exception as e:
        logger.warning(f"Could not set up CJK font: {e}")
        return None

def ensure_chinese_font_support():
    """
    Ensure matplotlib is configured for Chinese character support.
    
    This is a convenience function that can be called at the beginning of any
    plotting function to ensure proper font configuration.
    """
    return setup_chinese_font()

def test_chinese_rendering(text="测试中文渲染", save_path=None):
    """
    Test Chinese character rendering with the current font configuration.
    
    Args:
        text (str): Chinese text to test rendering
        save_path (str, optional): Path to save test plot
        
    Returns:
        matplotlib.figure.Figure: The test figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, text, fontsize=20, ha='center', va='center')
    ax.set_title(f'Chinese Character Test: {text}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Chinese rendering test saved to: {save_path}")
    
    return fig