"""
TOON Format Formatter for DeepLightRAG Retrieval Context

Formats retrieval results using TOON (Token-Oriented Object Notation) for maximum
token efficiency and LLM comprehension.

TOON achieves ~40% token reduction vs JSON while maintaining high LLM accuracy.
"""

from typing import Dict, List, Any
import re


"""
TOON Format Formatter for DeepLightRAG Retrieval Context

Formats retrieval results using TOON (Token-Oriented Object Notation) for maximum
token efficiency and LLM comprehension via the toon-python library.
"""

from typing import Dict, Any

try:
    import toon_python as toon
except ImportError:
    try:
        # Fallback if package name is different in some envs
        import toon
    except ImportError:
        toon = None

def format_toon_context(result: Dict[str, Any], simple: bool = True, use_tabs: bool = False, use_library: bool = True) -> str:
    """
    Format retrieval result to TOON using toon-python library.
    
    Args:
        result: Retrieval result from DeepLightRAG
        simple: Simplified output (ignored by library, kept for compatibility)
        use_tabs: Use tabs (passed to library options if supported)
        use_library: Always True now, kept for backward compatibility signature
        
    Returns:
        TOON-formatted string
    """
    if toon is None:
        raise ImportError("toon-python library not found. Please install with `pip install toon-python`")
    
    # Use encode method from library
    return toon.encode(result)


from ..interfaces import BaseFormatter

class ToonFormatter(BaseFormatter):
    """Backward compatibility wrapper for ToonFormatter"""
    
    def __init__(self, use_tabs: bool = False):
        self.use_tabs = use_tabs

    def format_retrieval_result(self, result: Dict[str, Any]) -> str:
        return format_toon_context(result, simple=False, use_tabs=self.use_tabs)

    def format_simple_context(self, result: Dict[str, Any]) -> str:
        return format_toon_context(result, simple=True, use_tabs=self.use_tabs)

