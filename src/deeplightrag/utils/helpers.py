"""
Utility functions for Visual-Graph RAG
"""

import time
from typing import Optional


def estimate_tokens(text: str, method: str = "chars") -> int:
    """
    Estimate token count from text

    Args:
        text: Input text
        method: Estimation method ('chars', 'words', 'tiktoken')

    Returns:
        Estimated token count
    """
    if method == "chars":
        # Rough estimate: 4 characters per token
        return len(text) // 4
    elif method == "words":
        # Rough estimate: 0.75 tokens per word
        return int(len(text.split()) * 0.75)
    elif method == "tiktoken":
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            return len(text) // 4
    else:
        return len(text) // 4


def format_time(seconds: float) -> str:
    """Format time in human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """Create a text-based progress bar"""
    percentage = current / total if total > 0 else 0
    filled = int(width * percentage)
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}] {percentage*100:.1f}%"


def calculate_compression_ratio(compressed_tokens: int, estimated_original: int) -> float:
    """Calculate compression ratio"""
    if compressed_tokens == 0:
        return 0.0
    return estimated_original / compressed_tokens


def format_number(num: int) -> str:
    """Format large numbers with commas"""
    return f"{num:,}"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


class Timer:
    """Context manager for timing operations"""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"{self.name}: {format_time(elapsed)}")

    @property
    def elapsed(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
