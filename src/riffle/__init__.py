"""Riffle: decode attention that scores per-page key summaries and reads only the top pages (the rest are dropped)."""
from .cache import G, RiffleCache, tokens_per_block
from .summaries import PAGE, RK, SIG, page_scores, summarize, top_dirs

__all__ = ["RiffleCache", "page_scores", "summarize", "top_dirs", "tokens_per_block", "PAGE", "RK", "SIG", "G"]
__version__ = "0.1.0"
