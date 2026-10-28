"""SparsePage: decode attention that scores per-page key summaries and reads only the top pages (the rest are dropped)."""
from .cache import G, SparsePageCache, tokens_per_block
from .summaries import PAGE, RK, SIG, page_scores, summarize, top_dirs

__all__ = ["SparsePageCache", "page_scores", "summarize", "top_dirs", "tokens_per_block", "PAGE", "RK", "SIG", "G"]
__version__ = "0.1.0"
