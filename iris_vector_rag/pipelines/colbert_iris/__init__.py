from .ingest import ColBERTIngestor
from .maxsim_indb import MaxSimInDB
from .pipeline import IRISColBERTPipeline
from .plaid import PLAIDBuilder, PLAIDNotBuiltError, PLAIDSearcher
from .schema import ColBERTSchema

__all__ = [
    "IRISColBERTPipeline",
    "ColBERTSchema",
    "ColBERTIngestor",
    "MaxSimInDB",
    "PLAIDBuilder",
    "PLAIDSearcher",
    "PLAIDNotBuiltError",
    "search_via_sp",
]


def search_via_sp(conn, query_token_vecs, top_k=10, n_probe=4):
    return PLAIDSearcher(conn).search_via_sp(
        conn, query_token_vecs, top_k=top_k, n_probe=n_probe
    )
