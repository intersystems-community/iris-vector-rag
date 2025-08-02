"""
Local cache data source implementation.

This module provides access to locally cached documents.
"""

from pathlib import Path
from typing import List, Optional
from quick_start.data.interfaces import IDataSource, DocumentMetadata


class LocalCacheDataSource(IDataSource):
    """Local cache data source implementation."""
    
    def __init__(self):
        """Initialize the local cache data source."""
        pass
    
    async def list_available_documents(
        self, 
        categories: List[str],
        limit: Optional[int] = None
    ) -> List[DocumentMetadata]:
        """List available documents for download."""
        # Stub implementation - returns empty list for now
        return []
    
    async def download_document(
        self, 
        metadata: DocumentMetadata,
        storage_path: Path
    ) -> Path:
        """Download a single document."""
        # Stub implementation
        storage_path.mkdir(parents=True, exist_ok=True)
        local_path = storage_path / f"{metadata.pmc_id}.xml"
        local_path.write_text("<article>Cached content</article>")
        return local_path
    
    async def verify_document(
        self, 
        metadata: DocumentMetadata,
        local_path: Path
    ) -> bool:
        """Verify downloaded document integrity."""
        return local_path.exists() and local_path.stat().st_size > 0