"""
PMC API data source implementation.

This module provides access to PMC documents via the PMC API.
"""

from pathlib import Path
from typing import List, Optional
from quick_start.data.interfaces import IDataSource, DocumentMetadata


class PMCAPIDataSource(IDataSource):
    """PMC API data source implementation."""
    
    def __init__(self):
        """Initialize the PMC API data source."""
        pass
    
    async def list_available_documents(
        self, 
        categories: List[str],
        limit: Optional[int] = None
    ) -> List[DocumentMetadata]:
        """List available documents for download."""
        # Stub implementation - returns mock data
        docs = []
        count = limit or 10
        
        for i in range(count):
            docs.append(DocumentMetadata(
                pmc_id=f"PMC{1000000 + i:06d}",
                title=f"Test {categories[0] if categories else 'Medical'} Document {i+1}",
                authors=[f"Dr. Test Author {i+1}"],
                abstract=f"This is a test {categories[0] if categories else 'medical'} document abstract {i+1}.",
                categories=categories[:1] if categories else ["medical"],
                file_size=1024 * (i + 1),
                download_url=f"https://example.com/PMC{1000000 + i:06d}.xml"
            ))
        
        return docs
    
    async def download_document(
        self, 
        metadata: DocumentMetadata,
        storage_path: Path
    ) -> Path:
        """Download a single document."""
        # Stub implementation - creates a mock file
        storage_path.mkdir(parents=True, exist_ok=True)
        local_path = storage_path / f"{metadata.pmc_id}.xml"
        
        # Create mock XML content
        mock_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<article>
    <front>
        <article-meta>
            <title-group>
                <article-title>{metadata.title}</article-title>
            </title-group>
            <abstract>
                <p>{metadata.abstract}</p>
            </abstract>
        </article-meta>
    </front>
    <body>
        <sec>
            <title>Introduction</title>
            <p>This is the introduction section of {metadata.pmc_id}.</p>
        </sec>
    </body>
</article>"""
        
        local_path.write_text(mock_content)
        return local_path
    
    async def verify_document(
        self, 
        metadata: DocumentMetadata,
        local_path: Path
    ) -> bool:
        """Verify downloaded document integrity."""
        # Stub implementation
        return local_path.exists() and local_path.stat().st_size > 0