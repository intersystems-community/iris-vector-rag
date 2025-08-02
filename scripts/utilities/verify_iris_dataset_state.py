#!/usr/bin/env python3
"""
IRIS Dataset State Verification Script

This script verifies the current state of the dataset in IRIS, specifically:
1. Count of documents in RAG.SourceDocuments
2. Count of token embeddings in RAG.DocumentTokenEmbeddings  
3. Count of documents with token embeddings
4. Sample of documents missing token embeddings
5. Summary of data state for RAGAS evaluation readiness

Uses the iris_rag.core.connection.ConnectionManager for database connectivity.
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager


class DatasetStateVerifier:
    """Verifies the current state of the IRIS dataset for RAG operations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the dataset state verifier.
        
        Args:
            config_path: Optional path to configuration file. If None, uses default.
        """
        # Use the default config path if none provided
        if config_path is None:
            config_path = os.path.join(project_root, "config", "default.yaml")
        
        self.config_manager = ConfigurationManager(config_path=config_path)
        self.connection_manager = ConnectionManager(self.config_manager)
        self.connection = None
        
    def connect(self):
        """Establish database connection."""
        try:
            self.connection = self.connection_manager.get_connection("iris")
            print("✓ Successfully connected to IRIS database")
        except Exception as e:
            print(f"✗ Failed to connect to IRIS database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            try:
                self.connection_manager.close_connection("iris")
                print("✓ Database connection closed")
            except Exception as e:
                print(f"Warning: Error closing connection: {e}")
    
    def execute_query(self, query: str, description: str) -> Any:
        """
        Execute a SQL query and return results.
        
        Args:
            query: SQL query to execute
            description: Description of what the query does
            
        Returns:
            Query results
        """
        try:
            print(f"\n📊 {description}")
            print(f"Query: {query}")
            
            cursor = self.connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            
            return result
        except Exception as e:
            print(f"✗ Error executing query: {e}")
            return None
    
    def count_source_documents(self) -> int:
        """Count total documents in RAG.SourceDocuments."""
        query = "SELECT COUNT(*) FROM RAG.SourceDocuments"
        result = self.execute_query(query, "Counting documents in RAG.SourceDocuments")
        
        if result and len(result) > 0:
            count = result[0][0]
            print(f"📄 Total documents in RAG.SourceDocuments: {count:,}")
            return count
        else:
            print("✗ Failed to count source documents")
            return 0
    
    def count_token_embeddings(self) -> int:
        """Count total token embeddings in RAG.DocumentTokenEmbeddings."""
        query = "SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings"
        result = self.execute_query(query, "Counting token embeddings in RAG.DocumentTokenEmbeddings")
        
        if result and len(result) > 0:
            count = result[0][0]
            print(f"🔢 Total token embeddings in RAG.DocumentTokenEmbeddings: {count:,}")
            return count
        else:
            print("✗ Failed to count token embeddings")
            return 0
    
    def count_documents_with_embeddings(self) -> int:
        """Count distinct documents that have token embeddings."""
        query = "SELECT COUNT(DISTINCT doc_id) FROM RAG.DocumentTokenEmbeddings"
        result = self.execute_query(query, "Counting distinct documents with token embeddings")
        
        if result and len(result) > 0:
            count = result[0][0]
            print(f"📋 Documents with token embeddings: {count:,}")
            return count
        else:
            print("✗ Failed to count documents with embeddings")
            return 0
    
    def find_missing_embeddings_sample(self) -> list:
        """Find sample of documents missing token embeddings."""
        query = """
        SELECT TOP 10 sd.doc_id 
        FROM RAG.SourceDocuments sd 
        LEFT JOIN RAG.DocumentTokenEmbeddings dte ON sd.doc_id = dte.doc_id 
        WHERE dte.doc_id IS NULL
        """
        result = self.execute_query(query, "Finding documents missing token embeddings (sample)")
        
        if result:
            missing_docs = [row[0] for row in result]
            if missing_docs:
                print(f"⚠️  Sample documents missing token embeddings:")
                for i, doc_id in enumerate(missing_docs, 1):
                    print(f"   {i}. {doc_id}")
            else:
                print("✓ No documents missing token embeddings found")
            return missing_docs
        else:
            print("✗ Failed to find missing embeddings")
            return []
    
    def verify_table_existence(self) -> Dict[str, bool]:
        """Verify that required tables exist."""
        tables_to_check = [
            "RAG.SourceDocuments",
            "RAG.DocumentTokenEmbeddings",
            "RAG.KnowledgeGraphNodes"  # Added table
        ]
        
        table_status = {}
        
        for table in tables_to_check:
            query = f"SELECT COUNT(*) FROM {table}"
            try:
                result = self.execute_query(query, f"Checking existence of {table}")
                table_status[table] = result is not None
                if table_status[table]:
                    print(f"✓ Table {table} exists and is accessible")
                else:
                    print(f"✗ Table {table} is not accessible")
            except Exception as e:
                print(f"✗ Table {table} does not exist or is not accessible: {e}")
                table_status[table] = False
        
        return table_status

    def verify_knowledge_graph_nodes_columns(self) -> Dict[str, bool]:
        """Verify that RAG.KnowledgeGraphNodes has the expected columns."""
        print("\n🔎 Verifying columns in RAG.KnowledgeGraphNodes")
        # Check for 'node_type' and also the old 'content' column
        columns_to_check = ["node_id", "node_name", "node_type", "embedding", "content"]
        column_status = {col: False for col in columns_to_check}

        if not self.connection:
            print("✗ Cannot verify columns: No database connection.")
            return column_status

        cursor = None
        try:
            cursor = self.connection.cursor()
            # Fetch all column names for the table
            # This is a more robust way than trying to select each one.
            # The specific query to get column metadata might vary by DB,
            # but %SQL.Statement is a common InterSystems IRIS approach.
            # A simpler, though less direct, method for some DBs is INFORMATION_SCHEMA.COLUMNS
            # For IRIS, let's try a common SQL way to list columns for a table.
            # If this specific metadata query fails, an alternative is to try selecting each column.
            
            # Attempt to get table metadata (specific to IRIS SQL)
            # This might need adjustment based on exact IRIS SQL dialect for metadata
            # For simplicity in this script, we'll try selecting the columns.
            # If a column doesn't exist, the SELECT will fail.

            for col_name in columns_to_check:
                try:
                    # Try selecting the column. If it fails, it likely doesn't exist.
                    # WHERE 1=0 makes sure we don't fetch data, just check schema.
                    cursor.execute(f"SELECT {col_name} FROM RAG.KnowledgeGraphNodes WHERE 1=0")
                    column_status[col_name] = True
                    print(f"   ✓ Column '{col_name}' exists.")
                except Exception as e:
                    # Check if the error is due to column not found (e.g., SQLCODE -29 for IRIS)
                    if "SQLCODE: <-29>" in str(e) or "Field not found" in str(e) or "Invalid column name" in str(e).lower():
                        print(f"   ✗ Column '{col_name}' does not exist or is inaccessible.")
                    else:
                        print(f"   ⚠️  Error checking column '{col_name}': {e}")
                    column_status[col_name] = False
            
        except Exception as e:
            print(f"✗ Error verifying RAG.KnowledgeGraphNodes columns: {e}")
        finally:
            if cursor:
                cursor.close()
        
        return column_status

    def generate_summary(self, stats: Dict[str, Any]) -> str:
        """Generate a comprehensive summary of the dataset state."""
        summary = []
        summary.append("=" * 80)
        summary.append("IRIS DATASET STATE SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Verification Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        # Table existence
        summary.append("📋 TABLE EXISTENCE:")
        for table, exists in stats['table_status'].items():
            status = "✓ EXISTS" if exists else "✗ MISSING"
            summary.append(f"   {table}: {status}")
        summary.append("")

        # Knowledge Graph Nodes Schema
        if 'kg_nodes_columns_status' in stats:
            summary.append("🧬 KNOWLEDGE GRAPH NODES SCHEMA (RAG.KnowledgeGraphNodes):")
            if stats['table_status'].get("RAG.KnowledgeGraphNodes", False):
                # Prioritize showing node_type status, then content if node_type is missing
                checked_cols_display_order = ["node_id", "node_name", "node_type", "embedding"]
                if not stats['kg_nodes_columns_status'].get('node_type', False) and stats['kg_nodes_columns_status'].get('content', False):
                    checked_cols_display_order.append('content')
                
                for col_name_display in checked_cols_display_order:
                    if col_name_display in stats['kg_nodes_columns_status']:
                        exists = stats['kg_nodes_columns_status'][col_name_display]
                        col_status = "✓ EXISTS" if exists else "✗ MISSING"
                        summary.append(f"   Column '{col_name_display}': {col_status}")
                # If content was checked but not in display order (because node_type exists), show it too for completeness
                if 'content' in stats['kg_nodes_columns_status'] and 'content' not in checked_cols_display_order:
                    exists = stats['kg_nodes_columns_status']['content']
                    col_status = "✓ EXISTS" if exists else "✗ MISSING"
                    summary.append(f"   Column 'content' (old name): {col_status}")

            else:
                summary.append("   Table RAG.KnowledgeGraphNodes does not exist or is inaccessible.")
            summary.append("")
        
        # Document counts
        summary.append("📊 DOCUMENT COUNTS:")
        summary.append(f"   Total Source Documents: {stats['source_docs']:,}")
        summary.append(f"   Total Token Embeddings: {stats['token_embeddings']:,}")
        summary.append(f"   Documents with Embeddings: {stats['docs_with_embeddings']:,}")
        summary.append("")
        
        # Coverage analysis
        if stats['source_docs'] > 0:
            coverage_pct = (stats['docs_with_embeddings'] / stats['source_docs']) * 100
            summary.append("📈 COVERAGE ANALYSIS:")
            summary.append(f"   Embedding Coverage: {coverage_pct:.1f}%")
            
            missing_count = stats['source_docs'] - stats['docs_with_embeddings']
            summary.append(f"   Documents Missing Embeddings: {missing_count:,}")
            summary.append("")
        
        # Readiness assessment
        summary.append("🎯 READINESS ASSESSMENT:")
        
        # 1000-document minimum check
        meets_1000_min = stats['source_docs'] >= 1000
        summary.append(f"   1000+ Document Minimum: {'✓ MET' if meets_1000_min else '✗ NOT MET'} ({stats['source_docs']:,} docs)")
        
        # ColBERT readiness check
        colbert_ready = stats['docs_with_embeddings'] >= 1000 and stats['token_embeddings'] > 0
        summary.append(f"   ColBERT Evaluation Ready: {'✓ READY' if colbert_ready else '✗ NOT READY'}")
        
        # GraphRAG readiness (depends on node_type column)
        kg_cols_status = stats.get('kg_nodes_columns_status', {})
        graphrag_node_type_exists = kg_cols_status.get('node_type', False)
        graphrag_content_exists_instead = not graphrag_node_type_exists and kg_cols_status.get('content', False)
        
        graphrag_ready = stats['table_status'].get("RAG.KnowledgeGraphNodes", False) and graphrag_node_type_exists
        summary.append(f"   GraphRAG Evaluation Ready: {'✓ READY' if graphrag_ready else '✗ NOT READY'}")
        if not graphrag_node_type_exists and stats['table_status'].get("RAG.KnowledgeGraphNodes", False) :
            if graphrag_content_exists_instead:
                summary.append("     ↳ Reason: 'node_type' column missing, but 'content' column (old name) exists. Needs rename.")
            else:
                summary.append("     ↳ Reason: 'node_type' column missing in RAG.KnowledgeGraphNodes.")

        # RAGAS readiness check
        ragas_ready = stats['source_docs'] >= 1000
        summary.append(f"   RAGAS Evaluation Ready: {'✓ READY' if ragas_ready else '✗ NOT READY'}")
        summary.append("")
        
        # Recommendations
        summary.append("💡 RECOMMENDATIONS:")
        if not meets_1000_min:
            summary.append("   • Load more documents to meet 1000-document minimum")
        if stats['docs_with_embeddings'] < stats['source_docs']:
            missing_pct = ((stats['source_docs'] - stats['docs_with_embeddings']) / stats['source_docs']) * 100
            summary.append(f"   • Generate token embeddings for {missing_pct:.1f}% of documents")
        if not graphrag_ready and stats['table_status'].get("RAG.KnowledgeGraphNodes", False):
            if graphrag_content_exists_instead:
                summary.append("   • CRITICAL: Fix 'RAG.KnowledgeGraphNodes' schema - Rename 'content' column to 'node_type'.")
            elif not kg_cols_status.get('node_type', False): # This line was modified
                 summary.append("   • CRITICAL: Fix 'RAG.KnowledgeGraphNodes' schema - Add 'node_type' column.")
        elif not graphrag_ready and not stats['table_status'].get("RAG.KnowledgeGraphNodes", False):
             summary.append("   • CRITICAL: Create 'RAG.KnowledgeGraphNodes' table for GraphRAG.")

        if colbert_ready and ragas_ready and graphrag_ready:
            summary.append("   • Dataset is ready for comprehensive evaluation!")
        elif colbert_ready and ragas_ready:
            summary.append("   • Dataset is ready for most evaluations, but GraphRAG requires schema fix.")
        
        summary.append("=" * 80)
        
        return "\n".join(summary)
    
    def run_verification(self) -> Dict[str, Any]:
        """Run complete dataset state verification."""
        print("🔍 Starting IRIS Dataset State Verification")
        print("=" * 60)
        
        try:
            # Connect to database
            self.connect()
            
            # Verify table existence
            table_status = self.verify_table_existence()
            
            # If tables don't exist, return early
            if not all(table_status.values()):
                return {
                    'table_status': table_status,
                    'source_docs': 0,
                    'token_embeddings': 0,
                    'docs_with_embeddings': 0,
                    'missing_sample': [],
                    'kg_nodes_columns_status': {col: False for col in ["node_id", "node_name", "node_type", "embedding", "content"]}
                }

            kg_nodes_columns_status = {}
            if table_status.get("RAG.KnowledgeGraphNodes", False):
                 kg_nodes_columns_status = self.verify_knowledge_graph_nodes_columns()
            else:
                print("\nℹ️ Skipping RAG.KnowledgeGraphNodes column check as table is missing or inaccessible.")
                kg_nodes_columns_status = {col: False for col in ["node_id", "node_name", "node_type", "embedding", "content"]}

            # Count documents and embeddings
            source_docs = self.count_source_documents()
            token_embeddings = self.count_token_embeddings()
            docs_with_embeddings = self.count_documents_with_embeddings()
            
            # Find missing embeddings sample
            missing_sample = self.find_missing_embeddings_sample()
            
            # Compile results
            stats = {
                'table_status': table_status,
                'source_docs': source_docs,
                'token_embeddings': token_embeddings,
                'docs_with_embeddings': docs_with_embeddings,
                'missing_sample': missing_sample,
                'kg_nodes_columns_status': kg_nodes_columns_status
            }
            
            # Generate and print summary
            summary = self.generate_summary(stats)
            print(f"\n{summary}")
            
            return stats
            
        except Exception as e:
            print(f"✗ Verification failed: {e}")
            raise
        finally:
            self.disconnect()


def main():
    """Main function to run the dataset verification."""
    try:
        verifier = DatasetStateVerifier()
        stats = verifier.run_verification()
        
        # Exit with appropriate code
        if stats['source_docs'] >= 1000:
            print("\n✅ Dataset verification completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  Dataset verification completed with warnings!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Dataset verification failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()