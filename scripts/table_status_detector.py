#!/usr/bin/env python3
"""
Table Status Detector for Self-Healing Make System.

Detects current population status of all RAG tables and calculates
system-wide readiness percentage.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TableStatus:
    """Status information for a single table."""
    table_name: str
    record_count: int
    is_populated: bool
    last_updated: Optional[datetime]
    health_score: float  # 0.0-1.0
    dependencies_met: bool
    error: Optional[str] = None

@dataclass
class ReadinessReport:
    """Overall system readiness report."""
    overall_percentage: float
    populated_tables: int
    total_tables: int
    missing_tables: List[str]
    blocking_issues: List[str]
    table_details: Dict[str, TableStatus]

class TableStatusDetector:
    """
    Detects current population status of all RAG tables.
    """
    
    def __init__(self, db_connection):
        """
        Initialize the detector with database connection.
        
        Args:
            db_connection: Database connection object
        """
        self.db_connection = db_connection
        self.required_tables = [
            "RAG.SourceDocuments",
            "RAG.ColBERTTokenEmbeddings", 
            "RAG.ChunkedDocuments",
            "RAG.GraphRAGEntities",
            "RAG.GraphRAGRelationships",
            "RAG.KnowledgeGraphNodes",
            "RAG.DocumentEntities"
        ]
        self.table_status_cache = {}
        self.last_check_time = None
        self.cache_ttl_seconds = 300  # 5 minutes
        
        # Dependency mapping
        self.dependency_map = {
            "RAG.ChunkedDocuments": ["RAG.SourceDocuments"],
            "RAG.ColBERTTokenEmbeddings": ["RAG.SourceDocuments"],
            "RAG.GraphRAGEntities": ["RAG.SourceDocuments"],
            "RAG.GraphRAGRelationships": ["RAG.GraphRAGEntities"],
            "RAG.KnowledgeGraphNodes": ["RAG.GraphRAGEntities"],
            "RAG.DocumentEntities": ["RAG.SourceDocuments", "RAG.GraphRAGEntities"]
        }
    
    def detect_table_status(self) -> Dict[str, TableStatus]:
        """
        Detects current population status of all RAG tables.
        Returns comprehensive status for each table.
        """
        current_time = time.time()
        
        # Check cache validity
        if (self.last_check_time and 
            (current_time - self.last_check_time) < self.cache_ttl_seconds):
            logger.debug("Returning cached table status")
            return self.table_status_cache
        
        logger.info("Detecting table status for all RAG tables...")
        status_results = {}
        cursor = self.db_connection.cursor()
        
        try:
            for table_name in self.required_tables:
                try:
                    # Get record count
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    record_count = cursor.fetchone()[0]
                    
                    # Get last updated timestamp (if available)
                    last_updated = None
                    try:
                        # Try common timestamp column names
                        for col in ['created_at', 'updated_at', 'timestamp']:
                            try:
                                cursor.execute(f"SELECT MAX({col}) FROM {table_name}")
                                result = cursor.fetchone()
                                if result and result[0]:
                                    last_updated = result[0]
                                    break
                            except:
                                continue
                    except Exception as e:
                        logger.debug(f"Could not get timestamp for {table_name}: {e}")
                    
                    # Calculate health score based on record count and dependencies
                    health_score = self.calculate_table_health_score(table_name, record_count)
                    
                    # Check if dependencies are met
                    dependencies_met = self.check_table_dependencies(table_name, status_results)
                    
                    # Create TableStatus object
                    table_status = TableStatus(
                        table_name=table_name,
                        record_count=record_count,
                        is_populated=(record_count > 0),
                        last_updated=last_updated,
                        health_score=health_score,
                        dependencies_met=dependencies_met
                    )
                    
                    status_results[table_name] = table_status
                    logger.debug(f"Table {table_name}: {record_count} records, "
                               f"health: {health_score:.2f}, deps: {dependencies_met}")
                    
                except Exception as e:
                    logger.error(f"Failed to check status for {table_name}: {e}")
                    status_results[table_name] = TableStatus(
                        table_name=table_name,
                        record_count=0,
                        is_populated=False,
                        last_updated=None,
                        health_score=0.0,
                        dependencies_met=False,
                        error=str(e)
                    )
        finally:
            cursor.close()
        
        # Update cache
        self.table_status_cache = status_results
        self.last_check_time = current_time
        
        logger.info(f"Table status detection completed for {len(status_results)} tables")
        return status_results
    
    def calculate_overall_readiness(self) -> ReadinessReport:
        """
        Calculates system-wide readiness percentage and identifies issues.
        """
        logger.info("Calculating overall system readiness...")
        
        table_statuses = self.detect_table_status()
        total_tables = len(self.required_tables)
        populated_tables = 0
        missing_tables = []
        blocking_issues = []
        
        for table_name, status in table_statuses.items():
            if status.is_populated:
                populated_tables += 1
            else:
                missing_tables.append(table_name)
                
                # Check for blocking issues
                if not status.dependencies_met:
                    blocking_issues.append(f"Dependencies not met for {table_name}")
                
                if status.error:
                    blocking_issues.append(f"Error accessing {table_name}: {status.error}")
        
        overall_percentage = (populated_tables / total_tables) * 100
        
        logger.info(f"Overall readiness: {overall_percentage:.1f}% "
                   f"({populated_tables}/{total_tables} tables populated)")
        
        return ReadinessReport(
            overall_percentage=overall_percentage,
            populated_tables=populated_tables,
            total_tables=total_tables,
            missing_tables=missing_tables,
            blocking_issues=blocking_issues,
            table_details=table_statuses
        )
    
    def calculate_table_health_score(self, table_name: str, record_count: int) -> float:
        """
        Calculates health score (0.0-1.0) based on expected vs actual record count.
        """
        # Get source document count for baseline
        source_doc_count = self.get_source_document_count()
        
        # Define expected record counts based on source documents
        expected_counts = {
            "RAG.SourceDocuments": source_doc_count,
            "RAG.ChunkedDocuments": source_doc_count * 3,  # ~3 chunks per doc
            "RAG.ColBERTTokenEmbeddings": source_doc_count * 50,  # ~50 tokens per doc
            "RAG.GraphRAGEntities": source_doc_count * 10,  # ~10 entities per doc
            "RAG.GraphRAGRelationships": source_doc_count * 5,  # ~5 relationships per doc
            "RAG.KnowledgeGraphNodes": source_doc_count * 8,  # ~8 nodes per doc
            "RAG.DocumentEntities": source_doc_count * 12  # ~12 doc-entity links per doc
        }
        
        expected_count = expected_counts.get(table_name, source_doc_count)
        
        if expected_count == 0:
            return 1.0 if record_count == 0 else 0.0
        
        ratio = min(record_count / expected_count, 1.0)
        return ratio
    
    def check_table_dependencies(self, table_name: str, current_statuses: Dict) -> bool:
        """
        Checks if table dependencies are satisfied.
        """
        dependencies = self.dependency_map.get(table_name, [])
        
        for dep_table in dependencies:
            if dep_table in current_statuses:
                if not current_statuses[dep_table].is_populated:
                    return False
            else:
                # Check dependency directly if not in current batch
                dep_status = self.get_single_table_status(dep_table)
                if not dep_status.is_populated:
                    return False
        
        return True
    
    def get_single_table_status(self, table_name: str) -> TableStatus:
        """
        Gets status for a single table (used for dependency checking).
        """
        cursor = self.db_connection.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            record_count = cursor.fetchone()[0]
            
            return TableStatus(
                table_name=table_name,
                record_count=record_count,
                is_populated=(record_count > 0),
                last_updated=None,
                health_score=1.0 if record_count > 0 else 0.0,
                dependencies_met=True  # Simplified for dependency check
            )
        except Exception as e:
            return TableStatus(
                table_name=table_name,
                record_count=0,
                is_populated=False,
                last_updated=None,
                health_score=0.0,
                dependencies_met=False,
                error=str(e)
            )
        finally:
            cursor.close()
    
    def get_source_document_count(self) -> int:
        """
        Gets the current count of source documents.
        """
        cursor = self.db_connection.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            return cursor.fetchone()[0]
        except Exception as e:
            logger.warning(f"Could not get source document count: {e}")
            return 0
        finally:
            cursor.close()

def main():
    """CLI entry point for table status detection."""
    import sys
    sys.path.append('.')
    
    from common.iris_connection_manager import get_iris_connection
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Get database connection
        connection = get_iris_connection()
        if not connection:
            print("❌ Could not establish database connection")
            sys.exit(1)
        
        # Create detector and run analysis
        detector = TableStatusDetector(connection)
        report = detector.calculate_overall_readiness()
        
        # Print results
        print("=" * 60)
        print("📊 RAG SYSTEM TABLE STATUS REPORT")
        print("=" * 60)
        print(f"📈 Overall Readiness: {report.overall_percentage:.1f}% "
              f"({report.populated_tables}/{report.total_tables} tables)")
        print()
        
        print("📋 TABLE DETAILS:")
        for table_name, status in report.table_details.items():
            status_icon = "✅" if status.is_populated else "❌"
            deps_icon = "✅" if status.dependencies_met else "⚠️"
            print(f"  {status_icon} {table_name}: {status.record_count:,} records "
                  f"(health: {status.health_score:.2f}, deps: {deps_icon})")
            if status.error:
                print(f"    ⚠️ Error: {status.error}")
        
        if report.missing_tables:
            print()
            print("❌ MISSING TABLES:")
            for table in report.missing_tables:
                print(f"  - {table}")
        
        if report.blocking_issues:
            print()
            print("🚨 BLOCKING ISSUES:")
            for issue in report.blocking_issues:
                print(f"  - {issue}")
        
        print()
        if report.overall_percentage == 100.0:
            print("🎉 ALL TABLES POPULATED - SYSTEM READY!")
        else:
            print(f"🔧 SELF-HEALING NEEDED - {100 - report.overall_percentage:.1f}% remaining")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Table status detection failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()