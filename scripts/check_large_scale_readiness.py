#!/usr/bin/env python3
"""
Large-Scale Testing Readiness Check

This script verifies the system is ready for 10K+ document testing by:
1. Checking IRIS Enterprise edition configuration
2. Counting current documents in the database
3. Verifying database capacity and performance
4. Checking system resources
"""

import os
import sys
import time
import psutil
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.iris_connection_manager import get_iris_connection


def check_iris_edition() -> Dict[str, Any]:
    """Check IRIS edition and configuration."""
    print("🔍 Checking IRIS Edition Configuration...")
    
    # Check environment variable
    iris_image = os.getenv('IRIS_DOCKER_IMAGE', 'containers.intersystems.com/intersystems/iris-arm64:latest-em')
    is_enterprise = 'em' in iris_image.lower() or 'enterprise' in iris_image.lower()
    
    print(f"   Docker Image: {iris_image}")
    print(f"   Enterprise Edition: {'✅ YES' if is_enterprise else '❌ NO (Community)'}")
    
    return {
        'docker_image': iris_image,
        'is_enterprise': is_enterprise,
        'recommended_for_large_scale': is_enterprise
    }


def check_database_connection() -> Dict[str, Any]:
    """Check database connection and basic functionality."""
    print("\n🔗 Checking Database Connection...")
    
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Test basic connectivity
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchone()
        
        # Test vector functionality
        cursor.execute("SELECT TO_VECTOR('[1,2,3]') as vector_test")
        vector_result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        print("   ✅ Database connection successful")
        print("   ✅ Vector functionality working")
        
        return {
            'connection_status': 'success',
            'vector_support': True,
            'error': None
        }
        
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        return {
            'connection_status': 'failed',
            'vector_support': False,
            'error': str(e)
        }


def count_documents() -> Dict[str, Any]:
    """Count documents in various tables."""
    print("\n📊 Counting Current Documents...")
    
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Count documents in main tables
        tables_to_check = [
            'SourceDocuments',
            'DocumentChunks', 
            'BasicRAGDocuments',
            'ColBERTDocuments',
            'CRAGDocuments',
            'SQLRAGDocuments',
            'HyDERAGDocuments',
            'HybridIFindDocuments'
        ]
        
        document_counts = {}
        total_documents = 0
        
        for table in tables_to_check:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                document_counts[table] = count
                print(f"   {table}: {count:,} documents")
                
                # Use SourceDocuments as the primary count
                if table == 'SourceDocuments':
                    total_documents = count
                    
            except Exception as e:
                document_counts[table] = 0
                print(f"   {table}: Table not found or error - {e}")
        
        cursor.close()
        conn.close()
        
        print(f"\n   📈 Total Source Documents: {total_documents:,}")
        
        # Determine readiness for large-scale testing
        ready_for_10k = total_documents >= 10000
        ready_for_1k = total_documents >= 1000
        
        if ready_for_10k:
            print("   ✅ Ready for 10K+ document testing")
        elif ready_for_1k:
            print("   ⚠️  Ready for 1K+ testing, need more docs for 10K+")
        else:
            print("   ❌ Need more documents for large-scale testing")
        
        return {
            'document_counts': document_counts,
            'total_documents': total_documents,
            'ready_for_10k': ready_for_10k,
            'ready_for_1k': ready_for_1k
        }
        
    except Exception as e:
        print(f"   ❌ Failed to count documents: {e}")
        return {
            'document_counts': {},
            'total_documents': 0,
            'ready_for_10k': False,
            'ready_for_1k': False,
            'error': str(e)
        }


def check_system_resources() -> Dict[str, Any]:
    """Check system resources for large-scale testing."""
    print("\n💻 Checking System Resources...")
    
    # Memory check
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    
    print(f"   Total Memory: {memory_gb:.1f} GB")
    print(f"   Available Memory: {available_gb:.1f} GB")
    
    # Disk space check
    disk = psutil.disk_usage('/')
    disk_free_gb = disk.free / (1024**3)
    
    print(f"   Free Disk Space: {disk_free_gb:.1f} GB")
    
    # CPU check
    cpu_count = psutil.cpu_count()
    print(f"   CPU Cores: {cpu_count}")
    
    # Recommendations for large-scale testing
    memory_sufficient = memory_gb >= 16  # 16GB+ recommended
    disk_sufficient = disk_free_gb >= 50  # 50GB+ recommended
    cpu_sufficient = cpu_count >= 4  # 4+ cores recommended
    
    overall_sufficient = memory_sufficient and disk_sufficient and cpu_sufficient
    
    if overall_sufficient:
        print("   ✅ System resources sufficient for large-scale testing")
    else:
        print("   ⚠️  System resources may be limited for large-scale testing")
        if not memory_sufficient:
            print("      - Consider increasing memory (16GB+ recommended)")
        if not disk_sufficient:
            print("      - Consider freeing disk space (50GB+ recommended)")
        if not cpu_sufficient:
            print("      - Consider using more CPU cores (4+ recommended)")
    
    return {
        'memory_gb': memory_gb,
        'available_memory_gb': available_gb,
        'free_disk_gb': disk_free_gb,
        'cpu_cores': cpu_count,
        'memory_sufficient': memory_sufficient,
        'disk_sufficient': disk_sufficient,
        'cpu_sufficient': cpu_sufficient,
        'overall_sufficient': overall_sufficient
    }


def check_database_performance() -> Dict[str, Any]:
    """Check database performance with sample operations."""
    print("\n⚡ Checking Database Performance...")
    
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Test vector operation performance
        start_time = time.time()
        cursor.execute("SELECT VECTOR_COSINE(TO_VECTOR('[1,2,3]'), TO_VECTOR('[4,5,6]')) as similarity")
        result = cursor.fetchone()
        vector_time = time.time() - start_time
        
        print(f"   Vector operation time: {vector_time*1000:.2f}ms")
        
        # Test simple query performance
        start_time = time.time()
        cursor.execute("SELECT COUNT(*) FROM %Dictionary.ClassDefinition")
        result = cursor.fetchone()
        query_time = time.time() - start_time
        
        print(f"   Simple query time: {query_time*1000:.2f}ms")
        
        cursor.close()
        conn.close()
        
        # Performance assessment
        vector_fast = vector_time < 0.1  # < 100ms
        query_fast = query_time < 0.05   # < 50ms
        
        if vector_fast and query_fast:
            print("   ✅ Database performance looks good")
        else:
            print("   ⚠️  Database performance may be slow")
        
        return {
            'vector_operation_time': vector_time,
            'simple_query_time': query_time,
            'vector_performance_good': vector_fast,
            'query_performance_good': query_fast,
            'overall_performance_good': vector_fast and query_fast
        }
        
    except Exception as e:
        print(f"   ❌ Performance check failed: {e}")
        return {
            'vector_operation_time': None,
            'simple_query_time': None,
            'vector_performance_good': False,
            'query_performance_good': False,
            'overall_performance_good': False,
            'error': str(e)
        }


def main():
    """Main function to run all readiness checks."""
    print("🚀 Large-Scale Testing Readiness Check")
    print("=" * 50)
    
    # Run all checks
    edition_check = check_iris_edition()
    connection_check = check_database_connection()
    document_check = count_documents()
    resource_check = check_system_resources()
    performance_check = check_database_performance()
    
    # Overall assessment
    print("\n" + "=" * 50)
    print("📋 OVERALL READINESS ASSESSMENT")
    print("=" * 50)
    
    ready_for_large_scale = (
        edition_check['is_enterprise'] and
        connection_check['connection_status'] == 'success' and
        document_check['ready_for_10k'] and
        resource_check['overall_sufficient'] and
        performance_check['overall_performance_good']
    )
    
    if ready_for_large_scale:
        print("✅ SYSTEM IS READY FOR 10K+ DOCUMENT TESTING")
    else:
        print("⚠️  SYSTEM NEEDS PREPARATION FOR 10K+ DOCUMENT TESTING")
        
        # Specific recommendations
        if not edition_check['is_enterprise']:
            print("   - Switch to IRIS Enterprise edition")
        if connection_check['connection_status'] != 'success':
            print("   - Fix database connection issues")
        if not document_check['ready_for_10k']:
            print(f"   - Load more documents (current: {document_check['total_documents']:,}, need: 10,000+)")
        if not resource_check['overall_sufficient']:
            print("   - Upgrade system resources")
        if not performance_check['overall_performance_good']:
            print("   - Optimize database performance")
    
    # Return summary for potential automation
    return {
        'ready_for_large_scale': ready_for_large_scale,
        'edition_check': edition_check,
        'connection_check': connection_check,
        'document_check': document_check,
        'resource_check': resource_check,
        'performance_check': performance_check
    }


if __name__ == "__main__":
    main()