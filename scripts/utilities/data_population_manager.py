#!/usr/bin/env python3
"""
Data Population Manager CLI for Self-Healing Make System.

Provides command-line interface for data population operations
integrated with make targets.
"""

import sys
import os
import argparse
import logging
import json
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class DataPopulationManager:
    """
    CLI manager for data population operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.db_connection = None
        self.orchestrator = None
        

    
    def check_status(self) -> Dict[str, Any]:
        """
        Check current table population status.
        
        Returns:
            Dictionary with status information
        """
        try:
            from scripts.utilities.table_status_detector import TableStatusDetector
            
            if not self.db_connection:
                if not self.initialize_connections():
                    return {"error": "Could not initialize database connection"}
            
            detector = TableStatusDetector(self.db_connection)
            report = detector.calculate_overall_readiness()
            
            return {
                "success": True,
                "overall_percentage": report.overall_percentage,
                "populated_tables": report.populated_tables,
                "total_tables": report.total_tables,
                "missing_tables": report.missing_tables,
                "blocking_issues": report.blocking_issues,
                "table_details": {
                    name: {
                        "record_count": status.record_count,
                        "is_populated": status.is_populated,
                        "health_score": status.health_score,
                        "dependencies_met": status.dependencies_met,
                        "error": status.error
                    }
                    for name, status in report.table_details.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {"error": str(e)}
    
    def populate_table(self, table_name: str) -> Dict[str, Any]:
        """
        Populate a specific table.
        
        Args:
            table_name: Name of table to populate
            
        Returns:
            Dictionary with population results
        """
        try:
            if not self.orchestrator:
                if not self.initialize_connections():
                    return {"error": "Could not initialize orchestrator"}
            
            logger.info(f"Starting population of table: {table_name}")
            
            # Use the orchestrator's population methods
            if hasattr(self.orchestrator, 'population_methods'):
                method = self.orchestrator.population_methods.get(table_name)
                if method:
                    success, count, details = method()
                    return {
                        "success": success,
                        "table_name": table_name,
                        "records_created": count,
                        "details": details
                    }
                else:
                    return {"error": f"No population method found for {table_name}"}
            else:
                return {"error": "Orchestrator not properly initialized"}
                
        except Exception as e:
            logger.error(f"Table population failed for {table_name}: {e}")
            return {"error": str(e)}
    
    def populate_missing(self) -> Dict[str, Any]:
        """
        Populate all missing tables in dependency order.
        
        Returns:
            Dictionary with population results
        """
        try:
            if not self.orchestrator:
                if not self.initialize_connections():
                    return {"error": "Could not initialize orchestrator"}
            
            logger.info("Starting population of all missing tables")
            
            # Get current status to identify missing tables
            status_result = self.check_status()
            if "error" in status_result:
                return status_result
            
            missing_tables = status_result.get("missing_tables", [])
            if not missing_tables:
                return {
                    "success": True,
                    "message": "All tables are already populated",
                    "populated_tables": []
                }
            
            logger.info(f"Found {len(missing_tables)} missing tables: {missing_tables}")
            
            # Populate tables in dependency order
            populated_tables = []
            failed_tables = []
            
            for table_name in self.orchestrator.TABLE_ORDER:
                if table_name in missing_tables:
                    logger.info(f"Populating {table_name}...")
                    result = self.populate_table(table_name)
                    
                    if result.get("success"):
                        populated_tables.append({
                            "table_name": table_name,
                            "records_created": result.get("records_created", 0),
                            "details": result.get("details", "")
                        })
                        logger.info(f"Successfully populated {table_name}")
                    else:
                        failed_tables.append({
                            "table_name": table_name,
                            "error": result.get("error", "Unknown error")
                        })
                        logger.error(f"Failed to populate {table_name}: {result.get('error')}")
            
            return {
                "success": len(failed_tables) == 0,
                "populated_tables": populated_tables,
                "failed_tables": failed_tables,
                "total_attempted": len(missing_tables)
            }
            
        except Exception as e:
            logger.error(f"Batch population failed: {e}")
            return {"error": str(e)}
    
    def validate_readiness(self, target_percentage: float = 100.0) -> Dict[str, Any]:
        """
        Validate system readiness against target.
        
        Args:
            target_percentage: Target readiness percentage (0-100)
            
        Returns:
            Dictionary with validation results
        """
        try:
            status_result = self.check_status()
            if "error" in status_result:
                return status_result
            
            current_percentage = status_result.get("overall_percentage", 0.0)
            target_met = current_percentage >= target_percentage
            
            return {
                "success": True,
                "target_met": target_met,
                "current_percentage": current_percentage,
                "target_percentage": target_percentage,
                "gap": max(0, target_percentage - current_percentage),
                "missing_tables": status_result.get("missing_tables", []),
                "blocking_issues": status_result.get("blocking_issues", [])
            }
            
        except Exception as e:
            logger.error(f"Readiness validation failed: {e}")
            return {"error": str(e)}

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def print_status_report(status: Dict[str, Any]):
    """Print formatted status report."""
    if "error" in status:
        print(f"❌ Error: {status['error']}")
        return
    
    print("=" * 60)
    print("📊 RAG SYSTEM STATUS REPORT")
    print("=" * 60)
    print(f"📈 Overall Readiness: {status['overall_percentage']:.1f}% "
          f"({status['populated_tables']}/{status['total_tables']} tables)")
    print()
    
    print("📋 TABLE DETAILS:")
    for table_name, details in status.get("table_details", {}).items():
        status_icon = "✅" if details["is_populated"] else "❌"
        deps_icon = "✅" if details["dependencies_met"] else "⚠️"
        print(f"  {status_icon} {table_name}: {details['record_count']:,} records "
              f"(health: {details['health_score']:.2f}, deps: {deps_icon})")
        if details.get("error"):
            print(f"    ⚠️ Error: {details['error']}")
    
    if status.get("missing_tables"):
        print()
        print("❌ MISSING TABLES:")
        for table in status["missing_tables"]:
            print(f"  - {table}")
    
    if status.get("blocking_issues"):
        print()
        print("🚨 BLOCKING ISSUES:")
        for issue in status["blocking_issues"]:
            print(f"  - {issue}")
    
    print("=" * 60)

def print_population_report(result: Dict[str, Any]):
    """Print formatted population report."""
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    if result.get("success"):
        print("✅ Population completed successfully!")
    else:
        print("⚠️ Population completed with some failures")
    
    populated = result.get("populated_tables", [])
    failed = result.get("failed_tables", [])
    
    if populated:
        print()
        print("✅ SUCCESSFULLY POPULATED:")
        for table in populated:
            print(f"  - {table['table_name']}: {table['records_created']:,} records")
            if table.get("details"):
                print(f"    Details: {table['details']}")
    
    if failed:
        print()
        print("❌ FAILED TO POPULATE:")
        for table in failed:
            print(f"  - {table['table_name']}: {table['error']}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Data Population Manager for Self-Healing Make System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                    # Check current status
  %(prog)s populate --table RAG.ChunkedDocuments  # Populate specific table
  %(prog)s populate --missing        # Populate all missing tables
  %(prog)s validate --target 80      # Validate 80% readiness
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check table population status")
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    
    # Populate command
    populate_parser = subparsers.add_parser("populate", help="Populate tables")
    populate_group = populate_parser.add_mutually_exclusive_group(required=True)
    populate_group.add_argument(
        "--table",
        help="Specific table to populate"
    )
    populate_group.add_argument(
        "--missing",
        action="store_true",
        help="Populate all missing tables"
    )
    populate_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate system readiness")
    validate_parser.add_argument(
        "--target",
        type=float,
        default=100.0,
        help="Target readiness percentage (default: 100.0)"
    )
    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Create manager
    manager = DataPopulationManager()
    
    try:
        if args.command == "status":
            result = manager.check_status()
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print_status_report(result)
        
        elif args.command == "populate":
            if args.table:
                result = manager.populate_table(args.table)
            else:  # --missing
                result = manager.populate_missing()
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print_population_report(result)
        
        elif args.command == "validate":
            result = manager.validate_readiness(args.target)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                if result.get("target_met"):
                    print(f"✅ Target readiness achieved: {result['current_percentage']:.1f}% >= {result['target_percentage']:.1f}%")
                else:
                    print(f"❌ Target readiness not met: {result['current_percentage']:.1f}% < {result['target_percentage']:.1f}%")
                    print(f"   Gap: {result['gap']:.1f}%")
                    if result.get("missing_tables"):
                        print(f"   Missing tables: {', '.join(result['missing_tables'])}")
        
        # Exit with appropriate code
        if "error" in result:
            sys.exit(1)
        elif args.command == "validate" and not result.get("target_met"):
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()