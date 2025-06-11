"""
Self-Healing Orchestrator for RAG System.

Main orchestrator for self-healing data population workflow coordination.
Integrates with existing DataPopulationOrchestrator and provides
comprehensive error recovery and progress tracking.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SelfHealingResult:
    """Result of self-healing cycle execution."""
    success: bool
    initial_readiness: float
    final_readiness: float
    tables_populated: List[str]
    errors_encountered: List[str]
    execution_time: float
    recommendations: List[str]
    cycles_executed: int

class SelfHealingOrchestrator:
    """
    Main orchestrator for self-healing data population.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.max_healing_cycles = self.config.get('max_healing_cycles', 3)
        self.healing_timeout_minutes = self.config.get('healing_timeout_minutes', 30)
        self.force_repopulation = self.config.get('force_repopulation', False)
        
        # Initialize components
        self.detector = None
        self.population_orchestrator = None
        self.db_connection = None
        
        # Initialize connections
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize required components."""
        try:
            from common.iris_connection_manager import get_iris_connection
            from scripts.table_status_detector import TableStatusDetector
            from rag_templates.validation.data_population_orchestrator import DataPopulationOrchestrator
            
            # Get database connection
            self.db_connection = get_iris_connection()
            if not self.db_connection:
                raise Exception("Could not establish database connection")
            
            # Initialize components
            self.detector = TableStatusDetector(self.db_connection)
            self.population_orchestrator = DataPopulationOrchestrator(
                config=self.config,
                db_connection=self.db_connection
            )
            
            logger.info("Self-healing orchestrator components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run_self_healing_cycle(self) -> SelfHealingResult:
        """
        Executes complete self-healing cycle with comprehensive error recovery.
        """
        start_time = time.time()
        cycle_count = 0
        initial_readiness = None
        final_readiness = None
        tables_populated = []
        errors_encountered = []
        recommendations = []
        
        try:
            logger.info("Starting self-healing cycle...")
            
            # Step 1: Detect current table status
            logger.info("Step 1: Detecting current table status...")
            initial_report = self.detector.calculate_overall_readiness()
            initial_readiness = initial_report.overall_percentage
            
            logger.info(f"Initial readiness: {initial_readiness:.1f}%")
            
            # Step 2: Analyze dependencies and missing data
            logger.info("Step 2: Analyzing dependencies and missing data...")
            missing_tables = initial_report.missing_tables
            blocking_issues = initial_report.blocking_issues
            
            if blocking_issues:
                logger.warning(f"Blocking issues found: {blocking_issues}")
                recommendations.append("Resolve blocking issues before population")
                for issue in blocking_issues:
                    errors_encountered.append(f"Blocking issue: {issue}")
            
            if not missing_tables:
                logger.info("All tables are already populated")
                final_readiness = initial_readiness
                return SelfHealingResult(
                    success=True,
                    initial_readiness=initial_readiness,
                    final_readiness=final_readiness,
                    tables_populated=[],
                    errors_encountered=[],
                    execution_time=time.time() - start_time,
                    recommendations=["System is already at 100% readiness"],
                    cycles_executed=0
                )
            
            # Step 3: Execute population tasks with error recovery
            logger.info(f"Step 3: Executing population for {len(missing_tables)} missing tables...")
            
            while cycle_count < self.max_healing_cycles and missing_tables:
                cycle_count += 1
                logger.info(f"Healing cycle {cycle_count}/{self.max_healing_cycles}")
                
                # Check timeout
                if (time.time() - start_time) > (self.healing_timeout_minutes * 60):
                    logger.error("Healing timeout exceeded")
                    errors_encountered.append("Healing timeout exceeded")
                    break
                
                cycle_success = True
                cycle_populated = []
                
                # Process tables in dependency order
                for table_name in self.population_orchestrator.TABLE_ORDER:
                    if table_name not in missing_tables:
                        continue
                    
                    logger.info(f"Populating {table_name}...")
                    
                    try:
                        # Get population method
                        method = self.population_orchestrator.population_methods.get(table_name)
                        if not method:
                            error_msg = f"No population method found for {table_name}"
                            logger.error(error_msg)
                            errors_encountered.append(error_msg)
                            cycle_success = False
                            continue
                        
                        # Execute population
                        success, count, details = method()
                        
                        if success:
                            logger.info(f"Successfully populated {table_name}: {count} records")
                            cycle_populated.append(table_name)
                            if table_name not in tables_populated:
                                tables_populated.append(table_name)
                        else:
                            error_msg = f"Failed to populate {table_name}: {details}"
                            logger.error(error_msg)
                            errors_encountered.append(error_msg)
                            cycle_success = False
                            
                    except Exception as e:
                        error_msg = f"Exception during {table_name} population: {str(e)}"
                        logger.error(error_msg)
                        errors_encountered.append(error_msg)
                        cycle_success = False
                
                # Re-evaluate status after cycle
                current_report = self.detector.calculate_overall_readiness()
                missing_tables = current_report.missing_tables
                current_readiness = current_report.overall_percentage
                
                logger.info(f"Cycle {cycle_count} completed. Readiness: {current_readiness:.1f}%")
                
                if not missing_tables:
                    logger.info("All tables populated successfully!")
                    break
                
                if not cycle_success and not self.force_repopulation:
                    logger.warning(f"Cycle {cycle_count} had failures, stopping...")
                    break
                elif not cycle_success:
                    logger.warning(f"Cycle {cycle_count} had failures, but force_repopulation enabled, continuing...")
            
            # Step 4: Final validation and recommendations
            logger.info("Step 4: Final validation and recommendations...")
            final_report = self.detector.calculate_overall_readiness()
            final_readiness = final_report.overall_percentage
            
            # Generate recommendations
            if final_readiness < 100.0:
                recommendations.append(f"Consider manual intervention for remaining tables: {final_report.missing_tables}")
                
                for table in final_report.missing_tables:
                    table_status = final_report.table_details.get(table)
                    if table_status:
                        if not table_status.dependencies_met:
                            recommendations.append(f"Resolve dependencies for {table}")
                        if table_status.error:
                            recommendations.append(f"Fix error for {table}: {table_status.error}")
            
            execution_time = time.time() - start_time
            success = (final_readiness >= initial_readiness) and (len([e for e in errors_encountered if "Blocking issue" not in e]) == 0)
            
            logger.info(f"Self-healing cycle completed. Success: {success}, Duration: {execution_time:.1f}s")
            
            return SelfHealingResult(
                success=success,
                initial_readiness=initial_readiness,
                final_readiness=final_readiness,
                tables_populated=tables_populated,
                errors_encountered=errors_encountered,
                execution_time=execution_time,
                recommendations=recommendations,
                cycles_executed=cycle_count
            )
            
        except Exception as e:
            logger.error(f"Self-healing cycle failed with exception: {e}")
            execution_time = time.time() - start_time
            
            return SelfHealingResult(
                success=False,
                initial_readiness=initial_readiness or 0.0,
                final_readiness=initial_readiness or 0.0,
                tables_populated=tables_populated,
                errors_encountered=[str(e)] + errors_encountered,
                execution_time=execution_time,
                recommendations=["Manual intervention required due to system error"],
                cycles_executed=cycle_count
            )
    
    def detect_and_heal(self, target_readiness: float = 1.0) -> bool:
        """
        Simplified interface for make targets.
        Returns True if target readiness achieved.
        
        Args:
            target_readiness: Desired readiness percentage (0.0-1.0)
            
        Returns:
            True if target readiness achieved, False otherwise
        """
        logger.info(f"Starting healing to achieve {target_readiness * 100:.1f}% readiness")
        
        try:
            result = self.run_self_healing_cycle()
            achieved_readiness = result.final_readiness / 100.0
            
            if achieved_readiness >= target_readiness:
                logger.info(f"Target readiness achieved: {achieved_readiness * 100:.1f}%")
                return True
            else:
                logger.warning(f"Target readiness not achieved: {achieved_readiness * 100:.1f}% < {target_readiness * 100:.1f}%")
                return False
                
        except Exception as e:
            logger.error(f"Healing failed with exception: {e}")
            return False
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            Dictionary with current status information
        """
        try:
            if not self.detector:
                self._initialize_components()
            
            report = self.detector.calculate_overall_readiness()
            
            return {
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
            logger.error(f"Failed to get current status: {e}")
            return {"error": str(e)}

def main():
    """CLI entry point for self-healing orchestrator."""
    import sys
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Self-Healing Orchestrator for RAG System"
    )
    parser.add_argument(
        "--target",
        type=float,
        default=1.0,
        help="Target readiness (0.0-1.0, default: 1.0)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force repopulation even on failures"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in minutes (default: 30)"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only check status, don't perform healing"
    )
    
    args = parser.parse_args()
    
    try:
        # Create orchestrator with config
        config = {
            'force_repopulation': args.force,
            'healing_timeout_minutes': args.timeout
        }
        
        orchestrator = SelfHealingOrchestrator(config)
        
        if args.status_only:
            # Just check status
            status = orchestrator.get_current_status()
            if "error" in status:
                print(f"❌ Error: {status['error']}")
                sys.exit(1)
            
            print("=" * 60)
            print("📊 CURRENT SYSTEM STATUS")
            print("=" * 60)
            print(f"📈 Overall Readiness: {status['overall_percentage']:.1f}% "
                  f"({status['populated_tables']}/{status['total_tables']} tables)")
            
            if status.get("missing_tables"):
                print(f"❌ Missing Tables: {', '.join(status['missing_tables'])}")
            
            if status.get("blocking_issues"):
                print("🚨 Blocking Issues:")
                for issue in status["blocking_issues"]:
                    print(f"  - {issue}")
            
            print("=" * 60)
        else:
            # Perform healing
            success = orchestrator.detect_and_heal(args.target)
            
            if success:
                print("✅ Self-healing completed successfully!")
                sys.exit(0)
            else:
                print("❌ Self-healing failed to achieve target readiness")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Self-healing orchestrator failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()