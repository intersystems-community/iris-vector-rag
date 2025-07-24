#!/usr/bin/env python3
"""
Reconciliation CLI for RAG Templates.

This module provides a command-line interface for the ReconciliationController,
implementing GitOps-style commands for managing RAG pipeline state reconciliation.

Commands:
- reconcile run: Execute reconciliation with optional force and dry-run modes
- reconcile status: Display current system status and drift analysis
- reconcile daemon: Run continuous reconciliation in daemon mode

Usage:
    python -m iris_rag.cli.reconcile_cli reconcile --help
    python -m iris_rag.cli.reconcile_cli reconcile run --pipeline colbert
    python -m iris_rag.cli.reconcile_cli reconcile status
    python -m iris_rag.cli.reconcile_cli reconcile daemon
"""

import sys
import logging
import click

from iris_rag.config.manager import ConfigurationManager
from iris_rag.controllers.reconciliation import ReconciliationController


# Configure logging for CLI
def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration for CLI operations."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_reconciliation_summary(result):
    """Print a formatted summary of reconciliation results."""
    click.echo("\n" + "="*60)
    click.echo("RECONCILIATION SUMMARY")
    click.echo("="*60)
    
    # Basic info
    click.echo(f"Reconciliation ID: {result.reconciliation_id}")
    click.echo(f"Success: {'✓' if result.success else '✗'}")
    click.echo(f"Execution Time: {format_duration(result.execution_time_seconds)}")
    
    # Current state
    click.echo(f"\nCurrent State:")
    click.echo(f"  Documents: {result.current_state.total_documents:,}")
    click.echo(f"  Token Embeddings: {result.current_state.total_token_embeddings:,}")
    
    # Desired state
    click.echo(f"\nDesired State:")
    click.echo(f"  Target Documents: {result.desired_state.target_document_count:,}")
    click.echo(f"  Embedding Model: {result.desired_state.embedding_model}")
    click.echo(f"  Vector Dimensions: {result.desired_state.vector_dimensions}")
    
    # Drift analysis
    click.echo(f"\nDrift Analysis:")
    click.echo(f"  Drift Detected: {'Yes' if result.drift_analysis.has_drift else 'No'}")
    click.echo(f"  Issues Found: {len(result.drift_analysis.issues)}")
    
    if result.drift_analysis.issues:
        click.echo(f"\n  Issues:")
        for issue in result.drift_analysis.issues:
            severity_icon = {"low": "ℹ", "medium": "⚠", "high": "⚠", "critical": "🚨"}.get(issue.severity, "•")
            click.echo(f"    {severity_icon} {issue.issue_type}: {issue.description}")
            if issue.affected_count > 0:
                click.echo(f"      Affected: {issue.affected_count:,} items")
    
    # Actions taken
    if result.actions_taken:
        click.echo(f"\nActions Taken: {len(result.actions_taken)}")
        for action in result.actions_taken:
            click.echo(f"  • {action.action_type}: {action.description}")
    
    # Convergence check
    if result.convergence_check:
        click.echo(f"\nConvergence:")
        click.echo(f"  Converged: {'✓' if result.convergence_check.converged else '✗'}")
        if result.convergence_check.remaining_issues:
            click.echo(f"  Remaining Issues: {len(result.convergence_check.remaining_issues)}")
    
    # Error information
    if result.error_message:
        click.echo(f"\nError: {result.error_message}")
    
    click.echo("="*60)


def print_status_summary(current_state, desired_state, drift_analysis):
    """Print a formatted summary of system status."""
    click.echo("\n" + "="*60)
    click.echo("SYSTEM STATUS")
    click.echo("="*60)
    
    # Current state
    click.echo(f"Current State:")
    click.echo(f"  Documents: {current_state.total_documents:,}")
    click.echo(f"  Token Embeddings: {current_state.total_token_embeddings:,}")
    click.echo(f"  Avg Embedding Size: {current_state.avg_embedding_size:.2f}")
    click.echo(f"  Observed At: {current_state.observed_at.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Quality issues
    quality = current_state.quality_issues
    click.echo(f"\nQuality Assessment:")
    click.echo(f"  Mock Embeddings Detected: {'Yes' if quality.mock_embeddings_detected else 'No'}")
    click.echo(f"  Diversity Score: {quality.avg_diversity_score:.3f}")
    click.echo(f"  Missing Embeddings: {quality.missing_embeddings_count:,}")
    click.echo(f"  Corrupted Embeddings: {quality.corrupted_embeddings_count:,}")
    
    # Desired state
    click.echo(f"\nDesired State:")
    click.echo(f"  Target Documents: {desired_state.target_document_count:,}")
    click.echo(f"  Embedding Model: {desired_state.embedding_model}")
    click.echo(f"  Vector Dimensions: {desired_state.vector_dimensions}")
    click.echo(f"  Diversity Threshold: {desired_state.diversity_threshold:.3f}")
    
    # Completeness requirements
    comp = desired_state.completeness_requirements
    click.echo(f"\nCompleteness Requirements:")
    click.echo(f"  Require All Docs: {'Yes' if comp.require_all_docs else 'No'}")
    click.echo(f"  Require Token Embeddings: {'Yes' if comp.require_token_embeddings else 'No'}")
    click.echo(f"  Min Quality Score: {comp.min_embedding_quality_score:.3f}")
    
    # Drift analysis
    click.echo(f"\nDrift Analysis:")
    click.echo(f"  Drift Detected: {'Yes' if drift_analysis.has_drift else 'No'}")
    click.echo(f"  Analysis Time: {drift_analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if drift_analysis.issues:
        click.echo(f"\n  Detected Issues ({len(drift_analysis.issues)}):")
        for issue in drift_analysis.issues:
            severity_icon = {"low": "ℹ", "medium": "⚠", "high": "⚠", "critical": "🚨"}.get(issue.severity, "•")
            click.echo(f"    {severity_icon} {issue.issue_type} ({issue.severity})")
            click.echo(f"      {issue.description}")
            if issue.affected_count > 0:
                click.echo(f"      Affected: {issue.affected_count:,} items")
            if issue.recommended_action:
                click.echo(f"      Recommended: {issue.recommended_action}")
    else:
        click.echo("  No drift issues detected")
    
    click.echo("="*60)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file')
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              help='Set logging level')
@click.pass_context
def reconcile(ctx, config, log_level):
    """
    RAG Pipeline Reconciliation CLI.
    
    Provides GitOps-style commands for managing RAG pipeline state reconciliation.
    Ensures data integrity and consistency across all RAG implementations.
    """
    # Setup logging
    setup_logging(log_level)
    
    # Initialize configuration manager
    try:
        config_manager = ConfigurationManager(config_path=config)
        ctx.ensure_object(dict)
        ctx.obj['config_manager'] = config_manager
        ctx.obj['log_level'] = log_level
    except Exception as e:
        click.echo(f"Error initializing configuration: {e}", err=True)
        sys.exit(1)


@reconcile.command()
@click.option('--pipeline', '-p', default='colbert', 
              type=click.Choice(['basic', 'colbert', 'noderag', 'graphrag', 'hyde', 'crag', 'hybrid_ifind']),
              help='Pipeline type to reconcile')
@click.option('--force', '-f', is_flag=True, help='Force reconciliation even if no drift detected')
@click.option('--dry-run', '-n', is_flag=True, help='Analyze drift without executing reconciliation actions')
@click.pass_context
def run(ctx, pipeline, force, dry_run):
    """
    Execute reconciliation for a specific pipeline.
    
    Performs the complete reconciliation cycle: observe current state,
    analyze drift, execute healing actions, and verify convergence.
    
    Examples:
        reconcile run --pipeline colbert
        reconcile run --pipeline basic --force
        reconcile run --pipeline noderag --dry-run
    """
    config_manager = ctx.obj['config_manager']
    
    try:
        # Initialize controller
        controller = ReconciliationController(config_manager)
        
        if dry_run:
            click.echo(f"Performing dry-run analysis for {pipeline} pipeline...")
            
            # Perform drift analysis only using controller method with pipeline-specific detection
            analysis_result = controller.analyze_drift_only(pipeline)
            current_state = analysis_result["current_state"]
            desired_state = analysis_result["desired_state"]
            drift_analysis = analysis_result["drift_analysis"]
            
            click.echo(f"\nDry-run completed for {pipeline} pipeline")
            print_status_summary(current_state, desired_state, drift_analysis)
            
            if drift_analysis.has_drift:
                click.echo(f"\n⚠ Drift detected! Run without --dry-run to execute reconciliation.")
                sys.exit(1)
            else:
                click.echo(f"\n✓ No drift detected. System is in desired state.")
        else:
            click.echo(f"Starting reconciliation for {pipeline} pipeline...")
            if force:
                click.echo("Force mode enabled - reconciliation will run regardless of drift detection")
            
            # Execute full reconciliation
            result = controller.reconcile(pipeline_type=pipeline, force=force)
            
            # Print results
            print_reconciliation_summary(result)
            
            # Exit with appropriate code
            if not result.success:
                sys.exit(1)
            elif result.drift_analysis.has_drift and not result.convergence_check.converged:
                click.echo("\n⚠ Reconciliation completed but convergence not achieved")
                sys.exit(2)
            else:
                click.echo(f"\n✓ Reconciliation completed successfully")
                
    except Exception as e:
        click.echo(f"Error during reconciliation: {e}", err=True)
        if ctx.obj['log_level'] == 'DEBUG':
            import traceback
            traceback.print_exc()
        sys.exit(1)


@reconcile.command()
@click.option('--pipeline', '-p', default='colbert',
              type=click.Choice(['basic', 'colbert', 'noderag', 'graphrag', 'hyde', 'crag', 'hybrid_ifind']),
              help='Pipeline type to check status for')
@click.option('--since', '-s', help='Filter status since time (e.g., "24h", "2023-01-01") - placeholder')
@click.pass_context
def status(ctx, pipeline, since):
    """
    Display current system status and drift analysis.
    
    Shows the current state of the system, desired state configuration,
    and any detected drift issues without executing reconciliation actions.
    
    Examples:
        reconcile status
        reconcile status --pipeline noderag
        reconcile status --since 24h
    """
    config_manager = ctx.obj['config_manager']
    
    try:
        # Initialize controller
        controller = ReconciliationController(config_manager)
        
        click.echo(f"Checking status for {pipeline} pipeline...")
        
        # Note: --since filtering is accepted but not implemented yet
        if since:
            click.echo(f"Note: --since filtering ({since}) is not yet implemented")
        
        # Use the controller's analyze_drift_only method which includes pipeline-specific detection
        analysis_result = controller.analyze_drift_only(pipeline)
        current_state = analysis_result["current_state"]
        desired_state = analysis_result["desired_state"]
        drift_analysis = analysis_result["drift_analysis"]
        
        # Print status summary
        print_status_summary(current_state, desired_state, drift_analysis)
        
        # Exit with appropriate code based on drift status
        if drift_analysis.has_drift:
            critical_issues = [issue for issue in drift_analysis.issues if issue.severity == 'critical']
            if critical_issues:
                sys.exit(2)  # Critical issues
            else:
                sys.exit(1)  # Non-critical drift
        else:
            sys.exit(0)  # No drift
            
    except Exception as e:
        click.echo(f"Error checking status: {e}", err=True)
        if ctx.obj['log_level'] == 'DEBUG':
            import traceback
            traceback.print_exc()
        sys.exit(1)


@reconcile.command()
@click.option('--pipeline', '-p', default='colbert',
              type=click.Choice(['basic', 'colbert', 'noderag', 'graphrag', 'hyde', 'crag', 'hybrid_ifind']),
              help='Pipeline type to monitor in daemon mode')
@click.option('--interval', '-i', default=3600, type=int,
              help='Reconciliation interval in seconds (default: 3600 = 1 hour)')
@click.option('--max-iterations', default=0, type=int,
              help='Maximum iterations (0 = infinite)')
@click.pass_context
def daemon(ctx, pipeline, interval, max_iterations):
    """
    Run continuous reconciliation in daemon mode.
    
    Continuously monitors the system and performs reconciliation at regular
    intervals. Useful for production environments requiring automatic healing.
    
    Examples:
        reconcile daemon --pipeline colbert
        reconcile daemon --interval 1800 --max-iterations 10
    """
    config_manager = ctx.obj['config_manager']
    
    try:
        # Initialize controller with interval override
        controller = ReconciliationController(config_manager, reconcile_interval_seconds=interval)
        
        click.echo(f"Starting reconciliation daemon for {pipeline} pipeline")
        click.echo(f"Interval: {interval} seconds ({format_duration(interval)})")
        if max_iterations > 0:
            click.echo(f"Max iterations: {max_iterations}")
        else:
            click.echo("Max iterations: infinite")
        click.echo("Press Ctrl+C to stop gracefully\n")
        
        # Use the controller's continuous reconciliation method
        controller.run_continuous_reconciliation(
            pipeline_type=pipeline,
            interval_seconds=interval,
            max_iterations=max_iterations
        )
        
        click.echo("\nDaemon stopped gracefully")
        
    except KeyboardInterrupt:
        click.echo("\nDaemon stopped by user")
    except Exception as e:
        click.echo(f"Error in daemon mode: {e}", err=True)
        if ctx.obj['log_level'] == 'DEBUG':
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    reconcile()