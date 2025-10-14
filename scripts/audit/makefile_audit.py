#!/usr/bin/env python3
"""Makefile Target Consistency Audit Tool

Performs static analysis of Makefile targets to detect configuration
inconsistencies, hardcoded values, and framework principle violations.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzers import analyze_all_issues
from models import AuditReport, Severity
from parsers import MakefileParser
from reporters import MarkdownReporter


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def filter_issues_by_severity(issues: list, min_severity: str) -> list:
    """Filter issues by minimum severity level."""
    severity_order = {
        "blocking": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
    }

    min_level = severity_order[min_severity]

    return [
        issue
        for issue in issues
        if severity_order[issue.severity.value] <= min_level
    ]


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Audit Makefile targets for configuration inconsistencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic audit with defaults
  %(prog)s

  # Custom Makefile and output location
  %(prog)s --makefile path/to/Makefile --output reports/audit.md

  # Only show high and blocking issues
  %(prog)s --min-severity high

  # Verbose output for debugging
  %(prog)s --verbose
        """,
    )

    parser.add_argument(
        "--makefile",
        default="./Makefile",
        help="Path to Makefile to audit (default: ./Makefile)",
    )
    parser.add_argument(
        "--output",
        default="specs/038-investigate-all-the/audit_report.md",
        help="Output path for markdown report (default: specs/038-investigate-all-the/audit_report.md)",
    )
    parser.add_argument(
        "--env-file",
        default="./.env",
        help="Path to .env file for configuration reference (default: ./.env)",
    )
    parser.add_argument(
        "--min-severity",
        choices=["blocking", "high", "medium", "low"],
        default="low",
        help="Minimum severity level to report (default: low)",
    )
    parser.add_argument(
        "--categories",
        default="all",
        help="Comma-separated issue categories to check, or 'all' (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Makefile Audit Tool 1.0.0",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Validate Makefile exists
        makefile_path = Path(args.makefile)
        if not makefile_path.exists():
            logger.error(f"Makefile not found at path: {args.makefile}")
            return 1

        logger.info(f"Starting Makefile audit")
        logger.info(f"Parsing: {makefile_path.absolute()}")

        # Parse Makefile
        parser_obj = MakefileParser(str(makefile_path.absolute()))
        targets = parser_obj.parse_makefile()

        total_lines = sum(
            1 for _ in open(makefile_path)
        )  # Count lines in Makefile

        logger.info(f"Found {len(targets)} targets")

        # Analyze issues
        logger.info("Analyzing targets for issues...")
        all_issues = analyze_all_issues(targets)

        # Filter by minimum severity
        filtered_issues = filter_issues_by_severity(all_issues, args.min_severity)

        # Create audit report
        report = AuditReport(
            makefile_path=str(makefile_path.absolute()),
            total_lines=total_lines,
            total_targets=len(targets),
            issues=filtered_issues,
            generated_at=datetime.now(),
            auditor_version="1.0.0",
        )

        logger.info(
            f"Analysis complete - found {report.total_issues} issues "
            f"(blocking: {report.blocking_count}, high: {report.high_count}, "
            f"medium: {report.medium_count}, low: {report.low_count})"
        )

        # Generate markdown report
        logger.info(f"Writing report to: {args.output}")
        markdown = MarkdownReporter.generate_report(report)

        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write report
        try:
            with open(output_path, "w") as f:
                f.write(markdown)
        except IOError as e:
            logger.error(
                f"Cannot write to output path: {args.output} ({e})"
            )
            return 3

        # Print summary to stdout (non-verbose mode)
        if not args.verbose:
            print(
                f"Auditing: {makefile_path.absolute()} ({total_lines} lines, {len(targets)} targets)"
            )
            print(
                f"Found {report.total_issues} issues: "
                f"{report.blocking_count} blocking, {report.high_count} high, "
                f"{report.medium_count} medium, {report.low_count} low"
            )
            print(f"Report written to: {args.output}")

        logger.info("Audit complete")
        return 0

    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except ValueError as e:
        logger.error(f"Invalid argument: {e}")
        return 2
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
