#!/usr/bin/env python3
"""
Status Updater Script

This script parses test and benchmark results to automatically update
project status documentation, including individual component status logs
and the main project status dashboard.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATUS_LOGS_DIR = PROJECT_ROOT / "project_status_logs"
DASHBOARD_FILE = PROJECT_ROOT / "PROJECT_STATUS_DASHBOARD.md"
BENCHMARK_REPORTS_DIR = PROJECT_ROOT / "benchmark_reports" # Assuming this is where reports are stored

# Mapping from RAG technique short name (used in status files) to how they might appear in reports
TECHNIQUE_NAME_MAPPING = {
    "BasicRAG": ["BasicRAG", "basic_rag"],
    "ColBERT": ["ColBERT", "colbert"],
    "CRAG": ["CRAG", "crag"],
    "GraphRAG": ["GraphRAG", "graphrag"],
    "HyDE": ["HyDE", "hyde"],
    "HybridIFindRAG": ["HybridIFindRAG", "hybrid_ifind_rag"],
    "NodeRAG": ["NodeRAG", "noderag"],
    # Add other variations if necessary
}

def get_latest_benchmark_report(report_dir: Path, pattern: str = "*.json") -> Path | None:
    """Finds the most recent benchmark report file matching the pattern."""
    reports = list(report_dir.glob(pattern))
    if not reports:
        return None
    return max(reports, key=os.path.getctime)

def parse_benchmark_results(report_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Parses a benchmark JSON report to extract status for each RAG technique.
    This is a placeholder and will need to be adapted to the actual report format.
    
    Expected output format:
    {
        "BasicRAG": {"status": "WORKING", "details": "All tests passed.", "last_tested": "YYYY-MM-DD HH:MM:SS"},
        "ColBERT": {"status": "FAILING", "details": "Test X failed.", "last_tested": "YYYY-MM-DD HH:MM:SS"},
        ...
    }
    """
    parsed_data = {}
    if not report_file or not report_file.exists():
        print(f"Error: Benchmark report file not found: {report_file}")
        return parsed_data

    try:
        with open(report_file, 'r') as f:
            data = json.load(f)
        
        report_timestamp_str = data.get("run_timestamp", datetime.now().isoformat())
        report_datetime = datetime.fromisoformat(report_timestamp_str.replace("Z", "+00:00"))
        last_tested_formatted = report_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

        # Placeholder: Actual parsing logic will depend on the benchmark report structure
        # Example: Iterate through results in the report
        # For now, let's assume a simple structure where keys are technique names
        # and values contain success/failure info.
        
        # This needs to be adapted based on actual benchmark output structure.
        # For example, if benchmark output is like:
        # { "run_timestamp": "...", "results": [ {"name": "BasicRAG", "passed": true, ...}, ... ] }

        results_list = data.get("results", [])
        if isinstance(results_list, list): # RAGAS-like structure
            for item in results_list:
                tech_name_report = item.get("name", item.get("pipeline_name"))
                status = "WORKING" if item.get("passed", False) or item.get("success_rate", 0) > 0.9 else "FAILING"
                details = item.get("summary", "Details not available.")
                if not item.get("passed", True) and item.get("error"):
                    details = f"Error: {item.get('error')}"
                
                for short_name, variations in TECHNIQUE_NAME_MAPPING.items():
                    if tech_name_report in variations:
                        parsed_data[short_name] = {
                            "status": status,
                            "details": details,
                            "last_tested": last_tested_formatted,
                            "source_report": str(report_file.name)
                        }
                        break
        elif isinstance(results_list, dict): # If results is a dict keyed by technique
             for tech_name_report, tech_data in results_list.items():
                status = "WORKING" if tech_data.get("passed", False) or tech_data.get("success_rate", 0) > 0.9 else "FAILING"
                details = tech_data.get("summary", "Details not available.")
                if not tech_data.get("passed", True) and tech_data.get("error"):
                    details = f"Error: {tech_data.get('error')}"

                for short_name, variations in TECHNIQUE_NAME_MAPPING.items():
                    if tech_name_report in variations:
                         parsed_data[short_name] = {
                            "status": status,
                            "details": details,
                            "last_tested": last_tested_formatted,
                            "source_report": str(report_file.name)
                        }
                         break
        else: # Fallback for other structures, e.g. top-level keys are techniques
            for tech_name_report, tech_data in data.items():
                if not isinstance(tech_data, dict): continue # Skip non-dict top-level items

                status = "WORKING" if tech_data.get("passed", False) or tech_data.get("success_rate", 0) > 0.9 else "FAILING"
                details = tech_data.get("summary", "Details not available.")
                if not tech_data.get("passed", True) and tech_data.get("error"):
                    details = f"Error: {tech_data.get('error')}"
                
                for short_name, variations in TECHNIQUE_NAME_MAPPING.items():
                    if tech_name_report in variations:
                        parsed_data[short_name] = {
                            "status": status,
                            "details": details,
                            "last_tested": last_tested_formatted,
                            "source_report": str(report_file.name)
                        }
                        break
        
        print(f"Successfully parsed benchmark report: {report_file.name}")

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {report_file}")
    except Exception as e:
        print(f"Error parsing benchmark report {report_file}: {e}")
        
    return parsed_data


def update_component_status_log(component_short_name: str, status_info: Dict[str, Any]):
    """Updates the individual status log file for a component."""
    log_file = STATUS_LOGS_DIR / f"COMPONENT_STATUS_{component_short_name}.md"
    
    if not log_file.exists():
        print(f"Warning: Status log file not found for {component_short_name}, creating: {log_file}")
        # Create a basic structure if it doesn't exist
        header = f"""# Component Status: {component_short_name}

**Overall Status:** {status_info.get('status', 'UNKNOWN')}

## Status History
"""
        with open(log_file, 'w') as f:
            f.write(header)

    try:
        with open(log_file, 'r+') as f:
            content = f.read()
            
            # Create new entry
            new_entry = f"""
### {status_info['last_tested']}
- **Status:** {status_info['status']}
- **Details:** {status_info['details']}
- **Source:** Automated update from benchmark report `{status_info.get('source_report', 'N/A')}`
"""
            # Prepend new entry after the "Status History" header
            history_header = "## Status History"
            if history_header in content:
                parts = content.split(history_header, 1)
                new_content_before_history = parts[0]
                
                # Update overall status in the header part
                overall_status_regex = r"(\*\*Overall Status:\*\* )([A-Z_]+)"
                new_content_before_history = re.sub(overall_status_regex, rf"\1{status_info['status']}", new_content_before_history)
                
                new_content = new_content_before_history + history_header + new_entry + parts[1]
            else: # Should not happen if file was created correctly
                new_content = content + "\n" + history_header + new_entry

            f.seek(0)
            f.write(new_content)
            f.truncate()
        print(f"Updated status log for {component_short_name}")
    except Exception as e:
        print(f"Error updating status log for {component_short_name}: {e}")


def update_dashboard(all_statuses: Dict[str, Dict[str, Any]]):
    """Updates the main project status dashboard."""
    if not DASHBOARD_FILE.exists():
        print(f"Error: Dashboard file not found: {DASHBOARD_FILE}")
        return

    try:
        with open(DASHBOARD_FILE, 'r') as f:
            lines = f.readlines()

        updated_lines = []
        in_status_table = False
        header_found = False

        for line in lines:
            if "## Current RAG Technique Status" in line:
                in_status_table = True
                header_found = True
                updated_lines.append(line)
                # Add table headers if they are not already there or to ensure format
                updated_lines.append("| Technique         | Status          | Last Tested                 | Details (from latest report) |\n")
                updated_lines.append("|-------------------|-----------------|-----------------------------|------------------------------|\n")
                # Add new statuses from all_statuses
                for tech_name, info in sorted(all_statuses.items()):
                    status = info.get('status', 'UNKNOWN')
                    last_tested = info.get('last_tested', 'N/A')
                    details = info.get('details', 'N/A').replace('\n', ' ') # Ensure details are single line for table
                    details_link = f"[{details[:30]}...](project_status_logs/COMPONENT_STATUS_{tech_name}.md)" if len(details) > 30 else details
                    if not details_link.strip(): details_link = "N/A"
                    
                    # Ensure consistent column widths for better readability
                    updated_lines.append(f"| {tech_name:<17} | {status:<15} | {last_tested:<27} | {details_link} |\n")
                continue # Skip old table content

            if in_status_table and line.strip().startswith("|"):
                # This skips the old table data since we've rewritten it
                continue
            if in_status_table and not line.strip().startswith("|") and line.strip() != "":
                # End of old table, resume appending other lines
                in_status_table = False
            
            if not in_status_table:
                updated_lines.append(line)
        
        if not header_found: # If the status section was missing entirely
            updated_lines.append("\n## Current RAG Technique Status\n")
            updated_lines.append("| Technique         | Status          | Last Tested                 | Details (from latest report) |\n")
            updated_lines.append("|-------------------|-----------------|-----------------------------|------------------------------|\n")
            for tech_name, info in sorted(all_statuses.items()):
                status = info.get('status', 'UNKNOWN')
                last_tested = info.get('last_tested', 'N/A')
                details = info.get('details', 'N/A').replace('\n', ' ')
                details_link = f"[{details[:30]}...](project_status_logs/COMPONENT_STATUS_{tech_name}.md)" if len(details) > 30 else details
                if not details_link.strip(): details_link = "N/A"
                updated_lines.append(f"| {tech_name:<17} | {status:<15} | {last_tested:<27} | {details_link} |\n")


        with open(DASHBOARD_FILE, 'w') as f:
            f.writelines(updated_lines)
        print(f"Updated project status dashboard: {DASHBOARD_FILE}")

    except Exception as e:
        print(f"Error updating dashboard: {e}")


def main():
    """Main function to drive the status update process."""
    print("Starting project status update...")

    # 1. Find the latest benchmark report
    #    This needs to be adapted based on where reports are stored and their naming.
    #    Example: look for the latest RAGAS JSON report.
    latest_report_file = get_latest_benchmark_report(BENCHMARK_REPORTS_DIR, "ragas_*.json") 
    if not latest_report_file:
        # Try another common pattern if RAGAS not found
        latest_report_file = get_latest_benchmark_report(BENCHMARK_REPORTS_DIR, "comprehensive_benchmark_report_*.json")
    
    if not latest_report_file:
        print("No suitable benchmark report found. Exiting.")
        return

    print(f"Using benchmark report: {latest_report_file}")

    # 2. Parse the benchmark report
    component_statuses = parse_benchmark_results(latest_report_file)
    if not component_statuses:
        print("Failed to parse any component statuses from the report. Exiting.")
        return

    # 3. Update individual component status logs
    if not STATUS_LOGS_DIR.exists():
        STATUS_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created status logs directory: {STATUS_LOGS_DIR}")

    for component_name, status_info in component_statuses.items():
        update_component_status_log(component_name, status_info)

    # 4. Update the main project status dashboard
    update_dashboard(component_statuses)

    print("Project status update finished.")

if __name__ == "__main__":
    main()