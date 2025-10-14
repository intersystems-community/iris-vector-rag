"""Markdown report generation for audit findings."""

from models import AuditReport, Severity


class MarkdownReporter:
    """Generates markdown audit reports."""

    @staticmethod
    def generate_report(report: AuditReport) -> str:
        """Generate markdown report per research.md format."""
        lines = []

        # Header
        lines.append("# Makefile Target Consistency Audit Report\n")
        lines.append(
            f"**Generated**: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        lines.append(f"**Audited**: {report.makefile_path} ({report.total_lines} lines, {report.total_targets} targets)\n")
        lines.append(f"**Auditor Version**: {report.auditor_version}\n")
        lines.append("\n---\n\n")

        # Summary
        lines.append("## Summary\n\n")
        lines.append(f"- **Blocking**: {report.blocking_count} issues\n")
        lines.append(f"- **High**: {report.high_count} issues\n")
        lines.append(f"- **Medium**: {report.medium_count} issues\n")
        lines.append(f"- **Low**: {report.low_count} issues\n")
        lines.append(f"- **Total**: {report.total_issues} issues\n\n")

        # Group issues by severity
        issues_by_sev = report.issues_by_severity()

        # Blocking Issues
        lines.append("## Blocking Issues\n\n")
        if issues_by_sev[Severity.BLOCKING]:
            lines.append(
                "| Target | Line | Category | Description | Fix |\n"
            )
            lines.append("|--------|------|----------|-------------|-----|\n")
            for issue in issues_by_sev[Severity.BLOCKING]:
                lines.append(
                    f"| {issue.target_name} | {issue.line_number} | "
                    f"{issue.category.value} | {issue.description} | "
                    f"{issue.recommended_fix} |\n"
                )
            lines.append("\n")
        else:
            lines.append("*No blocking issues found.*\n\n")

        # High Severity Issues
        lines.append("## High Severity Issues\n\n")
        if issues_by_sev[Severity.HIGH]:
            lines.append(
                "| Target | Line | Category | Description | Fix |\n"
            )
            lines.append("|--------|------|----------|-------------|-----|\n")
            for issue in issues_by_sev[Severity.HIGH]:
                lines.append(
                    f"| {issue.target_name} | {issue.line_number} | "
                    f"{issue.category.value} | {issue.description} | "
                    f"{issue.recommended_fix} |\n"
                )
            lines.append("\n")
        else:
            lines.append("*No high severity issues found.*\n\n")

        # Medium Severity Issues
        lines.append("## Medium Severity Issues\n\n")
        if issues_by_sev[Severity.MEDIUM]:
            lines.append(
                "| Target | Line | Category | Description | Fix |\n"
            )
            lines.append("|--------|------|----------|-------------|-----|\n")
            for issue in issues_by_sev[Severity.MEDIUM]:
                lines.append(
                    f"| {issue.target_name} | {issue.line_number} | "
                    f"{issue.category.value} | {issue.description} | "
                    f"{issue.recommended_fix} |\n"
                )
            lines.append("\n")
        else:
            lines.append("*No medium severity issues found.*\n\n")

        # Low Severity Issues
        lines.append("## Low Severity Issues\n\n")
        if issues_by_sev[Severity.LOW]:
            lines.append(
                "| Target | Line | Category | Description | Fix |\n"
            )
            lines.append("|--------|------|----------|-------------|-----|\n")
            for issue in issues_by_sev[Severity.LOW]:
                lines.append(
                    f"| {issue.target_name} | {issue.line_number} | "
                    f"{issue.category.value} | {issue.description} | "
                    f"{issue.recommended_fix} |\n"
                )
            lines.append("\n")
        else:
            lines.append("*No low severity issues found.*\n\n")

        # Recommendations
        lines.append("## Recommendations\n\n")
        if report.blocking_count > 0:
            lines.append("1. **Fix blocking issues immediately** - targets are non-functional\n")
        if report.high_count > 0:
            lines.append("2. **Address high severity issues in current sprint** - violate framework principles or produce incorrect results\n")
        if report.medium_count > 0:
            lines.append("3. **Schedule medium issues for next sprint** - inconsistent patterns that should be standardized\n")
        if report.low_count > 0:
            lines.append("4. **Address low issues opportunistically** - documentation gaps and minor improvements\n")

        if report.total_issues == 0:
            lines.append("*No issues found - Makefile targets are consistent with framework principles!*\n")

        lines.append("\n---\n\n")
        lines.append("*Generated by Makefile Target Consistency Audit Tool*\n")

        return "".join(lines)
