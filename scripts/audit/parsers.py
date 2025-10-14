"""Makefile parsing utilities."""

import re
from pathlib import Path
from typing import Dict, List, Optional

from models import MakeTarget


class MakefileParser:
    """Parser for GNU Makefiles."""

    # Regex patterns from research.md
    TARGET_PATTERN = re.compile(r"^([a-zA-Z0-9_-]+):\s*(.*)")
    VAR_ASSIGN_PATTERN = re.compile(r"^([A-Z_]+)\s*[:?]?=\s*(.+)$")
    ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_]+)(?::-([^}]+))?\}")
    HELP_TEXT_PATTERN = re.compile(r"##\s*(.+)")
    PHONY_PATTERN = re.compile(r"^\.PHONY:\s*(.+)")

    def __init__(self, makefile_path: str):
        """Initialize parser with Makefile path."""
        self.makefile_path = Path(makefile_path)
        if not self.makefile_path.exists():
            raise FileNotFoundError(f"Makefile not found: {makefile_path}")

    def parse_makefile(self) -> List[MakeTarget]:
        """Parse Makefile and return list of targets."""
        with open(self.makefile_path, "r") as f:
            lines = f.readlines()

        targets = []
        phony_targets = self._extract_phony_targets(lines)
        i = 0

        while i < len(lines):
            line = lines[i]
            match = self.TARGET_PATTERN.match(line.strip())

            if match:
                target = self._parse_target(lines, i, phony_targets)
                if target:
                    targets.append(target)
                    # Skip lines consumed by target parsing
                    i += self._count_target_lines(lines, i)
                    continue

            i += 1

        return targets

    def _extract_phony_targets(self, lines: List[str]) -> set:
        """Extract all .PHONY target declarations."""
        phony = set()
        for line in lines:
            match = self.PHONY_PATTERN.match(line.strip())
            if match:
                targets = match.group(1).split()
                phony.update(targets)
        return phony

    def _parse_target(
        self, lines: List[str], start_idx: int, phony_targets: set
    ) -> Optional[MakeTarget]:
        """Parse a single target starting at line index."""
        line = lines[start_idx].strip()
        match = self.TARGET_PATTERN.match(line)
        if not match:
            return None

        name = match.group(1)
        deps_str = match.group(2).strip()

        # Parse dependencies
        dependencies = deps_str.split() if deps_str else []

        # Extract help text from previous line
        help_text = None
        if start_idx > 0:
            prev_line = lines[start_idx - 1].strip()
            help_match = self.HELP_TEXT_PATTERN.search(prev_line)
            if help_match:
                help_text = help_match.group(1)

        # Parse commands (lines starting with tab)
        commands = []
        i = start_idx + 1
        while i < len(lines):
            line = lines[i]
            # Commands start with tab
            if line.startswith("\t"):
                # Handle line continuations (\\ at end)
                cmd = line[1:].rstrip()  # Remove leading tab
                while cmd.endswith("\\") and i + 1 < len(lines):
                    cmd = cmd[:-1]  # Remove backslash
                    i += 1
                    if lines[i].startswith("\t"):
                        cmd += lines[i][1:].rstrip()
                commands.append(cmd)
                i += 1
            elif line.strip() and not line.startswith("#"):
                # Non-empty, non-comment, non-command line = end of target
                break
            else:
                # Empty line or comment, continue
                i += 1

        # Extract environment variables from commands
        env_variables = self._extract_env_variables(commands)

        try:
            return MakeTarget(
                name=name,
                line_number=start_idx + 1,  # 1-indexed
                dependencies=dependencies,
                commands=commands,
                env_variables=env_variables,
                help_text=help_text,
                phony=name in phony_targets,
            )
        except ValueError as e:
            # Skip invalid targets
            print(f"Warning: Skipping invalid target at line {start_idx + 1}: {e}")
            return None

    def _count_target_lines(self, lines: List[str], start_idx: int) -> int:
        """Count lines consumed by target (for advancing parser)."""
        count = 1  # Target definition line
        i = start_idx + 1
        while i < len(lines):
            line = lines[i]
            if line.startswith("\t"):
                count += 1
                # Handle continuations
                while line.rstrip().endswith("\\") and i + 1 < len(lines):
                    i += 1
                    count += 1
                    line = lines[i]
                i += 1
            elif line.strip() and not line.startswith("#"):
                break
            else:
                count += 1
                i += 1
        return count

    def _extract_env_variables(self, commands: List[str]) -> Dict[str, str]:
        """Extract environment variable assignments from commands."""
        env_vars = {}

        for cmd in commands:
            # Look for export VAR=value or VAR=value patterns
            for match in re.finditer(
                r"(?:export\s+)?([A-Z_]+)=([^\s;]+)", cmd
            ):
                var_name = match.group(1)
                var_value = match.group(2)
                env_vars[var_name] = var_value

        return env_vars
