#!/usr/bin/env python3
"""Load the medical-graphrag-20 fixture into an IRIS database.

The same loader the test suite uses (tests/fixtures/manager.py): it creates the
RAG tables when they are missing, validates the checksum of the source data
(tests/fixtures/graphrag/medical_docs.json), and inserts 3 documents, 21 entities and
15 relationships. The connection comes from IRIS_HOST, IRIS_PORT, IRIS_USERNAME,
IRIS_PASSWORD and IRIS_NAMESPACE, as everywhere else in this repository.

    python scripts/fixtures/create_graphrag_dat_fixture.py --cleanup-first
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

FIXTURE_NAME = "medical-graphrag-20"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cleanup-first",
        action="store_true",
        help="delete the rows already in the fixture's tables before loading",
    )
    args = parser.parse_args(argv)

    from tests.fixtures.manager import FixtureManager
    from tests.support.iris_environment import IrisTarget, connect

    target = IrisTarget(
        container=os.environ.get("IRIS_CONTAINER_NAME", ""),
        host=os.environ.get("IRIS_HOST", "localhost"),
        port=int(os.environ["IRIS_PORT"]),
        namespace=os.environ.get("IRIS_NAMESPACE", "USER"),
        username=os.environ.get("IRIS_USERNAME", "_SYSTEM"),
        password=os.environ.get("IRIS_PASSWORD", "SYS"),
        ephemeral=False,
    )
    connection = connect(target)
    try:
        manager = FixtureManager(
            fixtures_root=REPO_ROOT / "tests" / "fixtures", connection=connection
        )
        result = manager.load_fixture(FIXTURE_NAME, cleanup_first=args.cleanup_first)
    finally:
        connection.close()

    if not result.success:
        print(f"Loading {FIXTURE_NAME} failed: {result.error_message}", file=sys.stderr)
        return 1
    print(f"Loaded {FIXTURE_NAME} v{result.fixture_version}: {result.rows_loaded} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
