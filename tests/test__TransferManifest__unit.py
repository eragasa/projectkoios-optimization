import hashlib
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).parents[1]
CANDIDATE_COMMIT = "879996761b67f8b78dc30dead3f03d28ff7cbea6"


def test_affine_metric_candidate_provenance_is_exact_and_parseable() -> None:
    transfer = tomllib.loads((ROOT / "TRANSFER.toml").read_text(encoding="utf-8"))

    source = next(
        item for item in transfer["sources"] if item["role"] == "owner-export-candidate"
    )
    assert source == {
        "role": "owner-export-candidate",
        "repository_path": (
            "/Users/eugene/worktrees/ksdft2effmass-hsh-owner-export-foundations"
        ),
        "commit": CANDIDATE_COMMIT,
        "root_tree": "80f994634d823aa5bca426d7451564f60d113c4e",
        "relevant_tree": "5e90d8596731dec6cffc29533830c0332d465eed",
        "status": (
            "reviewed source candidate; adapted by the Optimization owner before "
            "acceptance"
        ),
    }

    exported = {
        item["source_path"]: (item["blob"], item["sha256"])
        for item in transfer["files"]
        if item["source_path"].startswith("exports/optimization/")
    }
    assert exported == {
        "exports/optimization/affine.py": (
            "9739498c4e071d6d45c3f6f8abc5f9339d302112",
            "1c2473e40aa94a466c4b1b92605a2f8b83ba583853d43970dc8e172ef08e10ad",
        ),
        "exports/optimization/metric.py": (
            "191fb449f1f797fe3b288f1aceffa3cbd8725fc0",
            "576e84d81abc245a5f37b05e79a6c7ee4cfbccb240e73a5e6fc3bc139e227efc",
        ),
    }

    targets = (
        target
        for section in (transfer["files"], transfer["tests"])
        for item in section
        for target in item["targets"]
    )
    assert all((ROOT / target).exists() for target in targets)


def test_destination_license_retains_recorded_apache_bytes() -> None:
    digest = hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest()

    assert digest == "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
