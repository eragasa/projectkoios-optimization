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


def test_gmm_clustering_recovery_provenance_is_exact() -> None:
    transfer = tomllib.loads((ROOT / "TRANSFER.toml").read_text(encoding="utf-8"))

    source = next(
        item
        for item in transfer["files"]
        if item["source_path"] == "python/src/ksdft2effmass/application/engines/gmm.py"
    )
    assert source["blob"] == "d125af3779452aebb91f2e93fd29c78ed48fb6a1"
    assert source["sha256"] == (
        "f879ca06255dd1a0efbad07ffaa93535c762e47a60cd1e5283cbcedff35380ba"
    )
    evidence = next(
        item for item in transfer["tests"] if item["role"] == "numerical-gmm"
    )
    assert evidence["commit"] == "ff50d06e74ad094ec84e53149d1717d66e2154fa"
    assert evidence["source_blobs"] == ["ceb36b425334f30594000d73256a44ee055c75e6"]
    assert evidence["source_sha256"] == [
        "bc03e897edb6e5e19e715d32d94d7179005b52d663110e653f443a5a966daa5b"
    ]


def test_pareto_historical_mechanism_authorities_are_explicit() -> None:
    transfer = tomllib.loads((ROOT / "TRANSFER.toml").read_text(encoding="utf-8"))
    sources = {item["role"]: item for item in transfer["sources"]}

    assert sources["pareto-mechanism-authority"] == {
        "role": "pareto-mechanism-authority",
        "repository": "https://github.com/eragasa/pypospack",
        "commit": "be453fa7191e55a0426f66e8b5b5b0b103c8b29d",
        "root_tree": "7ac9c9f255aa7731f39ce35a0e561fc113082a6f",
        "status": (
            "authoritative source for the transferred Pareto selector and iterative "
            "sampling policy; exact selector blob also preserved in recovery-primary"
        ),
    }
    assert sources["pareto-mechanism-context-authority"] == {
        "role": "pareto-mechanism-context-authority",
        "repository": "https://github.com/eragasa/pymatmc2",
        "commit": "9d31d7fd4f8902f17864fbf391059101a3f5afda",
        "root_tree": "7773f886eaecfe919abf68d9e4f990fb398a82be",
        "historical_source_commit": "3063f031301c04e395cb9d7bc5064f2a52e1110c",
        "historical_source_tree": "fd9788560365e93faae596ac91f2530aa6bd92a8",
        "historical_mexm_commit": "3963a1a30ba1595fba69cd0f791dc31fe2e08f7c",
        "historical_mexm_tree": "a19cd86088aad5ecb0b3bbf8a4c73e0159a22fd3",
        "status": (
            "more authoritative historical mechanism context than KSD recovery "
            "copies; no PyMatMC2 source transferred"
        ),
    }
    assert sources["recovery-primary"]["status"] == (
        "read-only recovery evidence; not canonical or release authority"
    )


def test_destination_license_retains_recorded_apache_bytes() -> None:
    digest = hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest()

    assert digest == "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
