"""Small dependency-free PEP 517 backend for this pure-Python distribution."""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import os
import stat
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

_NAME = "projectkoios-optimization"
_NORMALIZED = "projectkoios_optimization"
_VERSION = "0.1.0.dev0"
_DIST_INFO = f"{_NORMALIZED}-{_VERSION}.dist-info"
_WHEEL = f"{_NORMALIZED}-{_VERSION}-py3-none-any.whl"
_SDIST = f"{_NORMALIZED}-{_VERSION}.tar.gz"
_ROOT = Path(__file__).resolve().parent.parent


def _epoch() -> int:
    return max(315_532_800, int(os.environ.get("SOURCE_DATE_EPOCH", "315532800")))


def _metadata() -> bytes:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    value = f"""Metadata-Version: 2.4
Name: {_NAME}
Version: {_VERSION}
Summary: Numerics and optimization for Project Koios
Author-email: Eugene Ragasa <eugene@projectkoios.com>
License-Expression: Apache-2.0
License-File: LICENSE
License-File: NOTICE
License-File: licenses/PyPosPack-BSD-2-Clause.txt
Requires-Python: >=3.14
Project-URL: Repository, https://github.com/eragasa/projectkoios-optimization
Project-URL: Documentation, https://github.com/eragasa/projectkoios-optimization/tree/main/docs
Provides-Extra: numpy
Requires-Dist: numpy<3,>=2.3; extra == 'numpy'
Provides-Extra: scipy
Requires-Dist: numpy<3,>=2.3; extra == 'scipy'
Requires-Dist: scipy<2,>=1.18; extra == 'scipy'
Provides-Extra: gmm
Requires-Dist: numpy<3,>=2.3; extra == 'gmm'
Requires-Dist: scipy<2,>=1.18; extra == 'gmm'
Requires-Dist: scikit-learn<2,>=1.8; extra == 'gmm'
Provides-Extra: benchmark
Requires-Dist: matplotlib<4,>=3.11; extra == 'benchmark'
Requires-Dist: numpy<3,>=2.3; extra == 'benchmark'
Requires-Dist: pymoo==0.6.2; extra == 'benchmark'
Requires-Dist: scikit-learn==1.8.0; extra == 'benchmark'
Requires-Dist: scipy<2,>=1.18; extra == 'benchmark'
Provides-Extra: development
Requires-Dist: build<2,>=1.3; extra == 'development'
Requires-Dist: mpmath<2,>=1.3; extra == 'development'
Requires-Dist: mypy<3,>=2.3; extra == 'development'
Requires-Dist: pytest<10,>=9.1; extra == 'development'
Requires-Dist: ruff<1,>=0.16; extra == 'development'
Description-Content-Type: text/markdown

{readme}"""
    return value.encode()


def _source_files() -> tuple[Path, ...]:
    explicit = (
        _ROOT / ".gitignore",
        _ROOT / "AGENTS.md",
        _ROOT / "LICENSE",
        _ROOT / "NOTICE",
        _ROOT / "README.md",
        _ROOT / "TRANSFER.toml",
        _ROOT / "pyproject.toml",
    )
    generated_parts = {
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
    }
    discovered: list[Path] = []
    for root_name in (
        "benchmarks",
        "build_backend",
        "docs",
        "licenses",
        "src",
        "tests",
    ):
        for path in (_ROOT / root_name).rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(_ROOT)
            if generated_parts.intersection(relative.parts):
                continue
            if any(part.endswith(".egg-info") for part in relative.parts):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            discovered.append(path)
    return tuple(sorted((*explicit, *discovered), key=lambda item: item.as_posix()))


def _wheel_files() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for package_name in ("numerics", "optimization"):
        package = _ROOT / f"src/python/projectkoios/{package_name}"
        for path in sorted(package.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_file() and (path.suffix == ".py" or path.name == "py.typed"):
                relative = path.relative_to(package).as_posix()
                files[f"projectkoios/{package_name}/{relative}"] = path.read_bytes()
    files[f"{_DIST_INFO}/licenses/LICENSE"] = (_ROOT / "LICENSE").read_bytes()
    files[f"{_DIST_INFO}/licenses/NOTICE"] = (_ROOT / "NOTICE").read_bytes()
    files[f"{_DIST_INFO}/licenses/PyPosPack-BSD-2-Clause.txt"] = (
        _ROOT / "licenses/PyPosPack-BSD-2-Clause.txt"
    ).read_bytes()
    files[f"{_DIST_INFO}/TRANSFER.toml"] = (_ROOT / "TRANSFER.toml").read_bytes()
    files[f"{_DIST_INFO}/METADATA"] = _metadata()
    files[f"{_DIST_INFO}/WHEEL"] = (
        b"Wheel-Version: 1.0\n"
        b"Generator: projectkoios-optimization-build-backend 1\n"
        b"Root-Is-Purelib: true\n"
        b"Tag: py3-none-any\n"
    )
    return files


def _record(files: dict[str, bytes]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name in sorted(files):
        payload = files[name]
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
        writer.writerow((name, f"sha256={digest.decode()}", len(payload)))
    writer.writerow((f"{_DIST_INFO}/RECORD", "", ""))
    return stream.getvalue().encode()


def _zip_info(name: str) -> zipfile.ZipInfo:
    timestamp = time.gmtime(_epoch())[:6]
    info = zipfile.ZipInfo(name, timestamp)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def get_requires_for_build_wheel(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return []


def get_requires_for_build_sdist(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return []


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    directory = Path(metadata_directory) / _DIST_INFO
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "METADATA").write_bytes(_metadata())
    (directory / "WHEEL").write_text(
        "Wheel-Version: 1.0\n"
        "Generator: projectkoios-optimization-build-backend 1\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n",
        encoding="utf-8",
    )
    return _DIST_INFO


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    output = Path(wheel_directory) / _WHEEL
    files = _wheel_files()
    files[f"{_DIST_INFO}/RECORD"] = _record(files)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            archive.writestr(_zip_info(name), files[name], compresslevel=9)
    return _WHEEL


def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    output = Path(sdist_directory) / _SDIST
    prefix = f"{_NORMALIZED}-{_VERSION}"
    epoch = _epoch()
    entries = {
        path.relative_to(_ROOT).as_posix(): path.read_bytes()
        for path in _source_files()
    }
    entries["PKG-INFO"] = _metadata()
    with (
        output.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=epoch) as compressed,
        tarfile.open(
            fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
        ) as archive,
    ):
        for name in sorted(entries):
            payload = entries[name]
            info = tarfile.TarInfo(f"{prefix}/{name}")
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = epoch
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    return _SDIST
