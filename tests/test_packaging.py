"""Automated PyPI packaging verification and distribution wheel build test suite."""

import os
import sys
import tempfile
import zipfile
import tarfile
import subprocess
from pathlib import Path
import pytest

import microgen

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_package_version_consistency():
    assert hasattr(microgen, "__version__")
    assert microgen.__version__ == "1.0.1"


def test_pyproject_toml_exists_and_valid():
    pyproject_path = ROOT_DIR / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml does not exist in root directory"

    content = pyproject_path.read_text()
    assert 'name = "microgen-llm"' in content
    assert 'version = "1.0.1"' in content
    assert 'microgen = "microgen.cli.main:main"' in content


def test_build_sdist_and_wheel():
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = subprocess.run(
            [sys.executable, "-m", "build", "--outdir", tmp_dir],
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert res.returncode == 0, f"python -m build failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

        built_files = os.listdir(tmp_dir)
        wheels = [f for f in built_files if f.endswith(".whl")]
        sdists = [f for f in built_files if f.endswith(".tar.gz")]

        assert len(wheels) == 1, f"Expected 1 wheel artifact, found: {wheels}"
        assert len(sdists) == 1, f"Expected 1 sdist artifact, found: {sdists}"

        wheel_path = os.path.join(tmp_dir, wheels[0])
        sdist_path = os.path.join(tmp_dir, sdists[0])

        # Inspect wheel contents
        with zipfile.ZipFile(wheel_path, "r") as zip_file:
            file_list = zip_file.namelist()
            assert any("microgen/__init__.py" in f for f in file_list), "microgen/__init__.py missing from wheel"
            assert any("microgen/sdk/engine.py" in f for f in file_list), "microgen/sdk/engine.py missing from wheel"
            assert any("microgen/cli/main.py" in f for f in file_list), "microgen/cli/main.py missing from wheel"
            assert any("microgen/memory/__init__.py" in f for f in file_list), "microgen/memory/__init__.py missing from wheel"
            assert any("entry_points.txt" in f for f in file_list), "entry_points.txt missing from wheel dist-info"

        # Inspect sdist contents
        with tarfile.open(sdist_path, "r:gz") as tar_file:
            tar_names = tar_file.getnames()
            assert any("pyproject.toml" in f for f in tar_names), "pyproject.toml missing from sdist archive"
            assert any("microgen/__init__.py" in f for f in tar_names), "microgen/__init__.py missing from sdist archive"
