# Releasing

Short checklist for publishing a version to PyPI. Everything else in this
repository is generated or tested automatically; publishing is the one
manual step.

1. Make sure the tree is green and the artifacts match the sources:

   ```bash
   python -m pytest -q
   python tools/generate_reference_results.py --check
   python tools/generate_reference_results.py --verify-determinism
   ```

2. Bump the version in `pyproject.toml`, `CMakeLists.txt` and
   `python/vessel_gnc/__init__.py`, then regenerate the reference
   artifacts (the version is part of the recorded provenance) and commit.

3. Build both distributions. The wheel compiles the C++ core through
   scikit-build-core, so a C++20 toolchain and CMake >= 3.20 must be
   available:

   ```bash
   python -m build
   ```

4. Smoke-test the wheel in a clean environment:

   ```bash
   python -m venv /tmp/vgnc-release && /tmp/vgnc-release/bin/pip install dist/*.whl
   /tmp/vgnc-release/bin/python -c "import vessel_gnc; print(vessel_gnc.__version__)"
   ```

5. Upload (PyPI API token in `~/.pypirc` or `TWINE_USERNAME=__token__`):

   ```bash
   python -m twine upload dist/*
   ```

6. Tag the release (`git tag vX.Y.Z && git push --tags`) and check that
   the README badges resolve.

Builds are per-interpreter: publish wheels for each Python minor version
you support (3.12, 3.13) and keep the source distribution as the fallback
for other platforms. `cibuildwheel` is the tool to reach for once the
first manual release is proven.
