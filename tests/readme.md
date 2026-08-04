# GPUMD tests

The CMake build registers the molecular-force unit and integration tests with CTest.

Configure, build, and run all currently registered tests from the repository root:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

The `check` target combines the last two steps for the test executables and `gpumd`:

```bash
cmake --build build --target check -j2
```

Useful subsets are selected by labels:

```bash
ctest --test-dir build -L unit --output-on-failure
ctest --test-dir build -L integration --output-on-failure
ctest --test-dir build -L gpu --output-on-failure
```

CUDA runtime tests report `Skipped` when no accessible GPU is present. The
`integration.molecular_force_run_in` test launches the CMake-built `gpumd` in an isolated
temporary directory.

The no-op regression needs a separate, unmodified GPUMD executable, so it is opt-in:

```bash
cmake -S . -B build \
  -DBUILD_TESTING=ON \
  -DGPUMD_BASELINE_EXECUTABLE=/absolute/path/to/baseline/gpumd
cmake --build build --target check -j2
```

The older tests under `tests/gpumd/` are not yet all registered with CTest. They remain
available through the legacy scripts while their inputs, generated outputs, and numerical
tolerances are standardized.
