# Releasing

The version is written in exactly one place, `project(neraulgraphics VERSION x.y.z)`
in `CMakeLists.txt`. The C++ `version.h`, the `find_package` version check, the
Python package version and `neraulgraphics.__version__` are all generated from it.

## Steps

1. **Choose the number.** Before 1.0: bump the minor version (0.3.0) for anything
   that changes the API, the patch version (0.2.1) for fixes only.

2. **Bump it** in `CMakeLists.txt`:

   ```cmake
   project(neraulgraphics VERSION 0.3.0 LANGUAGES CXX CUDA)
   ```

3. **Add a section to `CHANGELOG.md`** headed `## [0.3.0]`. It becomes the
   release notes.

4. **Test on a machine with a GPU.** CI has no GPU, so this is the only place the
   rendering tests run:

   ```bash
   cmake -S . -B build && cmake --build build && (cd build && ctest)
   pip install ".[test]" && pytest
   ```

5. **Commit, tag and push:**

   ```bash
   git commit -am "release: 0.3.0"
   git tag v0.3.0
   git push origin main v0.3.0
   ```

The tag starts `.github/workflows/release.yml`. It checks that the tag,
`CMakeLists.txt` and `CHANGELOG.md` agree, builds the source archive and the
wheels, and publishes them as the GitHub Release `v0.3.0`.

## Dry run

*Actions → release → Run workflow* builds everything without publishing. The
files appear as downloadable artifacts on the run's page.

To build the wheels locally instead (needs Docker):

```bash
pipx run cibuildwheel --platform linux
```

## If the workflow fails

Nothing is published unless every job succeeds. Fix the problem, then move the
tag to the fixed commit and push it again:

```bash
git tag -f v0.3.0
git push -f origin v0.3.0
```

Only do this for a tag whose release was never published. Once people may have
downloaded a version, release a new one instead.
