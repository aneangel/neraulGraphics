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
wheels (Linux x86_64 with CUDA; CPU-only for macOS, Windows and Linux aarch64), and
publishes them as the GitHub Release `v0.3.0`. If PyPI is switched on (below), it
then uploads the same files there.

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

## Publishing to PyPI

PyPI uploads use *trusted publishing*: the workflow proves to PyPI who it is, so no API token is
stored anywhere. It is off until you set it up once, and then it is automatic:

1. On [pypi.org](https://pypi.org), under *Your account > Publishing*, add a **pending publisher**
   with project name `neraulgraphics`, owner `aneangel`, repository `neraulGraphics`, workflow
   `release.yml`, and environment `pypi`.
2. On GitHub, in the repository's *Settings > Environments*, create an environment named `pypi`.
   (Optionally require your approval there, so a tag cannot upload without you clicking.)
3. In *Settings > Secrets and variables > Actions > Variables*, add `PUBLISH_TO_PYPI` = `true`.

From the next tag on, the `pypi` job runs after the GitHub Release succeeds. Once the first version
is up, change the README's install line to `pip install neraulgraphics`.

Two things to know. A PyPI version can never be replaced or reused: a bad upload can only be
*yanked* (hidden from new installs), so look at the dry run's artifacts first. And the Linux
x86_64 wheels contain the CUDA backend, so they are the largest; all are well under PyPI's 100 MB limit.

