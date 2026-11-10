# Shortcuts for the everyday loop. CMake does the real work; these just save typing.
#
#   make raytracer                      build the viewer, then run it with ./raytracer
#   make run                            build and run it
#   make run ARGS="--instances 5000"    build and run it with options
#   make test                           build and run the C++ tests (needs a GPU)
#   make clean                          delete the build directory and ./raytracer
#
# BUILD=somewhere changes the build directory. CMake options still work the usual
# way, e.g. make raytracer CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=native".

BUILD ?= build
CMAKE_ARGS ?=
ARGS ?=

.DEFAULT_GOAL := raytracer
.PHONY: raytracer run test clean

# Configuring an already-configured build directory is a no-op that takes a moment,
# and building only recompiles what changed.
raytracer:
	cmake -S . -B $(BUILD) $(CMAKE_ARGS)
	cmake --build $(BUILD) --target neraulgraphics-viewer --parallel
	ln -sf $(BUILD)/neraulgraphics-viewer raytracer
	@echo "built ./raytracer"

run: raytracer
	./raytracer $(ARGS)

test:
	cmake -S . -B $(BUILD) $(CMAKE_ARGS)
	cmake --build $(BUILD) --target neraulgraphics-tests --parallel
	ctest --test-dir $(BUILD) --output-on-failure

clean:
	rm -rf $(BUILD) raytracer
