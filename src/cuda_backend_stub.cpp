// Stands in for the CUDA backend in a library built without CUDA: it is never usable, and asking
// for it says why.

#include "backend.h"

namespace neraulgraphics::detail {

std::unique_ptr<Backend> make_cuda_backend(const SceneInput &) {
    throw CudaError(-1, "this neraulgraphics was built without CUDA support, so only Device::Cpu is available");
}

bool cuda_usable() { return false; }

}  // namespace neraulgraphics::detail
