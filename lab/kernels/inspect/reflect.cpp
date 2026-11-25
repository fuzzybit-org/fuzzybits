// Which backend AdaptiveCpp's JIT reflection reports inside a kernel (src/mc/prng/state128.h picks its multiply
// form from it). Build: acpp -O2 --acpp-targets=generic reflect.cpp -o reflect
#include <cstdio>
#include <sycl/sycl.hpp>
int main() {
    sycl::queue q;
    int *out = sycl::malloc_shared<int>(4, q);
    out[0] = out[1] = out[2] = out[3] = -7;
    q.single_task([=]() {
         out[3] = 0;
#if defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__)
         out[3] = 1;
         __acpp_if_target_sscp(
             namespace jit = sycl::AdaptiveCpp_jit;
             out[0] = jit::reflection_query::compiler_backend::is_known() ? 1 : 0;
             out[1] = static_cast<int>(jit::reflect<jit::reflection_query::compiler_backend>());
             out[2] =
                 jit::reflect<jit::reflection_query::compiler_backend>() == jit::compiler_backend::amdgpu ? 1 : 0;);
#endif
     }).wait();
    std::printf("device=%s sscp_macro=%d known=%d backend=%d is_amdgpu=%d\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), out[3], out[0], out[1], out[2]);
    sycl::free(out, q);
}
