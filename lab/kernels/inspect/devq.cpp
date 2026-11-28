// Time a queue construction and the device queries the kernels make, three rounds per process.
// Build: acpp -O2 --acpp-targets=generic devq.cpp -o devq
#include <chrono>
#include <cstdio>
#include <sycl/sycl.hpp>
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
int main() {
    for (int round = 0; round < 3; ++round) {
        auto t0 = clk::now();
        {
            sycl::queue q;
            auto t1 = clk::now();
            const auto dev = q.get_device();
            auto t2 = clk::now();
            const auto wg = dev.get_info<sycl::info::device::max_work_group_size>();
            auto t3 = clk::now();
            const auto cu = dev.get_info<sycl::info::device::max_compute_units>();
            auto t4 = clk::now();
            const auto cu2 = dev.get_info<sycl::info::device::max_compute_units>();
            auto t5 = clk::now();
            const bool cpu = dev.is_cpu();
            auto t6 = clk::now();
            std::printf("round %d: queue %.1f ms | max_work_group_size %.2f ms (%zu) | max_compute_units %.2f ms (%u), "
                        "again %.2f ms (%u) | is_cpu %.3f ms (%d)\n",
                        round, ms(t0, t1), ms(t2, t3), wg, ms(t3, t4), cu, ms(t4, t5), cu2, ms(t5, t6), cpu);
        }
    }
}
