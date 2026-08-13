// Micro-benchmark of the C++ simulation kernel (plan §19).
// Build with:  cmake -B build -DVESSEL_GNC_BUILD_BENCHMARKS=ON
// Run:        ./build/benchmark_core

#include <chrono>
#include <cstdio>

#include "vessel_gnc/dynamics.hpp"
#include "vessel_gnc/integrator.hpp"

int main() {
    using namespace vessel_gnc;
    using clock = std::chrono::steady_clock;

    const ModelParams params = default_params();
    Environment env;
    env.current_east = 0.15;
    const Control control{25.0, 1.0};

    constexpr int kSteps = 1'000'000;
    State s;
    const auto t0 = clock::now();
    for (int i = 0; i < kSteps; ++i) {
        s = rk4_step(s, control, env, params, 0.01);
    }
    const auto t1 = clock::now();
    const double ns_per_step =
        std::chrono::duration<double, std::nano>(t1 - t0).count() / kSteps;

    std::printf("3-DOF RK4 propagation: %.1f ns/step (state: x = %.2f m, u = %.2f m/s)\n",
                ns_per_step, s.x, s.u);
    std::printf("1,000 s simulation (C++ kernel only): %.1f ms\n",
                ns_per_step * 100'000 / 1e6);
    return 0;
}
