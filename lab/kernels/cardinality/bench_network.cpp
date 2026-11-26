// A k-of-n cardinality gate against the two-level AND/OR networks that compute
// the same bits, on the production kernels (src/mc/kernel/gate.h,
// cardinality.h) with the production form, lanes and work-group order of the
// device:
//
//   card  one cardinality launch over G voters;
//   sop   sum of products: an AND launch over the C(n, k) k-subsets of each
//   voter, then an OR launch of
//         C(n, k) inputs per voter;
//   pos   product of sums: an OR launch over the C(n, k - 1) (n - k +
//   1)-subsets of each voter, then an AND
//         launch of C(n, k - 1) inputs per voter;
//   one   for k = 1 or k = n: one OR or AND launch of n inputs.
//
// Each launch's kernel time is the median of its profiled launches after a
// warm-up launch (which JIT-compiles it); a network's time is the sum of its
// two launches. Every network output is compared word for word with the
// cardinality output. Voters draw n distinct inputs from a pool of P buffers
// whose bits are 1 with the probability (in 64ths) that makes a voter true
// about half the time.
//
// Usage: bench_network [--B bitpacks] [--pool P] [--reps R] [--budget-ms MS]
// [--max-net GATES] n:k[:G] ...

#include "mc/event/node.h"
#include "mc/event/sample_shape.h"
#include "mc/kernel/cardinality.h"
#include "mc/kernel/gate.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using bitpack_t = std::uint64_t;
using size_t_ = std::size_t;
namespace kn = scram::mc::kernel;
namespace ev = scram::mc::event;
using scram::core::Connective;
using inputs_t = std::vector<std::pair<std::vector<bitpack_t *>, size_t_>>;
using shape_t = ev::sample_shape<size_t_>;

int g_reps = 15;
double g_budget_ms = 3000;
// Experiments: unused local memory per work-group (bytes), which caps the work-groups resident per SM; and a
// forced work-group order (-1: production's, 0: gate-major, 1: block-major).
std::size_t g_pad = 0;
int g_order = -1;

kn::gate_order order_for(const sycl::device &dev, const bool mixed) {
    if (g_order == 0) {
        return kn::gate_order::gate_major;
    }
    if (g_order == 1) {
        return kn::gate_order::block_major;
    }
    return kn::gate_order_for(dev, mixed);
}

template <typename K> sycl::event launch(sycl::queue &q, const K &k) {
    if (g_pad == 0) {
        return q.parallel_for(k.get_range(), k);
    }
    const std::size_t words = g_pad / sizeof(std::uint32_t);
    return q.submit([&](sycl::handler &h) {
        sycl::local_accessor<std::uint32_t, 1> pad(sycl::range<1>(words), h);
        h.parallel_for(k.get_range(), [=](sycl::nd_item<1> it) {
            if (it.get_local_id(0) == 0xFFFFFFFFu) { // never: keeps the allocation
                pad[0] = 0;
            }
            k(it);
        });
    });
}

template <typename K> double time_kernel(sycl::queue &q, const K &k) {
    launch(q, k).wait(); // JIT and warm-up
    std::vector<double> ts;
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < g_reps; ++r) {
        auto e = launch(q, k);
        e.wait();
        const auto s = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto f = e.template get_profiling_info<sycl::info::event_profiling::command_end>();
        ts.push_back(static_cast<double>(f - s) * 1e-6);
        const double el = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (r >= 2 && el > g_budget_ms) {
            break;
        }
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

// ---- generic gates, as the kernel builder picks them
template <Connective op_, std::uint32_t slots_, bool lanes_>
double run_gate(sycl::queue &q, const ev::gate_block<bitpack_t, size_t_> &blk, const shape_t &shape) {
    const kn::op<op_, bitpack_t, size_t_, slots_, lanes_> k(blk, shape, order_for(q.get_device(), slots_ == 0));
    return time_kernel(q, k);
}
template <Connective op_, bool lanes_, std::size_t... I>
double run_gate_fixed(sycl::queue &q, const ev::gate_block<bitpack_t, size_t_> &blk, const shape_t &shape,
                      const std::uint32_t slots, std::index_sequence<I...>) {
    double t = -1;
    const auto try_slots = [&]<std::uint32_t s_>() {
        if (slots == s_) {
            t = run_gate<op_, s_, lanes_>(q, blk, shape);
        }
    };
    (try_slots.template operator()<kn::kGateSlots[I]>(), ...);
    return t;
}
template <Connective op_>
double run_gate_prod(sycl::queue &q, const ev::gate_block<bitpack_t, size_t_> &blk, const shape_t &shape,
                     const std::uint32_t slots) {
    const bool lanes = q.get_device().is_cpu();
    constexpr auto each = std::make_index_sequence<kn::kGateSlots.size()>{};
    if (slots == 0) {
        return lanes ? run_gate<op_, 0, true>(q, blk, shape) : run_gate<op_, 0, false>(q, blk, shape);
    }
    return lanes ? run_gate_fixed<op_, true>(q, blk, shape, slots, each)
                 : run_gate_fixed<op_, false>(q, blk, shape, slots, each);
}

// ---- cardinality gates, as the kernel builder picks them
template <kn::cardinality_form form_, std::uint32_t slots_, bool lanes_>
double run_card(sycl::queue &q, const ev::cardinality_gate_block<bitpack_t, size_t_> &blk, const shape_t &shape) {
    const kn::cardinality<bitpack_t, size_t_, form_, slots_, lanes_> k(
        blk, shape, order_for(q.get_device(), form_ != kn::cardinality_form::fixed));
    return time_kernel(q, k);
}
template <bool lanes_, std::size_t... I>
double run_card_fixed(sycl::queue &q, const ev::cardinality_gate_block<bitpack_t, size_t_> &blk, const shape_t &shape,
                      const std::uint32_t slots, std::index_sequence<I...>) {
    double t = -1;
    const auto try_slots = [&]<std::uint32_t s_>() {
        if (slots == s_) {
            t = run_card<kn::cardinality_form::fixed, s_, lanes_>(q, blk, shape);
        }
    };
    (try_slots.template operator()<kn::kCardinalitySlots[I]>(), ...);
    return t;
}
double run_card_prod(sycl::queue &q, const ev::cardinality_gate_block<bitpack_t, size_t_> &blk, const shape_t &shape,
                     const std::size_t n) {
    const bool lanes = q.get_device().is_cpu();
    const auto form = kn::cardinality_form_for(n);
    constexpr auto each = std::make_index_sequence<kn::kCardinalitySlots.size()>{};
    if (form == kn::cardinality_form::fixed) {
        const auto slots = kn::cardinality_slots_for(n);
        return lanes ? run_card_fixed<true>(q, blk, shape, slots, each)
                     : run_card_fixed<false>(q, blk, shape, slots, each);
    }
    if (form == kn::cardinality_form::counter) {
        return lanes ? run_card<kn::cardinality_form::counter, 0, true>(q, blk, shape)
                     : run_card<kn::cardinality_form::counter, 0, false>(q, blk, shape);
    }
    return lanes ? run_card<kn::cardinality_form::wide, 0, true>(q, blk, shape)
                 : run_card<kn::cardinality_form::wide, 0, false>(q, blk, shape);
}

double binom(unsigned n, unsigned k) {
    double c = 1;
    for (unsigned i = 1; i <= k; ++i) {
        c = c * (n - k + i) / i;
    }
    return c;
}

// All m-subsets of {0, ..., n - 1}, in lexicographic order.
std::vector<std::vector<unsigned>> subsets(unsigned n, unsigned m) {
    std::vector<std::vector<unsigned>> out;
    std::vector<unsigned> s(m);
    for (unsigned i = 0; i < m; ++i) {
        s[i] = i;
    }
    while (true) {
        out.push_back(s);
        int i = static_cast<int>(m) - 1;
        while (i >= 0 && s[i] == n - m + i) {
            --i;
        }
        if (i < 0) {
            break;
        }
        ++s[i];
        for (unsigned j = i + 1; j < m; ++j) {
            s[j] = s[j - 1] + 1;
        }
    }
    return out;
}

// P(at least k of n | p)
double tail(unsigned n, unsigned k, double p) {
    double t = 0;
    for (unsigned j = k; j <= n; ++j) {
        t += binom(n, j) * std::pow(p, j) * std::pow(1 - p, n - j);
    }
    return t;
}

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

} // namespace

int main(int argc, char **argv) {
    std::size_t B = 16384;
    std::size_t pool_n = 1024;
    std::size_t max_net = 4096;
    std::vector<std::string> specs;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--B") {
            B = std::stoul(argv[++i]);
        } else if (a == "--pool") {
            pool_n = std::stoul(argv[++i]);
        } else if (a == "--reps") {
            g_reps = std::stoi(argv[++i]);
        } else if (a == "--budget-ms") {
            g_budget_ms = std::stod(argv[++i]);
        } else if (a == "--max-net") {
            max_net = std::stoul(argv[++i]);
        } else if (a == "--pad") {
            g_pad = std::stoul(argv[++i]);
        } else if (a == "--order") {
            const std::string o = argv[++i];
            g_order = o == "gate" ? 0 : (o == "block" ? 1 : -1);
        } else {
            specs.push_back(a);
        }
    }
    sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
    const auto dev = q.get_device();
    std::printf("# device: %s | cpu=%d | B=%zu pool=%zu max_net=%zu order=%s pad=%zu forced_order=%d\n",
                dev.get_info<sycl::info::device::name>().c_str(), dev.is_cpu() ? 1 : 0, B, pool_n, max_net,
                kn::gate_order_for(dev) == kn::gate_order::block_major ? "block-major" : "gate-major", g_pad, g_order);
    std::printf("R\tshape\tn\tk\tform\tG\tnet_gates\tcard_ms\tnet1_ms\tnet2_"
                "ms\tnet_ms\tnet/card\tnaive\tok\n");
    std::mt19937_64 rng(12345);
    const shape_t shape{1, B};
    bitpack_t *pool = sycl::malloc_device<bitpack_t>(pool_n * B, q);

    for (const auto &spec : specs) {
        const auto c1 = spec.find(':');
        const auto c2 = spec.find(':', c1 + 1);
        const auto n = static_cast<unsigned>(std::stoul(spec.substr(0, c1)));
        const auto k = static_cast<unsigned>(std::stoul(spec.substr(c1 + 1, c2 - c1 - 1)));
        const std::size_t G0 =
            c2 == std::string::npos ? std::clamp<std::size_t>(2048 / n, 2, 8192) : std::stoul(spec.substr(c2 + 1));
        if (k < 1 || k > n || n > pool_n) {
            std::fprintf(stderr, "bad shape %s\n", spec.c_str());
            return 2;
        }
        // Density in 64ths that makes a voter true about half the time.
        unsigned p64 = 1;
        double best = 2;
        for (unsigned d = 1; d < 64; ++d) {
            const double e = std::abs(tail(n, k, d / 64.0) - 0.5);
            if (e < best) {
                best = e;
                p64 = d;
            }
        }
        const std::uint64_t seed = rng();
        q.parallel_for(sycl::range<1>(pool_n * B), [=](sycl::id<1> id) {
             const std::size_t idx = id[0];
             std::uint64_t x = 0;
             for (int j = 6; j >= 1; --j) {
                 const std::uint64_t r = splitmix64(seed ^ (std::uint64_t(idx) << 4) ^ std::uint64_t(j));
                 x = ((p64 >> (6 - j)) & 1u) ? (x | r) : (x & r);
             }
             pool[idx] = x;
         }).wait();

        // The networks: forms that exist for this shape, each with its own voter
        // count.
        struct form_t {
            const char *name;
            unsigned m;    // inputs of a first-level gate
            std::size_t c; // first-level gates per voter
            Connective first, second;
        };
        std::vector<form_t> forms;
        if (k == 1) {
            forms.push_back({"one", n, 1, Connective::kOr, Connective::kOr});
        } else if (k == n) {
            forms.push_back({"one", n, 1, Connective::kAnd, Connective::kAnd});
        } else {
            forms.push_back({"sop", k, static_cast<std::size_t>(binom(n, k)), Connective::kAnd, Connective::kOr});
            forms.push_back(
                {"pos", n - k + 1, static_cast<std::size_t>(binom(n, k - 1)), Connective::kOr, Connective::kAnd});
        }
        for (const auto &f : forms) {
            const bool one = std::string(f.name) == "one";
            const std::size_t G = one ? G0 : std::min<std::size_t>(G0, max_net / f.c);
            if (G < 1) {
                std::printf("R\t%s\t%u\t%u\t%s\t0\t%zu\t\t\t\t\t\t\tskipped (network "
                            "over %zu gates)\n",
                            spec.c_str(), n, k, f.name, f.c, max_net);
                continue;
            }
            // Voters: n distinct pool buffers each.
            std::vector<std::vector<bitpack_t *>> voters(G);
            std::vector<std::size_t> perm(pool_n);
            for (std::size_t i = 0; i < pool_n; ++i) {
                perm[i] = i;
            }
            for (std::size_t v = 0; v < G; ++v) {
                for (unsigned j = 0; j < n; ++j) {
                    std::swap(perm[j], perm[j + rng() % (pool_n - j)]);
                    voters[v].push_back(pool + perm[j] * B);
                }
            }
            // Cardinality gates [k, n].
            inputs_t card_in;
            std::vector<std::pair<size_t_, size_t_>> bounds;
            for (const auto &v : voters) {
                card_in.emplace_back(v, 0);
                bounds.emplace_back(k, n);
            }
            auto card = ev::create_cardinality_gate_block<bitpack_t, size_t_>(q, card_in, bounds, B,
                                                                              kn::cardinality_slots_for(n));
            const double t_card = run_card_prod(q, card, shape, n);

            // The network.
            double t1 = 0, t2 = 0;
            ev::gate_block<bitpack_t, size_t_> first{}, second{};
            const bitpack_t *net_out = nullptr;
            std::size_t net_gates = 0;
            if (one) {
                first = ev::create_gate_block<bitpack_t, size_t_>(q, card_in, B, kn::gate_slots_of(f.first));
                t1 = f.first == Connective::kOr
                         ? run_gate_prod<Connective::kOr>(q, first, shape, kn::gate_slots_of(f.first))
                         : run_gate_prod<Connective::kAnd>(q, first, shape, kn::gate_slots_of(f.first));
                net_out = first.buffers;
                net_gates = G;
            } else {
                const auto subs = subsets(n, f.m);
                inputs_t first_in;
                for (const auto &v : voters) {
                    for (const auto &s : subs) {
                        std::vector<bitpack_t *> ptrs;
                        for (const auto j : s) {
                            ptrs.push_back(v[j]);
                        }
                        first_in.emplace_back(std::move(ptrs), 0);
                    }
                }
                const auto s1 = kn::gate_slots_of(f.first);
                first = ev::create_gate_block<bitpack_t, size_t_>(q, first_in, B, s1);
                inputs_t second_in;
                for (std::size_t v = 0; v < G; ++v) {
                    std::vector<bitpack_t *> ptrs;
                    for (std::size_t j = 0; j < f.c; ++j) {
                        ptrs.push_back(first.buffers + (v * f.c + j) * B);
                    }
                    second_in.emplace_back(std::move(ptrs), 0);
                }
                const auto s2 = kn::gate_slots_of(f.second);
                second = ev::create_gate_block<bitpack_t, size_t_>(q, second_in, B, s2);
                if (f.first == Connective::kAnd) {
                    t1 = run_gate_prod<Connective::kAnd>(q, first, shape, s1);
                    t2 = run_gate_prod<Connective::kOr>(q, second, shape, s2);
                } else {
                    t1 = run_gate_prod<Connective::kOr>(q, first, shape, s1);
                    t2 = run_gate_prod<Connective::kAnd>(q, second, shape, s2);
                }
                net_out = second.buffers;
                net_gates = G * f.c + G;
            }
            // Check: the network's bits are the cardinality gate's.
            std::vector<bitpack_t> a(G * B), b(G * B);
            q.memcpy(a.data(), card.buffers, G * B * sizeof(bitpack_t)).wait();
            q.memcpy(b.data(), net_out, G * B * sizeof(bitpack_t)).wait();
            std::size_t bad = 0, ones = 0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                bad += a[i] != b[i];
                ones += static_cast<std::size_t>(std::popcount(a[i]));
            }
            // Naive (bytes) ratio: words moved per bitpack, network against
            // cardinality.
            const double naive =
                one ? 1.0 : (static_cast<double>(f.c) * (f.m + 1) + static_cast<double>(f.c) + 1) / (n + 1);
            const double t_net = t1 + t2;
            std::printf("R\t%s\t%u\t%u\t%s\t%zu\t%zu\t%.5f\t%.5f\t%.5f\t%.5f\t%.3f\t%"
                        ".2f\t%s ones=%.3f p=%u/64\n",
                        spec.c_str(), n, k, f.name, G, net_gates, t_card, t1, t2, t_net, t_net / t_card, naive,
                        bad == 0 ? "ok" : "WRONG", static_cast<double>(ones) / (64.0 * a.size()), p64);
            std::fflush(stdout);
            ev::destroy_cardinality_gate_block(q, card);
            ev::destroy_gate_block(q, first);
            if (!one) {
                ev::destroy_gate_block(q, second);
            }
        }
    }
    sycl::free(pool, q);
    return 0;
}
