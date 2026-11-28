// Generic gate microbenchmark: AND, OR, XOR, NAND, NOR (n inputs) and NOT, NULL
// (one input) over G gates x B bitpacks, inputs drawn from a pool of device
// buffers. Times a replica of the kernel that #275 replaced (`prod`:
// src/mc/kernel/gate.h's op<OpType> before it, with the local range of
// working_set::compute_optimal_local_range_3d_for_gpu) against
// one-word-per-work-item kernels with a runtime or compile-time input count, at
// 32 to 256 bits per work-item, and against the cardinality kernels of card.h
// posing as AND, OR, NAND and NOR. Every variant is checked against a host
// evaluation on sampled words.
//
// Shapes: OP:N or OP:N:G (G gates; default G = target / N).
#include "card.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using card::word;

namespace {

enum op_t : int { o_and = 0, o_or, o_xor, o_nand, o_nor, o_not, o_null };
const char *const kOpNames[] = {"and", "or", "xor", "nand", "nor", "not", "null"};

// ------------------------------------------------------------------ the replaced
// kernel (src/mc/kernel/gate.h before #275)
struct prod_gate { // event::gate<std::uint64_t, std::uint32_t>
    word *buffer;
    word **inputs;
    std::uint32_t num_inputs;
    std::uint32_t negated_inputs_offset;
};

template <int OP> struct prod_op {
    const prod_gate *gates;
    std::uint32_t count;
    std::uint32_t bitpacks;
    void operator()(const sycl::nd_item<3> &item) const {
        const auto gate_idx = static_cast<std::uint32_t>(item.get_global_id(0));
        const auto batch_idx = static_cast<std::uint32_t>(item.get_global_id(1));
        const auto bitpack_idx = static_cast<std::uint32_t>(item.get_global_id(2));
        if (gate_idx >= count || batch_idx >= 1 || bitpack_idx >= bitpacks) {
            return;
        }
        const std::uint32_t index = batch_idx * bitpacks + bitpack_idx;
        const auto &g = gates[gate_idx];
        word result = (OP == o_and || OP == o_nand) ? ~word(0) : word(0);
        for (std::uint32_t i = 0; i < g.negated_inputs_offset; ++i) {
            const word val = g.inputs[i][index];
            if constexpr (OP == o_or || OP == o_nor) {
                result |= val;
            } else if constexpr (OP == o_and || OP == o_nand) {
                result &= val;
            } else if constexpr (OP == o_xor) {
                result ^= val;
            } else {
                result = val;
            }
        }
        for (std::uint32_t i = g.negated_inputs_offset; i < g.num_inputs; ++i) {
            const word val = ~(g.inputs[i][index]);
            if constexpr (OP == o_or || OP == o_nor) {
                result |= val;
            } else if constexpr (OP == o_and || OP == o_nand) {
                result &= val;
            } else if constexpr (OP == o_xor) {
                result ^= val;
            } else {
                result = val;
            }
        }
        if constexpr (OP == o_nand || OP == o_nor || OP == o_not) {
            result = ~result;
        }
        g.buffer[index] = result;
    }
};

std::size_t closest_power_of_2(std::size_t n) {
    if (n == 0) {
        return 1;
    }
    std::size_t best = 1, diff = ~std::size_t{0};
    for (std::size_t x = 0; x < 63; ++x) {
        const std::size_t p2 = std::size_t{1} << x;
        const std::size_t d = p2 > n ? p2 - n : n - p2;
        if (d < diff) {
            diff = d;
            best = p2;
        }
    }
    return best;
}

sycl::range<3> prod_local_range(std::size_t gates, std::size_t bitpacks, std::size_t max_wg, const sycl::range<3> &max_items) {
    auto budget = static_cast<int>(std::log2(max_wg));
    const int lx = std::min(static_cast<int>(std::log2(closest_power_of_2(std::min(gates, max_items[0])))), budget);
    budget -= lx;
    const int lz = std::min(static_cast<int>(std::log2(closest_power_of_2(std::min(bitpacks, max_items[2])))), budget);
    return {std::size_t{1} << lx, 1, std::size_t{1} << lz};
}

std::size_t round_up(std::size_t a, std::size_t b) { return (a + b - 1) / b * b; }

// ------------------------------------------------------------------------------------------
// new kernels A lane is V: 32 to 512 bits of a gate's buffer, read through byte
// offsets so the width per work-item is independent of the 64-bit storage word.
// Typed, naturally aligned access (device buffers are cache-line aligned). A memcpy from a byte pointer
// leaves the compiler without the alignment, which cost 1.4-3.5x on sm_70.
template <typename V> [[gnu::always_inline]] inline V ldv(const word *base, std::size_t i) {
    return static_cast<const V *>(__builtin_assume_aligned(static_cast<const void *>(base), sizeof(V)))[i];
}
template <typename V> [[gnu::always_inline]] inline void stv(word *base, std::size_t i, const V &v) {
    static_cast<V *>(__builtin_assume_aligned(static_cast<void *>(base), sizeof(V)))[i] = v;
}

typedef word u64x2 __attribute__((ext_vector_type(2)));
typedef word u64x4 __attribute__((ext_vector_type(4)));
typedef word u64x16 __attribute__((ext_vector_type(16)));

struct gdesc {
    const word *const *in;
    word *out;
    std::uint32_t n;
    std::uint32_t neg;
    std::uint64_t mask; // fixed-count kernels: bit j set iff slot j holds an input
};

template <int OP, typename V> [[gnu::always_inline]] inline V ident() { return (OP == o_and || OP == o_nand) ? static_cast<V>(~V{}) : V{}; }
template <int OP, typename V> [[gnu::always_inline]] inline void comb(V &a, const V &x) {
    if constexpr (OP == o_and || OP == o_nand) {
        a &= x;
    } else if constexpr (OP == o_or || OP == o_nor) {
        a |= x;
    } else if constexpr (OP == o_xor) {
        a ^= x;
    } else {
        a = x;
    }
}
template <int OP, typename V> [[gnu::always_inline]] inline void fin(V &a) {
    if constexpr (OP == o_nand || OP == o_nor || OP == o_not) {
        a = ~a;
    }
}

// Work-group -> (gate, block of words). Gate-major runs all of a gate's blocks in a row; block-major runs
// every gate at a block before the next block, so gates that share inputs read them while they are cached.
struct decode {
    std::uint32_t gate;
    std::uint32_t idx;
};
[[gnu::always_inline]] inline decode decode_group(const sycl::nd_item<1> &it, std::uint32_t groups, std::uint32_t ngates, bool block_major) {
    const auto group = static_cast<std::uint32_t>(it.get_group(0));
    const std::uint32_t gate = block_major ? group % ngates : group / groups;
    const std::uint32_t block = block_major ? group / ngates : group - gate * groups;
    return {gate, block * static_cast<std::uint32_t>(it.get_local_range(0)) + static_cast<std::uint32_t>(it.get_local_id(0))};
}

template <typename V> [[gnu::always_inline]] inline V ldv_a(const word *base, std::size_t i) {
    return *static_cast<const V *>(__builtin_assume_aligned(reinterpret_cast<const char *>(base) + i * sizeof(V), sizeof(V)));
}

template <int OP, typename V, bool ALIGNED = false> struct k_loop {
    const gdesc *gates;
    std::uint32_t items;
    std::uint32_t groups;
    std::uint32_t ngates;
    bool block_major;
    void operator()(const sycl::nd_item<1> &it) const {
        const decode d = decode_group(it, groups, ngates, block_major);
        if (d.idx >= items) {
            return;
        }
        const gdesc g = gates[d.gate];
        V a = ident<OP, V>();
        std::uint32_t i = 0;
#pragma unroll 4
        for (; i < g.neg; ++i) {
            comb<OP>(a, ALIGNED ? ldv_a<V>(g.in[i], d.idx) : ldv<V>(g.in[i], d.idx));
        }
#pragma unroll 4
        for (; i < g.n; ++i) {
            const V x = ALIGNED ? ldv_a<V>(g.in[i], d.idx) : ldv<V>(g.in[i], d.idx);
            comb<OP>(a, static_cast<V>(~x));
        }
        fin<OP>(a);
        stv<V>(g.out, d.idx, a);
    }
};

// As src/mc/kernel/gate.h's counted form: the input count read at run time, BLK loads in flight per block
// (BLK = 1: one input at a time, each with its negation test).
template <int OP, typename V, int BLK = 8> struct k_blk8 {
    const gdesc *gates;
    std::uint32_t items;
    std::uint32_t groups;
    std::uint32_t ngates;
    bool block_major;
    void operator()(const sycl::nd_item<1> &it) const {
        const decode d = decode_group(it, groups, ngates, block_major);
        if (d.idx >= items) {
            return;
        }
        const gdesc g = gates[d.gate];
        V a = ident<OP, V>();
        std::uint32_t i = 0;
        for (; i + BLK <= g.n; i += BLK) {
            V x[BLK];
#pragma unroll
            for (std::uint32_t j = 0; j < BLK; ++j) {
                x[j] = ldv<V>(g.in[i + j], d.idx);
                if (i + j >= g.neg) {
                    x[j] = ~x[j];
                }
            }
#pragma unroll
            for (std::uint32_t j = 0; j < BLK; ++j) {
                comb<OP>(a, x[j]);
            }
        }
        for (; i < g.n; ++i) {
            V x = ldv<V>(g.in[i], d.idx);
            if (i >= g.neg) {
                x = ~x;
            }
            comb<OP>(a, x);
        }
        fin<OP>(a);
        stv<V>(g.out, d.idx, a);
    }
};

template <int OP, int N, typename V> struct k_fixed {
    const gdesc *gates;
    std::uint32_t items;
    std::uint32_t groups;
    std::uint32_t ngates;
    bool block_major;
    void operator()(const sycl::nd_item<1> &it) const {
        const decode d = decode_group(it, groups, ngates, block_major);
        if (d.idx >= items) {
            return;
        }
        const gdesc g = gates[d.gate];
        V a = ident<OP, V>();
#pragma unroll
        for (int j = 0; j < N; ++j) {
            if (((g.mask >> j) & 1u) != 0) {
                V x = ldv<V>(g.in[j], d.idx);
                if (static_cast<std::uint32_t>(j) >= g.neg) {
                    x = ~x;
                }
                comb<OP>(a, x);
            }
        }
        fin<OP>(a);
        stv<V>(g.out, d.idx, a);
    }
};

// As k_fixed, but every slot is loaded before any is combined (as the cardinality kernels' fixed form does).
template <int OP, int N, typename V> struct k_gather {
    const gdesc *gates;
    std::uint32_t items;
    std::uint32_t groups;
    std::uint32_t ngates;
    bool block_major;
    void operator()(const sycl::nd_item<1> &it) const {
        const decode d = decode_group(it, groups, ngates, block_major);
        if (d.idx >= items) {
            return;
        }
        const gdesc g = gates[d.gate];
        V x[N];
#pragma unroll
        for (int j = 0; j < N; ++j) {
            x[j] = ident<OP, V>();
            if (((g.mask >> j) & 1u) != 0) {
                x[j] = ldv<V>(g.in[j], d.idx);
                if (static_cast<std::uint32_t>(j) >= g.neg) {
                    x[j] = ~x[j];
                }
            }
        }
        V a = x[0];
#pragma unroll
        for (int j = 1; j < N; ++j) {
            comb<OP>(a, x[j]);
        }
        fin<OP>(a);
        stv<V>(g.out, d.idx, a);
    }
};

// The cardinality kernels of card.h on [lo, hi] (AND: [n, n], OR: [1, n], NAND:
// [0, n - 1], NOR: [0, 0]).
template <typename V, typename A> struct k_card {
    const card::gate_desc *gates;
    std::uint32_t words;
    std::uint32_t groups;
    std::uint32_t ngates;
    bool block_major;
    void operator()(const sycl::nd_item<1> &it) const {
        const decode d = decode_group(it, groups, ngates, block_major);
        const std::size_t w = std::size_t{d.idx} * card::lanes_v<V>;
        if (w >= words) {
            return;
        }
        const card::gate_desc g = gates[d.gate];
        card::st<V>(g.out + w, A::template eval<V>(g, w));
    }
};
template <int N> struct a_fixedM {
    template <typename V> static V eval(const card::gate_desc &g, std::size_t w) { return card::eval_fixedM<N, V>(g, w); }
};
struct a_bin8m {
    template <typename V> static V eval(const card::gate_desc &g, std::size_t w) { return card::eval_bin8_mid<V>(g, w); }
};

template <typename K> sycl::event launch1(sycl::queue &q, std::size_t gates, std::size_t groups, std::size_t wg, const K &k) {
    return q.parallel_for(sycl::nd_range<1>({gates * groups * wg}, {wg}), k);
}

// ---------------------------------------------------------------------------------------------
// harness
word splitmix64(word x) {
    word z = x + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct timing {
    double ms;
    int reps;
};

timing time_it(sycl::queue &q, const std::function<std::vector<sycl::event>()> &launch, int reps, double budget_ms) {
    std::vector<double> ts;
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        auto evs = launch();
        q.wait();
        double ms = 0;
        for (auto &e : evs) {
            const auto s = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            const auto f = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            ms += static_cast<double>(f - s) * 1e-6;
        }
        ts.push_back(ms);
        const double el = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (r >= 2 && el > budget_ms) {
            break;
        }
    }
    std::sort(ts.begin(), ts.end());
    return {ts[ts.size() / 2], static_cast<int>(ts.size())};
}

bool has(const std::string &list, const std::string &v) {
    std::size_t a = 0;
    while (a <= list.size()) {
        const auto b = list.find(',', a);
        if (list.substr(a, b == std::string::npos ? std::string::npos : b - a) == v) {
            return true;
        }
        if (b == std::string::npos) {
            break;
        }
        a = b + 1;
    }
    return false;
}

const char *kDefaultShapes[] = {
    "and:2", "and:3", "and:4", "and:8", "and:16", "and:32", "and:64", "or:2",  "or:3",   "or:4",       "or:8",      "or:16",
    "or:32", "or:64", "xor:2", "xor:4", "xor:16", "nand:3", "nor:3",  "not:1", "null:1", "and:2:4096", "or:3:4096", "not:1:4096",
};

std::uint32_t fixed_class(std::uint32_t n) {
    static const std::uint32_t list[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    for (const auto c : list) {
        if (c >= n) {
            return c;
        }
    }
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    std::size_t B = 16384, pool_n = 1024, target = 2048, wg = 256;
    int reps = 15;
    double budget_ms = 3000;
    std::string variants;
    std::vector<std::string> shapes;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--B") {
            B = std::stoul(next());
        } else if (a == "--pool") {
            pool_n = std::stoul(next());
        } else if (a == "--target") {
            target = std::stoul(next());
        } else if (a == "--wg") {
            wg = std::stoul(next());
        } else if (a == "--reps") {
            reps = std::stoi(next());
        } else if (a == "--budget-ms") {
            budget_ms = std::stod(next());
        } else if (a == "--variants") {
            variants = next();
        } else if (a == "--shape") {
            shapes.push_back(next());
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (shapes.empty()) {
        for (const char *s : kDefaultShapes) {
            shapes.emplace_back(s);
        }
    }
    sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
    const auto dev = q.get_device();
    const bool cpu = dev.is_cpu();
    if (variants.empty()) {
        variants = cpu ? "prod,loop,loop.v4,loop.v8,loop.v16,fixed.v8,card.v8"
                       : "prod,loop,loop.u32,loop.x2,loop.x4,fixed,fixed.u32,fixed."
                         "x2,card";
    }
    const std::size_t max_wg = dev.get_info<sycl::info::device::max_work_group_size>();
    const auto max_items = dev.get_info<sycl::info::device::max_work_item_sizes<3>>();
    std::printf("# device: %s | cpu=%d | B=%zu pool=%zu target=%zu wg=%zu\n", dev.get_info<sycl::info::device::name>().c_str(), cpu ? 1 : 0, B, pool_n, target,
                wg);
    std::printf("R\tshape\tn\tlo\thi\tG\tB\tvariant\tms\tGBps\tok\treps\n");

    word *pool = sycl::malloc_device<word>(pool_n * B, q);
    std::vector<word> pool_h(pool_n * B);
    std::mt19937_64 rng(1);

    for (const auto &spec : shapes) {
        const auto c1 = spec.find(':');
        const auto c2 = spec.find(':', c1 + 1);
        const std::string opname = spec.substr(0, c1);
        const auto n = static_cast<std::uint32_t>(std::stoul(spec.substr(c1 + 1, c2 - c1 - 1)));
        int op = -1;
        for (int k = 0; k < 7; ++k) {
            if (opname == kOpNames[k]) {
                op = k;
            }
        }
        if (op < 0 || n == 0 || ((op == o_not || op == o_null) && n != 1)) {
            std::fprintf(stderr, "bad shape %s\n", spec.c_str());
            return 2;
        }
        const std::size_t G = c2 == std::string::npos ? std::clamp<std::size_t>(target / n, 4, 8192) : std::stoul(spec.substr(c2 + 1));
        const std::uint32_t npos = n - n / 2;
        const std::uint32_t nneg = n / 2;

        // Densities that leave both outputs likely: AND-like p = 2^(-1/n), OR-like
        // 1 - 2^(-1/n), XOR 1/2.
        double p = 0.5;
        if (op == o_and || op == o_nand) {
            p = std::pow(0.5, 1.0 / n);
        } else if (op == o_or || op == o_nor) {
            p = 1.0 - std::pow(0.5, 1.0 / n);
        }
        const auto pA = static_cast<std::uint32_t>(std::clamp(std::lround(64.0 * p), 1L, 63L));
        const std::size_t half = pool_n / 2;
        const word seed = rng();
        q.parallel_for(sycl::range<1>(pool_n * B), [=](sycl::id<1> id) {
             const std::size_t idx = id[0];
             const std::size_t b = idx / B;
             const std::size_t w = idx % B;
             const std::uint32_t p6 = b < half ? pA : 64 - pA;
             word x = 0;
             for (int j = 6; j >= 1; --j) {
                 const word r = splitmix64(seed ^ (word(b) << 40) ^ (word(w) << 4) ^ word(j));
                 x = ((p6 >> (6 - j)) & 1u) ? (x | r) : (x & r);
             }
             pool[idx] = x;
         }).wait();
        q.memcpy(pool_h.data(), pool, pool_n * B * sizeof(word)).wait();

        std::vector<std::uint32_t> idx(G * n);
        for (std::size_t g = 0; g < G; ++g) {
            for (std::uint32_t j = 0; j < n; ++j) {
                idx[g * n + j] = j < npos ? static_cast<std::uint32_t>(rng() % half) : static_cast<std::uint32_t>(half + rng() % (pool_n - half));
            }
        }
        word *out = sycl::malloc_device<word>(G * B, q);
        std::vector<word> out_h(G * B);

        std::vector<std::size_t> words;
        for (std::size_t w = 0; w < 32 && w < B; ++w) {
            words.push_back(w);
            words.push_back(B - 1 - w);
        }
        for (int r = 0; r < 64; ++r) {
            words.push_back(rng() % B);
        }
        std::vector<word> expect(G * words.size());
        std::size_t ones = 0;
        for (std::size_t g = 0; g < G; ++g) {
            for (std::size_t k = 0; k < words.size(); ++k) {
                const std::size_t w = words[k];
                word a = (op == o_and || op == o_nand) ? ~word(0) : word(0);
                for (std::uint32_t i = 0; i < n; ++i) {
                    word x = pool_h[static_cast<std::size_t>(idx[g * n + i]) * B + w];
                    if (i >= npos) {
                        x = ~x;
                    }
                    if (op == o_and || op == o_nand) {
                        a &= x;
                    } else if (op == o_or || op == o_nor) {
                        a |= x;
                    } else if (op == o_xor) {
                        a ^= x;
                    } else {
                        a = x;
                    }
                }
                if (op == o_nand || op == o_nor || op == o_not) {
                    a = ~a;
                }
                expect[g * words.size() + k] = a;
                ones += static_cast<std::size_t>(std::popcount(a));
            }
        }
        std::printf("# %s: op=%s n=%u G=%zu density=%u/64 expected-ones=%.3f\n", spec.c_str(), kOpNames[op], n, G, pA,
                    static_cast<double>(ones) / (64.0 * expect.size()));

        const double bytes = static_cast<double>(G) * B * 8.0 * (n + 1);
        auto report = [&](const char *variant, const std::function<std::vector<sycl::event>()> &launch) {
            q.memset(out, 0xA5, G * B * sizeof(word)).wait();
            launch();
            q.wait();
            q.memcpy(out_h.data(), out, G * B * sizeof(word)).wait();
            std::size_t bad = 0;
            for (std::size_t g = 0; g < G; ++g) {
                for (std::size_t k = 0; k < words.size(); ++k) {
                    bad += out_h[g * B + words[k]] != expect[g * words.size() + k];
                }
            }
            const timing t = time_it(q, launch, reps, budget_ms);
            std::printf("R\t%s\t%u\t0\t0\t%zu\t%zu\t%s\t%.5f\t%.1f\t%s\t%d\n", spec.c_str(), n, G, B, variant, t.ms, bytes / (t.ms * 1e6),
                        bad == 0 ? "ok" : "WRONG", t.reps);
            std::fflush(stdout);
        };

        // ---- the replaced kernel
        std::vector<void *> owned;
        if (has(variants, "prod")) {
            prod_gate *dg = sycl::malloc_shared<prod_gate>(G, q);
            word **dp = sycl::malloc_shared<word *>(G * n, q);
            for (std::size_t g = 0; g < G; ++g) {
                for (std::uint32_t i = 0; i < n; ++i) {
                    dp[g * n + i] = pool + static_cast<std::size_t>(idx[g * n + i]) * B;
                }
                dg[g] = {out + g * B, dp + g * n, n, npos};
            }
            const auto lr = prod_local_range(G, B, max_wg, max_items);
            const sycl::nd_range<3> range({round_up(G, lr[0]), 1, round_up(B, lr[2])}, lr);
            const auto count = static_cast<std::uint32_t>(G);
            const auto bp = static_cast<std::uint32_t>(B);
            auto go = [&]<int OP>() { report("prod", [&]() { return std::vector<sycl::event>{q.parallel_for(range, prod_op<OP>{dg, count, bp})}; }); };
            switch (op) {
            case o_and:
                go.template operator()<o_and>();
                break;
            case o_or:
                go.template operator()<o_or>();
                break;
            case o_xor:
                go.template operator()<o_xor>();
                break;
            case o_nand:
                go.template operator()<o_nand>();
                break;
            case o_nor:
                go.template operator()<o_nor>();
                break;
            case o_not:
                go.template operator()<o_not>();
                break;
            default:
                go.template operator()<o_null>();
                break;
            }
            sycl::free(dg, q);
            sycl::free(dp, q);
        }

        // ---- descriptors for the new kernels, padded to `slots` pointer slots (0:
        // n)
        auto upload = [&](std::uint32_t slots) {
            const std::uint32_t width = slots ? slots : n;
            std::vector<const word *> p(G * width, nullptr);
            std::vector<gdesc> d(G);
            const word **dp = sycl::malloc_device<const word *>(G * width, q);
            for (std::size_t g = 0; g < G; ++g) {
                for (std::uint32_t i = 0; i < n; ++i) {
                    p[g * width + i] = pool + static_cast<std::size_t>(idx[g * n + i]) * B;
                }
                d[g] = {dp + g * width, out + g * B, n, npos, n >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << n) - 1};
            }
            q.memcpy(dp, p.data(), p.size() * sizeof(const word *)).wait();
            gdesc *dd = sycl::malloc_device<gdesc>(G, q);
            q.memcpy(dd, d.data(), G * sizeof(gdesc)).wait();
            owned.push_back(dp);
            owned.push_back(dd);
            return static_cast<const gdesc *>(dd);
        };
        const gdesc *d_plain = upload(0);
        const std::uint32_t cls = fixed_class(n);
        const gdesc *d_fixed = cls ? upload(cls) : nullptr;

        const auto ng = static_cast<std::uint32_t>(G);
        auto run_loop = [&]<typename V, bool BM = false, bool AL = false, int B8 = 0>(const char *name) {
            const auto items = static_cast<std::uint32_t>(B * sizeof(word) / sizeof(V));
            const std::size_t l = std::min<std::size_t>(wg, items);
            const auto groups = static_cast<std::uint32_t>((items + l - 1) / l);
            auto go = [&]<int OP>() {
                report(name, [&]() {
                    if constexpr (B8 > 0) {
                        return std::vector<sycl::event>{launch1(q, G, groups, l, k_blk8<OP, V, B8>{d_plain, items, groups, ng, BM})};
                    } else {
                        return std::vector<sycl::event>{launch1(q, G, groups, l, k_loop<OP, V, AL>{d_plain, items, groups, ng, BM})};
                    }
                });
            };
            switch (op) {
            case o_and:
                go.template operator()<o_and>();
                break;
            case o_or:
                go.template operator()<o_or>();
                break;
            case o_xor:
                go.template operator()<o_xor>();
                break;
            case o_nand:
                go.template operator()<o_nand>();
                break;
            case o_nor:
                go.template operator()<o_nor>();
                break;
            case o_not:
                go.template operator()<o_not>();
                break;
            default:
                go.template operator()<o_null>();
                break;
            }
        };
        auto run_fixed = [&]<typename V, bool BM = false, bool GATHER = false>(const char *name) {
            if (!d_fixed) {
                return;
            }
            const auto items = static_cast<std::uint32_t>(B * sizeof(word) / sizeof(V));
            const std::size_t l = std::min<std::size_t>(wg, items);
            const auto groups = static_cast<std::uint32_t>((items + l - 1) / l);
            auto go = [&]<int OP, int N>() {
                report(name, [&]() {
                    if constexpr (GATHER) {
                        return std::vector<sycl::event>{launch1(q, G, groups, l, k_gather<OP, N, V>{d_fixed, items, groups, ng, BM})};
                    } else {
                        return std::vector<sycl::event>{launch1(q, G, groups, l, k_fixed<OP, N, V>{d_fixed, items, groups, ng, BM})};
                    }
                });
            };
            auto by_n = [&]<int OP>() {
                switch (cls) {
                case 1:
                    go.template operator()<OP, 1>();
                    break;
                case 2:
                    go.template operator()<OP, 2>();
                    break;
                case 3:
                    go.template operator()<OP, 3>();
                    break;
                case 4:
                    go.template operator()<OP, 4>();
                    break;
                case 6:
                    go.template operator()<OP, 6>();
                    break;
                case 8:
                    go.template operator()<OP, 8>();
                    break;
                case 12:
                    go.template operator()<OP, 12>();
                    break;
                case 16:
                    go.template operator()<OP, 16>();
                    break;
                case 24:
                    go.template operator()<OP, 24>();
                    break;
                case 32:
                    go.template operator()<OP, 32>();
                    break;
                case 48:
                    go.template operator()<OP, 48>();
                    break;
                default:
                    go.template operator()<OP, 64>();
                    break;
                }
            };
            switch (op) {
            case o_and:
                by_n.template operator()<o_and>();
                break;
            case o_or:
                by_n.template operator()<o_or>();
                break;
            case o_xor:
                by_n.template operator()<o_xor>();
                break;
            case o_nand:
                by_n.template operator()<o_nand>();
                break;
            case o_nor:
                by_n.template operator()<o_nor>();
                break;
            case o_not:
                by_n.template operator()<o_not>();
                break;
            default:
                by_n.template operator()<o_null>();
                break;
            }
        };
        // Cardinality kernels on [lo, hi]: fixed slots up to 16 inputs (card.h padp
        // classes), counter above.
        auto run_card = [&]<typename V, bool BM = false>(const char *name) {
            std::uint32_t lo = 0, hi = 0;
            if (op == o_and) {
                lo = n, hi = n;
            } else if (op == o_or) {
                lo = 1, hi = n;
            } else if (op == o_nand) {
                lo = 0, hi = n - 1;
            } else if (op == o_nor) {
                lo = 0, hi = 0;
            } else {
                return; // XOR is parity, NOT and NULL are one input: not a range of
                        // interest
            }
            static const std::uint32_t prod_cls[] = {1, 2, 3, 4, 6, 8, 12, 16};
            std::uint32_t slots = 0;
            for (const auto c : prod_cls) {
                if (c >= n) {
                    slots = c;
                    break;
                }
            }
            const std::uint32_t width = slots ? slots : n;
            std::vector<const word *> p(G * width, nullptr);
            std::vector<card::gate_desc> d(G);
            const word **dp = sycl::malloc_device<const word *>(G * width, q);
            for (std::size_t g = 0; g < G; ++g) {
                for (std::uint32_t i = 0; i < n; ++i) {
                    p[g * width + i] = pool + static_cast<std::size_t>(idx[g * n + i]) * B;
                }
                card::gate_desc &x = d[g];
                x.in = dp + g * width;
                x.out = out + g * B;
                x.n = n;
                x.neg = npos;
                x.lo = lo;
                x.hi = hi;
                x.cls = card::k_bin;
                x.arg = static_cast<std::uint32_t>(std::bit_width(n));
                x.mask = n >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << n) - 1;
            }
            q.memcpy(dp, p.data(), p.size() * sizeof(const word *)).wait();
            card::gate_desc *dd = sycl::malloc_device<card::gate_desc>(G, q);
            q.memcpy(dd, d.data(), G * sizeof(card::gate_desc)).wait();
            owned.push_back(dp);
            owned.push_back(dd);
            const auto words_per_gate = static_cast<std::uint32_t>(B);
            const std::size_t items = B / card::lanes_v<V>;
            const std::size_t l = std::min<std::size_t>(wg, items);
            const auto groups = static_cast<std::uint32_t>((items + l - 1) / l);
            auto go = [&]<typename A>() {
                report(name, [&]() { return std::vector<sycl::event>{launch1(q, G, groups, l, k_card<V, A>{dd, words_per_gate, groups, ng, BM})}; });
            };
            switch (slots) {
            case 1:
                go.template operator()<a_fixedM<1>>();
                break;
            case 2:
                go.template operator()<a_fixedM<2>>();
                break;
            case 3:
                go.template operator()<a_fixedM<3>>();
                break;
            case 4:
                go.template operator()<a_fixedM<4>>();
                break;
            case 6:
                go.template operator()<a_fixedM<6>>();
                break;
            case 8:
                go.template operator()<a_fixedM<8>>();
                break;
            case 12:
                go.template operator()<a_fixedM<12>>();
                break;
            case 16:
                go.template operator()<a_fixedM<16>>();
                break;
            default:
                go.template operator()<a_bin8m>();
                break;
            }
        };

        if (has(variants, "loop"))
            run_loop.template operator()<word>("loop");
        if (has(variants, "loop.u32"))
            run_loop.template operator()<std::uint32_t>("loop.u32");
        if (has(variants, "loop.x2"))
            run_loop.template operator()<u64x2>("loop.x2");
        if (has(variants, "loop.x4"))
            run_loop.template operator()<u64x4>("loop.x4");
        if (has(variants, "loop.v4"))
            run_loop.template operator()<card::v4>("loop.v4");
        if (has(variants, "loop.v8"))
            run_loop.template operator()<card::v8>("loop.v8");
        if (has(variants, "loop.v16"))
            run_loop.template operator()<u64x16>("loop.v16");
        if (has(variants, "fixed"))
            run_fixed.template operator()<word>("fixed");
        if (has(variants, "fixed.u32"))
            run_fixed.template operator()<std::uint32_t>("fixed.u32");
        if (has(variants, "fixed.x2"))
            run_fixed.template operator()<u64x2>("fixed.x2");
        if (has(variants, "fixed.v8"))
            run_fixed.template operator()<card::v8>("fixed.v8");
        if (has(variants, "card"))
            run_card.template operator()<word>("card");
        if (has(variants, "loop.bm"))
            run_loop.template operator()<word, true>("loop.bm");
        if (has(variants, "loop.x2a"))
            run_loop.template operator()<u64x2, false, true>("loop.x2a");
        if (has(variants, "loop.x2a.bm"))
            run_loop.template operator()<u64x2, true, true>("loop.x2a.bm");
        if (has(variants, "loop.v8.bm"))
            run_loop.template operator()<card::v8, true>("loop.v8.bm");
        if (has(variants, "fixed.bm"))
            run_fixed.template operator()<word, true>("fixed.bm");
        if (has(variants, "fixed.v8.bm"))
            run_fixed.template operator()<card::v8, true>("fixed.v8.bm");
        if (has(variants, "blk8"))
            run_loop.template operator()<word, false, false, 8>("blk8");
        if (has(variants, "blk8.bm"))
            run_loop.template operator()<word, true, false, 8>("blk8.bm");
        if (has(variants, "blk8.v8"))
            run_loop.template operator()<card::v8, false, false, 8>("blk8.v8");
        if (has(variants, "blk4.v8"))
            run_loop.template operator()<card::v8, false, false, 4>("blk4.v8");
        if (has(variants, "blk2.v8"))
            run_loop.template operator()<card::v8, false, false, 2>("blk2.v8");
        if (has(variants, "blk1.v8"))
            run_loop.template operator()<card::v8, false, false, 1>("blk1.v8");
        if (has(variants, "blk1"))
            run_loop.template operator()<word, false, false, 1>("blk1");
        if (has(variants, "blk1.bm"))
            run_loop.template operator()<word, true, false, 1>("blk1.bm");
        if (has(variants, "gather.bm"))
            run_fixed.template operator()<word, true, true>("gather.bm");
        if (has(variants, "gather"))
            run_fixed.template operator()<word, false, true>("gather");
        if (has(variants, "card.bm"))
            run_card.template operator()<word, true>("card.bm");
        if (has(variants, "card.v8.bm"))
            run_card.template operator()<card::v8, true>("card.v8.bm");
        if (has(variants, "card.v8"))
            run_card.template operator()<card::v8>("card.v8");

        for (void *ptr : owned) {
            sycl::free(ptr, q);
        }
        sycl::free(out, q);
    }
    sycl::free(pool, q);
    return 0;
}
