// Cardinality gate microbenchmark. Times replicas of the at-least kernel and of
// the lowering of at most and exactly (NOT and AND gates over at-least gates)
// that #274 replaced (`prod`, `lowered`: src/mc/kernel/gate.h and
// src/runtime/pdag_from_pcfb.cc before it), and the one-word-per-work-item
// kernels of card.h, over G gates x B words with inputs drawn from a pool of
// device buffers. Every variant is checked against a host reference on sampled
// words.
//
// Shapes: atleast:K:N  atmost:K:N  exactly:K:N  range:LO:HI:N
#include "card.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using card::word;

namespace {

// ------------------------------------------------------------------ replicas of
// the replaced kernels (src/mc/kernel/gate.h before #274)
struct prod_gate { // event::atleast_gate<std::uint64_t, std::uint32_t>
    word *buffer;
    word **inputs;
    std::uint32_t num_inputs;
    std::uint32_t negated_inputs_offset;
    std::uint8_t at_least;
};

struct prod_atleast {
    const prod_gate *gates;
    std::uint32_t count;
    std::uint32_t bitpacks;
    void operator()(const sycl::nd_item<3> &item) const {
        static constexpr std::uint32_t NUM_BITS = 64;
        const auto gate_idx = static_cast<std::uint32_t>(item.get_global_id(0));
        const auto batch_idx = static_cast<std::uint32_t>(item.get_global_id(1));
        const auto bitpack_idx = static_cast<std::uint32_t>(item.get_group().get_group_id(2));
        const auto lane = item.get_local_id(2);
        if (lane >= NUM_BITS || gate_idx >= count || batch_idx >= 1 || bitpack_idx >= bitpacks) {
            return;
        }
        const std::size_t index = batch_idx * bitpacks + bitpack_idx;
        const auto &g = gates[gate_idx];
        const auto num_inputs = g.num_inputs;
        const auto negations_offset = g.negated_inputs_offset;
        const std::uint8_t k = g.at_least;
        std::uint8_t cnt = 0;
        const word mask = word(1) << lane;
        for (auto i = 0u; i < negations_offset; ++i) {
            cnt += (g.inputs[i][index] & mask) ? word(1) : word(0);
        }
        for (auto i = negations_offset; i < num_inputs; ++i) {
            cnt += (~g.inputs[i][index] & mask) ? word(1) : word(0);
        }
        const word my_bit = (cnt >= k) ? mask : word(0);
        const auto &wg = item.get_group();
        const auto lo = sycl::reduce_over_group(wg, static_cast<std::uint32_t>(my_bit), sycl::bit_or<std::uint32_t>());
        const auto hi = sycl::reduce_over_group(wg, static_cast<std::uint32_t>(my_bit >> 32), sycl::bit_or<std::uint32_t>());
        if (lane == 0) {
            g.buffer[index] = (static_cast<word>(hi) << 32) | static_cast<word>(lo);
        }
    }
};

struct prod_simple_gate { // event::gate<std::uint64_t, std::uint32_t>
    word *buffer;
    word **inputs;
    std::uint32_t num_inputs;
    std::uint32_t negated_inputs_offset;
};

template <int OP> struct prod_op { // OP 0: kNot, 1: kAnd
    const prod_simple_gate *gates;
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
        word result = OP == 1 ? ~word(0) : word(0);
        for (std::uint32_t i = 0; i < g.negated_inputs_offset; ++i) {
            const word v = g.inputs[i][index];
            result = OP == 1 ? (result & v) : v;
        }
        for (std::uint32_t i = g.negated_inputs_offset; i < g.num_inputs; ++i) {
            const word v = ~g.inputs[i][index];
            result = OP == 1 ? (result & v) : v;
        }
        if (OP == 0) {
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

// working_set::compute_optimal_local_range_3d_for_gpu, which the CPU path also
// takes.
sycl::range<3> prod_local_range(std::size_t gates, std::size_t bitpacks, std::size_t max_wg, const sycl::range<3> &max_items) {
    auto budget = static_cast<int>(std::log2(max_wg));
    const int lx = std::min(static_cast<int>(std::log2(closest_power_of_2(std::min(gates, max_items[0])))), budget);
    budget -= lx;
    const int ly = 0;
    const int lz = std::min(static_cast<int>(std::log2(closest_power_of_2(std::min(bitpacks, max_items[2])))), budget);
    return {std::size_t{1} << lx, std::size_t{1} << ly, std::size_t{1} << lz};
}

std::size_t round_up(std::size_t a, std::size_t b) { return (a + b - 1) / b * b; }

// -------------------------------------------------------------------------------------
// new kernels (card.h)
template <typename V, typename A> struct kern {
    const card::gate_desc *gates;
    std::uint32_t words;
    void operator()(const sycl::nd_item<2> &it) const {
        const std::size_t w = it.get_global_id(1) * card::lanes_v<V>;
        if (w >= words) {
            return;
        }
        const card::gate_desc g = gates[it.get_global_id(0)];
        card::st<V>(g.out + w, A::template eval<V>(g, w));
    }
};

#define CARD_ALGO(NAME, ...)                                                                                                                                   \
    struct NAME {                                                                                                                                              \
        template <typename V> static V eval(const card::gate_desc &g, std::size_t w) { return __VA_ARGS__; }                                                   \
    };
CARD_ALGO(a_ripple, card::eval_ripple_any<V>(g, w))
CARD_ALGO(a_bin, card::eval_bin_any<V, 16>(g, w))
CARD_ALGO(a_bin8, card::eval_bin_any<V, 8>(g, w))
CARD_ALGO(a_therm, card::eval_therm_upto8<V>(g, w))
CARD_ALGO(a_tblk4, card::eval_therm_blk_upto8<V, 4>(g, w))
CARD_ALGO(a_tblk8, card::eval_therm_blk_upto8<V, 8>(g, w))
CARD_ALGO(a_auto, card::eval_auto<V>(g, w))
CARD_ALGO(a_smallsw, card::eval_small_switch<V>(g, w))
template <int N> struct a_fixedM {
    template <typename V> static V eval(const card::gate_desc &g, std::size_t w) { return card::eval_fixedM<N, V>(g, w); }
};
CARD_ALGO(a_bin8m, card::eval_bin8_mid<V>(g, w))
CARD_ALGO(a_small16, card::eval_small16_switch<V>(g, w))
template <int N> struct a_fixedN {
    template <typename V> static V eval(const card::gate_desc &g, std::size_t w) { return card::eval_fixedN<N, V>(g, w); }
};
template <int N, int D> struct a_fixedND {
    template <typename V> static V eval(const card::gate_desc &g, std::size_t w) { return card::eval_fixedND<N, D, V>(g, w); }
};
template <int N, int LO, int HI> struct a_fixed {
    template <typename V> static V eval(const card::gate_desc &g, std::size_t w) { return card::eval_fixed<N, LO, HI, V>(g, w); }
};

template <typename K> sycl::event launch2(sycl::queue &q, std::size_t gates, std::size_t items, std::size_t wg, const K &k) {
    const std::size_t l = std::min(wg, items);
    return q.parallel_for(sycl::nd_range<2>({gates, round_up(items, l)}, {1, l}), k);
}

word splitmix64(word x) {
    word z = x + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct shape {
    std::string name;
    std::uint32_t n, lo, hi; // hi = n: no upper bound
};

shape parse_shape(const std::string &s) {
    std::vector<std::string> f;
    std::size_t a = 0;
    while (true) {
        const auto b = s.find(':', a);
        f.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) {
            break;
        }
        a = b + 1;
    }
    auto u = [](const std::string &x) { return static_cast<std::uint32_t>(std::stoul(x)); };
    if (f[0] == "atleast" && f.size() == 3) {
        return {s, u(f[2]), u(f[1]), u(f[2])};
    }
    if (f[0] == "atmost" && f.size() == 3) {
        return {s, u(f[2]), 0, std::min(u(f[1]), u(f[2]))};
    }
    if (f[0] == "exactly" && f.size() == 3) {
        return {s, u(f[2]), u(f[1]), std::min(u(f[1]), u(f[2]))};
    }
    if (f[0] == "range" && f.size() == 4) {
        return {s, u(f[3]), u(f[1]), std::min(u(f[2]), u(f[3]))};
    }
    std::fprintf(stderr, "bad shape %s\n", s.c_str());
    std::exit(2);
}

struct options {
    std::size_t B = 16384;
    std::size_t pool = 1024;
    std::size_t target = 2048; // G * n
    std::size_t gmin = 4, gmax = 8192;
    int reps = 20;
    double budget_ms = 3000;
    std::size_t wg = 256;
    std::uint32_t dmax = 4;
    std::string variants;
    std::vector<std::string> shapes;
    std::uint64_t seed = 1;
};

const char *kDefaultShapes[] = {
    "atleast:2:3",      "atleast:3:4",   "atleast:2:4",  "atleast:3:5",    "atleast:2:6",    "atleast:4:8",     "atleast:8:16",
    "atleast:16:32",    "atleast:32:64", "atleast:2:64", "atleast:63:64",  "atleast:64:128", "atleast:128:256", "atleast:300:600",
    "atleast:512:1024", "atmost:1:3",    "atmost:2:4",   "atmost:8:16",    "atmost:32:64",   "exactly:1:3",     "exactly:2:4",
    "exactly:8:16",     "exactly:32:64", "exactly:1:64", "range:16:48:64",
};

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

// Counted-side choice for the thermometer: depth on the ones side, D1, and on
// the zeros side, D0.
struct therm_plan {
    bool trivial_false = false, trivial_true = false, flip = false;
    std::uint32_t depth = 0, lo = 0, hi = 0;
};

therm_plan plan(std::uint32_t n, std::uint32_t lo, std::uint32_t hi) {
    therm_plan p;
    hi = std::min(hi, n);
    if (lo > hi) {
        p.trivial_false = true;
        return p;
    }
    if (lo == 0 && hi == n) {
        p.trivial_true = true;
        return p;
    }
    const std::uint32_t d1 = hi < n ? hi + 1 : lo;
    const std::uint32_t d0 = lo > 0 ? n - lo + 1 : n - hi;
    p.flip = d0 < d1;
    p.depth = std::min(d0, d1);
    p.lo = p.flip ? n - hi : lo;
    p.hi = p.flip ? n - lo : hi;
    return p;
}

} // namespace

int main(int argc, char **argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--B") {
            o.B = std::stoul(next());
        } else if (a == "--pool") {
            o.pool = std::stoul(next());
        } else if (a == "--target") {
            o.target = std::stoul(next());
        } else if (a == "--gmin") {
            o.gmin = std::stoul(next());
        } else if (a == "--gmax") {
            o.gmax = std::stoul(next());
        } else if (a == "--reps") {
            o.reps = std::stoi(next());
        } else if (a == "--budget-ms") {
            o.budget_ms = std::stod(next());
        } else if (a == "--wg") {
            o.wg = std::stoul(next());
        } else if (a == "--dmax") {
            o.dmax = static_cast<std::uint32_t>(std::stoul(next()));
        } else if (a == "--variants") {
            o.variants = next();
        } else if (a == "--shape") {
            o.shapes.push_back(next());
        } else if (a == "--seed") {
            o.seed = std::stoull(next());
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (o.shapes.empty()) {
        for (const char *s : kDefaultShapes) {
            o.shapes.emplace_back(s);
        }
    }
    sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
    const auto dev = q.get_device();
    const bool cpu = dev.is_cpu();
    if (o.variants.empty()) {
        o.variants = cpu ? "prod,lowered,auto,auto.v4,auto.v8,bin.v4,bin8.v4,bin.v8,bin8.v8,"
                           "therm.v4,tblk4.v4,tblk8.v4,fixed.v4,fixedN.v4,fixedND.v4,"
                           "fixedN.v8,fixedND.v8"
                         : "prod,lowered,ripple,bin,bin8,therm,tblk4,tblk8,auto,fixed,"
                           "fixedN,fixedND";
    }
    const std::size_t max_wg = dev.get_info<sycl::info::device::max_work_group_size>();
    const auto max_items = dev.get_info<sycl::info::device::max_work_item_sizes<3>>();
    std::printf("# device: %s | cpu=%d | max_wg=%zu | B=%zu pool=%zu target=%zu "
                "wg=%zu dmax=%u\n",
                dev.get_info<sycl::info::device::name>().c_str(), cpu ? 1 : 0, max_wg, o.B, o.pool, o.target, o.wg, o.dmax);
    std::printf("R\tshape\tn\tlo\thi\tG\tB\tvariant\tms\tGBps\tok\treps\n");

    const std::size_t B = o.B;
    word *pool = sycl::malloc_device<word>(o.pool * B, q);
    std::vector<word> pool_h(o.pool * B);
    std::mt19937_64 rng(o.seed);

    for (const auto &spec : o.shapes) {
        const shape sh = parse_shape(spec);
        const std::uint32_t n = sh.n, lo = sh.lo, hi = sh.hi;
        const std::size_t G = std::clamp<std::size_t>(o.target / n, o.gmin, o.gmax);
        const std::uint32_t npos = n - n / 2;
        const std::uint32_t nneg = n / 2;

        // Pool densities: the counted value of every input is 1 with probability ~
        // the threshold / n.
        const double thr = lo > 0 ? lo : hi;
        const auto pA = static_cast<std::uint32_t>(std::clamp(std::lround(64.0 * thr / n), 1L, 63L));
        const std::size_t half = o.pool / 2;
        const word seed = rng();
        q.parallel_for(sycl::range<1>(o.pool * B), [=](sycl::id<1> id) {
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
        q.memcpy(pool_h.data(), pool, o.pool * B * sizeof(word)).wait();

        // Gates: positive inputs from the first half of the pool, negated ones from
        // the second half.
        std::vector<std::uint32_t> idx(static_cast<std::size_t>(G) * n);
        {
            std::vector<std::uint32_t> a(half), b(o.pool - half);
            for (std::size_t i = 0; i < half; ++i) {
                a[i] = static_cast<std::uint32_t>(i);
            }
            for (std::size_t i = 0; i < b.size(); ++i) {
                b[i] = static_cast<std::uint32_t>(half + i);
            }
            for (std::size_t g = 0; g < G; ++g) {
                for (std::uint32_t j = 0; j < npos; ++j) {
                    const std::size_t r = j + rng() % (a.size() - (j % a.size()));
                    std::swap(a[j % a.size()], a[r % a.size()]);
                    idx[g * n + j] = a[j % a.size()];
                }
                for (std::uint32_t j = 0; j < nneg; ++j) {
                    const std::size_t r = j + rng() % (b.size() - (j % b.size()));
                    std::swap(b[j % b.size()], b[r % b.size()]);
                    idx[g * n + npos + j] = b[j % b.size()];
                }
            }
        }

        word *out = sycl::malloc_device<word>(G * B, q);
        word *scrA = sycl::malloc_device<word>(2 * G * B, q);
        word *scrN = sycl::malloc_device<word>(G * B, q);
        std::vector<word> out_h(G * B);

        // Reference on sampled words.
        std::vector<std::size_t> words;
        if (B <= 128) {
            for (std::size_t w = 0; w < B; ++w) {
                words.push_back(w);
            }
        } else {
            for (std::size_t w = 0; w < 32; ++w) {
                words.push_back(w);
                words.push_back(B - 1 - w);
            }
            for (int r = 0; r < 64; ++r) {
                words.push_back(rng() % B);
            }
        }
        std::vector<word> expect(G * words.size());
        for (std::size_t g = 0; g < G; ++g) {
            for (std::size_t k = 0; k < words.size(); ++k) {
                const std::size_t w = words[k];
                std::uint32_t cnt[64] = {};
                for (std::uint32_t i = 0; i < n; ++i) {
                    word x = pool_h[static_cast<std::size_t>(idx[g * n + i]) * B + w];
                    if (i >= npos) {
                        x = ~x;
                    }
                    for (int l = 0; l < 64; ++l) {
                        cnt[l] += (x >> l) & 1u;
                    }
                }
                word e = 0;
                for (int l = 0; l < 64; ++l) {
                    e |= word(cnt[l] >= lo && cnt[l] <= hi) << l;
                }
                expect[g * words.size() + k] = e;
            }
        }
        std::size_t ones = 0;
        for (const auto e : expect) {
            ones += static_cast<std::size_t>(std::popcount(e));
        }
        std::printf("# %s: n=%u [%u,%u] G=%zu density=%u/64 expected-ones=%.3f\n", spec.c_str(), n, lo, hi, G, pA,
                    static_cast<double>(ones) / (64.0 * expect.size()));

        auto check = [&](const word *dev_out) {
            q.memcpy(out_h.data(), dev_out, G * B * sizeof(word)).wait();
            std::size_t bad = 0;
            for (std::size_t g = 0; g < G; ++g) {
                for (std::size_t k = 0; k < words.size(); ++k) {
                    bad += out_h[g * B + words[k]] != expect[g * words.size() + k];
                }
            }
            return bad;
        };

        const double bytes = static_cast<double>(G) * B * 8.0 * (n + 1);
        auto report = [&](const char *variant, const std::function<std::vector<sycl::event>()> &launch, const word *result) {
            q.memset(const_cast<word *>(result), 0xA5, G * B * sizeof(word)).wait();
            launch();
            q.wait();
            const std::size_t bad = check(result);
            const timing t = time_it(q, launch, o.reps, o.budget_ms);
            std::printf("R\t%s\t%u\t%u\t%u\t%zu\t%zu\t%s\t%.5f\t%.1f\t%s\t%d\n", spec.c_str(), n, lo, hi, G, B, variant, t.ms, bytes / (t.ms * 1e6),
                        bad == 0 ? "ok" : "WRONG", t.reps);
            std::fflush(stdout);
        };

        // ---- the replaced kernels: at-least gates, then NOT / AND for at-most,
        // exactly and ranges.
        const bool atleast = hi >= n;
        auto build_prod = [&](std::vector<prod_gate> &gs, std::vector<word *> &ptrs, std::uint32_t k, word *buf, std::size_t gi_offset) {
            for (std::size_t g = 0; g < G; ++g) {
                for (std::uint32_t i = 0; i < n; ++i) {
                    ptrs.push_back(pool + static_cast<std::size_t>(idx[g * n + i]) * B);
                }
                gs.push_back({buf + (gi_offset + g) * B, nullptr, n, npos, static_cast<std::uint8_t>(k)});
            }
        };
        if (has(o.variants, "prod") && atleast) {
            std::vector<prod_gate> gs;
            std::vector<word *> ptrs;
            build_prod(gs, ptrs, lo, out, 0);
            prod_gate *dg = sycl::malloc_shared<prod_gate>(gs.size(), q);
            word **dp = sycl::malloc_shared<word *>(ptrs.size(), q);
            std::copy(ptrs.begin(), ptrs.end(), dp);
            for (std::size_t g = 0; g < gs.size(); ++g) {
                gs[g].inputs = dp + g * n;
            }
            std::copy(gs.begin(), gs.end(), dg);
            const auto count = static_cast<std::uint32_t>(gs.size());
            const auto bp = static_cast<std::uint32_t>(B);
            report(
                "prod",
                [&]() { return std::vector<sycl::event>{q.parallel_for(sycl::nd_range<3>({count, 1, B * 64}, {1, 1, 64}), prod_atleast{dg, count, bp})}; },
                out);
            sycl::free(dg, q);
            sycl::free(dp, q);
        }
        if (has(o.variants, "lowered") && !atleast) {
            // ATLEAST(lo) for gates [0, G) when lo > 0, ATLEAST(hi + 1) for gates [G,
            // 2G); NOT; AND when lo > 0.
            std::vector<prod_gate> gs;
            std::vector<word *> ptrs;
            if (lo > 0) {
                build_prod(gs, ptrs, lo, scrA, 0);
            }
            build_prod(gs, ptrs, hi + 1, scrA, G);
            prod_gate *dg = sycl::malloc_shared<prod_gate>(gs.size(), q);
            word **dp = sycl::malloc_shared<word *>(ptrs.size() + 3 * G, q);
            std::copy(ptrs.begin(), ptrs.end(), dp);
            for (std::size_t g = 0; g < gs.size(); ++g) {
                gs[g].inputs = dp + g * n;
            }
            std::copy(gs.begin(), gs.end(), dg);
            // NOT gates over ATLEAST(hi + 1); AND gates over (ATLEAST(lo), NOT).
            word **np = dp + ptrs.size();
            word **ap = np + G;
            prod_simple_gate *ng = sycl::malloc_shared<prod_simple_gate>(G, q);
            prod_simple_gate *ag = sycl::malloc_shared<prod_simple_gate>(G, q);
            for (std::size_t g = 0; g < G; ++g) {
                np[g] = scrA + (G + g) * B;
                ng[g] = {lo > 0 ? scrN + g * B : out + g * B, np + g, 1, 1};
                ap[2 * g] = scrA + g * B;
                ap[2 * g + 1] = scrN + g * B;
                ag[g] = {out + g * B, ap + 2 * g, 2, 2};
            }
            const auto count = static_cast<std::uint32_t>(gs.size());
            const auto gcount = static_cast<std::uint32_t>(G);
            const auto bp = static_cast<std::uint32_t>(B);
            const auto lr = prod_local_range(G, B, max_wg, max_items);
            const sycl::nd_range<3> op_range({round_up(G, lr[0]), 1, round_up(B, lr[2])}, lr);
            report(
                lo > 0 ? "lowered:al2+not+and" : "lowered:al+not",
                [&]() {
                    std::vector<sycl::event> evs;
                    evs.push_back(q.parallel_for(sycl::nd_range<3>({count, 1, B * 64}, {1, 1, 64}), prod_atleast{dg, count, bp}));
                    evs.push_back(q.parallel_for(op_range, prod_op<0>{ng, gcount, bp}));
                    if (lo > 0) {
                        evs.push_back(q.parallel_for(op_range, prod_op<1>{ag, gcount, bp}));
                    }
                    return evs;
                },
                out);
            sycl::free(dg, q);
            sycl::free(dp, q);
            sycl::free(ng, q);
            sycl::free(ag, q);
        }

        // ---- new kernels. Descriptor sets: plain (input order as given, arg =
        // planes) and planned (counted side flipped where the thermometer is
        // cheaper on the zeros side, arg = depth).
        const therm_plan tp = plan(n, lo, hi);
        const auto planes = static_cast<std::uint32_t>(std::bit_width(n));
        auto make_set = [&](bool use_plan, std::uint32_t dmax, std::vector<const word *> &ptrs) {
            std::vector<card::gate_desc> d(G);
            ptrs.clear();
            for (std::size_t g = 0; g < G; ++g) {
                card::gate_desc &x = d[g];
                x.out = out + g * B;
                x.n = n;
                const bool therm = use_plan && !tp.trivial_false && !tp.trivial_true && tp.depth <= dmax;
                if (use_plan && tp.trivial_false) {
                    x.cls = card::k_false;
                } else if (use_plan && tp.trivial_true) {
                    x.cls = card::k_true;
                } else {
                    x.cls = therm ? card::k_therm : card::k_bin;
                }
                const bool flip = therm && tp.flip;
                const std::size_t base = ptrs.size();
                if (!flip) {
                    for (std::uint32_t i = 0; i < n; ++i) {
                        ptrs.push_back(pool + static_cast<std::size_t>(idx[g * n + i]) * B);
                    }
                    x.neg = npos;
                    x.lo = therm ? tp.lo : lo;
                    x.hi = therm ? tp.hi : hi;
                } else { // count zeros: negated inputs read as they are, positive ones
                         // complemented
                    for (std::uint32_t i = npos; i < n; ++i) {
                        ptrs.push_back(pool + static_cast<std::size_t>(idx[g * n + i]) * B);
                    }
                    for (std::uint32_t i = 0; i < npos; ++i) {
                        ptrs.push_back(pool + static_cast<std::size_t>(idx[g * n + i]) * B);
                    }
                    x.neg = nneg;
                    x.lo = tp.lo;
                    x.hi = tp.hi;
                }
                x.arg = therm ? tp.depth : planes;
                x.mask = n >= 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << n) - 1);
                x.in = reinterpret_cast<const word *const *>(base); // patched below
            }
            return d;
        };
        auto upload = [&](std::vector<card::gate_desc> d, const std::vector<const word *> &ptrs, std::vector<void *> &owned) {
            const word **dp = sycl::malloc_device<const word *>(ptrs.size(), q);
            q.memcpy(dp, ptrs.data(), ptrs.size() * sizeof(const word *)).wait();
            for (auto &x : d) {
                x.in = dp + reinterpret_cast<std::size_t>(x.in);
            }
            card::gate_desc *dd = sycl::malloc_device<card::gate_desc>(d.size(), q);
            q.memcpy(dd, d.data(), d.size() * sizeof(card::gate_desc)).wait();
            owned.push_back(dp);
            owned.push_back(dd);
            return static_cast<const card::gate_desc *>(dd);
        };
        std::vector<void *> owned;
        std::vector<const word *> ptrs;
        const card::gate_desc *d_plain = upload(make_set(false, 0, ptrs), ptrs, owned);
        const card::gate_desc *d_therm = upload(make_set(true, 8, ptrs), ptrs, owned);
        const card::gate_desc *d_auto = upload(make_set(true, o.dmax, ptrs), ptrs, owned);
        const auto bw = static_cast<std::uint32_t>(B);
        const std::size_t wg = o.wg;

        // #274's runtime lowering of exactly / ranges: one cardinality launch over 2G gates (at least lo of
        // the inputs, at least n - hi of their complements), then the replaced AND kernel.
        if (has(o.variants, "lowered.new") && lo > 0 && hi < n) {
            std::vector<card::gate_desc> d2(2 * G);
            std::vector<const word *> p2;
            for (std::size_t g = 0; g < 2 * G; ++g) {
                const std::size_t src = g % G;
                const bool upper = g >= G;
                card::gate_desc &x = d2[g];
                x.out = scrA + g * B;
                x.n = n;
                x.cls = card::k_bin;
                x.arg = static_cast<std::uint32_t>(std::bit_width(n));
                x.in = reinterpret_cast<const word *const *>(p2.size());
                if (!upper) {
                    for (std::uint32_t i = 0; i < n; ++i) {
                        p2.push_back(pool + static_cast<std::size_t>(idx[src * n + i]) * B);
                    }
                    x.neg = npos;
                    x.lo = lo;
                    x.hi = n;
                } else { // complements: the negated inputs read as they are, then the positive ones complemented
                    for (std::uint32_t i = npos; i < n; ++i) {
                        p2.push_back(pool + static_cast<std::size_t>(idx[src * n + i]) * B);
                    }
                    for (std::uint32_t i = 0; i < npos; ++i) {
                        p2.push_back(pool + static_cast<std::size_t>(idx[src * n + i]) * B);
                    }
                    x.neg = nneg;
                    x.lo = n - hi;
                    x.hi = n;
                }
            }
            std::vector<void *> owned2;
            const card::gate_desc *dd2 = upload(d2, p2, owned2);
            prod_simple_gate *ag = sycl::malloc_shared<prod_simple_gate>(G, q);
            word **ap = sycl::malloc_shared<word *>(2 * G, q);
            for (std::size_t g = 0; g < G; ++g) {
                ap[2 * g] = scrA + g * B;
                ap[2 * g + 1] = scrA + (G + g) * B;
                ag[g] = {out + g * B, ap + 2 * g, 2, 2};
            }
            const auto gcount = static_cast<std::uint32_t>(G);
            const auto bp = static_cast<std::uint32_t>(B);
            const auto lr = prod_local_range(G, B, max_wg, max_items);
            const sycl::nd_range<3> op_range({round_up(G, lr[0]), 1, round_up(B, lr[2])}, lr);
            const std::size_t wgl = o.wg;
            const std::size_t lanes = cpu ? 8 : 1;
            report(
                "lowered.new",
                [&]() {
                    std::vector<sycl::event> evs;
                    if (n <= 8) {
                        if (cpu) {
                            evs.push_back(launch2(q, 2 * G, B / lanes, wgl, kern<card::v8, a_smallsw>{dd2, static_cast<std::uint32_t>(B)}));
                        } else {
                            evs.push_back(launch2(q, 2 * G, B, wgl, kern<word, a_smallsw>{dd2, static_cast<std::uint32_t>(B)}));
                        }
                    } else {
                        if (cpu) {
                            evs.push_back(launch2(q, 2 * G, B / lanes, wgl, kern<card::v8, a_bin8m>{dd2, static_cast<std::uint32_t>(B)}));
                        } else {
                            evs.push_back(launch2(q, 2 * G, B, wgl, kern<word, a_bin8m>{dd2, static_cast<std::uint32_t>(B)}));
                        }
                    }
                    evs.push_back(q.parallel_for(op_range, prod_op<1>{ag, gcount, bp}));
                    return evs;
                },
                out);
            for (void *p : owned2) {
                sycl::free(p, q);
            }
            sycl::free(ag, q);
            sycl::free(ap, q);
        }
        // Fixed-size sets: each gate's pointer array padded with nullptr to N slots, mask = its real inputs.
        auto padded_set = [&](std::uint32_t N, std::vector<void *> &own) {
            std::vector<card::gate_desc> d(G);
            std::vector<const word *> p;
            for (std::size_t g = 0; g < G; ++g) {
                card::gate_desc &x = d[g];
                x.out = out + g * B;
                x.n = n;
                x.neg = npos;
                x.lo = lo;
                x.hi = hi;
                x.cls = card::k_bin;
                x.arg = 0;
                x.mask = n >= 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << n) - 1);
                x.in = reinterpret_cast<const word *const *>(p.size());
                for (std::uint32_t i = 0; i < N; ++i) {
                    p.push_back(i < n ? pool + static_cast<std::size_t>(idx[g * n + i]) * B : nullptr);
                }
            }
            return upload(d, p, own);
        };
        auto pad_class = [&](const std::string &rule) -> std::uint32_t {
            static const std::uint32_t pow2[] = {1, 2, 4, 8, 16, 32, 64};
            static const std::uint32_t pow2_8[] = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64};
            static const std::uint32_t steps[] = {1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 48, 64};
            static const std::uint32_t prod[] = {1, 2, 3, 4, 6, 8, 12, 16};
            auto pick = [&](const auto &list) -> std::uint32_t {
                for (const auto c : list) {
                    if (c >= n) {
                        return c;
                    }
                }
                return 0;
            };
            if (rule == "pad2")
                return pick(pow2);
            if (rule == "pad8")
                return pick(pow2_8);
            if (rule == "pad15")
                return pick(steps);
            if (rule == "padp")
                return pick(prod);
            return 0;
        };
        const bool therm_ok = !tp.trivial_false && !tp.trivial_true && tp.depth <= 8;
        auto run = [&]<typename V, typename A>(const std::string &name, const card::gate_desc *d) {
            report(name.c_str(), [&]() { return std::vector<sycl::event>{launch2(q, G, B / card::lanes_v<V>, wg, kern<V, A>{d, bw})}; }, out);
            return true;
        };
        auto dispatch = [&]<typename V>(const std::string &algo, const std::string &name) -> bool {
            if (algo == "ripple") {
                return run.template operator()<V, a_ripple>(name, d_plain);
            }
            if (algo == "bin") {
                return run.template operator()<V, a_bin>(name, d_plain);
            }
            if (algo == "bin8") {
                return run.template operator()<V, a_bin8>(name, d_plain);
            }
            if (algo == "smallsw" && n <= 8) {
                return run.template operator()<V, a_smallsw>(name, d_plain);
            }
            if (algo == "bin8m" && n < 256) {
                return run.template operator()<V, a_bin8m>(name, d_plain);
            }
            if (algo == "small16" && n <= 16) {
                return run.template operator()<V, a_small16>(name, d_plain);
            }
            if (algo == "auto") {
                return run.template operator()<V, a_auto>(name, d_auto);
            }
            if (algo == "therm" && therm_ok) {
                return run.template operator()<V, a_therm>(name, d_therm);
            }
            if (algo == "tblk4" && therm_ok) {
                return run.template operator()<V, a_tblk4>(name, d_therm);
            }
            if (algo == "tblk8" && therm_ok) {
                return run.template operator()<V, a_tblk8>(name, d_therm);
            }
            if (algo == "pad2" || algo == "pad8" || algo == "pad15" || algo == "padp") {
                const std::uint32_t N = pad_class(algo);
                if (N == 0) {
                    return false;
                }
                std::vector<void *> own;
                const card::gate_desc *d = padded_set(N, own);
                bool done = false;
                const auto try_slots = [&]<std::uint32_t K>() {
                    if (N == K) {
                        done = run.template operator()<V, a_fixedM<K>>(name, d);
                    }
                };
                [&]<std::uint32_t... K>(std::integer_sequence<std::uint32_t, K...>) {
                    (try_slots.template operator()<K>(), ...);
                }(std::integer_sequence<std::uint32_t, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64>{});
                for (void *p : own) {
                    sycl::free(p, q);
                }
                return done;
            }
            if (algo == "fixedN") {
                switch (n) {
                case 2:
                    return run.template operator()<V, a_fixedN<2>>(name, d_plain);
                case 3:
                    return run.template operator()<V, a_fixedN<3>>(name, d_plain);
                case 4:
                    return run.template operator()<V, a_fixedN<4>>(name, d_plain);
                case 5:
                    return run.template operator()<V, a_fixedN<5>>(name, d_plain);
                case 6:
                    return run.template operator()<V, a_fixedN<6>>(name, d_plain);
                case 7:
                    return run.template operator()<V, a_fixedN<7>>(name, d_plain);
                case 8:
                    return run.template operator()<V, a_fixedN<8>>(name, d_plain);
                case 9:
                    return run.template operator()<V, a_fixedN<9>>(name, d_plain);
                case 10:
                    return run.template operator()<V, a_fixedN<10>>(name, d_plain);
                case 11:
                    return run.template operator()<V, a_fixedN<11>>(name, d_plain);
                case 12:
                    return run.template operator()<V, a_fixedN<12>>(name, d_plain);
                case 13:
                    return run.template operator()<V, a_fixedN<13>>(name, d_plain);
                case 14:
                    return run.template operator()<V, a_fixedN<14>>(name, d_plain);
                case 15:
                    return run.template operator()<V, a_fixedN<15>>(name, d_plain);
                case 16:
                    return run.template operator()<V, a_fixedN<16>>(name, d_plain);
                default:
                    return false;
                }
            }
            if (algo == "fixedND" && therm_ok && n <= 8 && tp.depth <= 4) {
                switch (n * 8 + tp.depth) {
                case 17:
                    return run.template operator()<V, a_fixedND<2, 1>>(name, d_therm);
                case 18:
                    return run.template operator()<V, a_fixedND<2, 2>>(name, d_therm);
                case 25:
                    return run.template operator()<V, a_fixedND<3, 1>>(name, d_therm);
                case 26:
                    return run.template operator()<V, a_fixedND<3, 2>>(name, d_therm);
                case 27:
                    return run.template operator()<V, a_fixedND<3, 3>>(name, d_therm);
                case 33:
                    return run.template operator()<V, a_fixedND<4, 1>>(name, d_therm);
                case 34:
                    return run.template operator()<V, a_fixedND<4, 2>>(name, d_therm);
                case 35:
                    return run.template operator()<V, a_fixedND<4, 3>>(name, d_therm);
                case 36:
                    return run.template operator()<V, a_fixedND<4, 4>>(name, d_therm);
                case 41:
                    return run.template operator()<V, a_fixedND<5, 1>>(name, d_therm);
                case 42:
                    return run.template operator()<V, a_fixedND<5, 2>>(name, d_therm);
                case 43:
                    return run.template operator()<V, a_fixedND<5, 3>>(name, d_therm);
                case 44:
                    return run.template operator()<V, a_fixedND<5, 4>>(name, d_therm);
                case 49:
                    return run.template operator()<V, a_fixedND<6, 1>>(name, d_therm);
                case 50:
                    return run.template operator()<V, a_fixedND<6, 2>>(name, d_therm);
                case 51:
                    return run.template operator()<V, a_fixedND<6, 3>>(name, d_therm);
                case 52:
                    return run.template operator()<V, a_fixedND<6, 4>>(name, d_therm);
                case 57:
                    return run.template operator()<V, a_fixedND<7, 1>>(name, d_therm);
                case 58:
                    return run.template operator()<V, a_fixedND<7, 2>>(name, d_therm);
                case 59:
                    return run.template operator()<V, a_fixedND<7, 3>>(name, d_therm);
                case 60:
                    return run.template operator()<V, a_fixedND<7, 4>>(name, d_therm);
                case 65:
                    return run.template operator()<V, a_fixedND<8, 1>>(name, d_therm);
                case 66:
                    return run.template operator()<V, a_fixedND<8, 2>>(name, d_therm);
                case 67:
                    return run.template operator()<V, a_fixedND<8, 3>>(name, d_therm);
                case 68:
                    return run.template operator()<V, a_fixedND<8, 4>>(name, d_therm);
                default:
                    return false;
                }
            }
            if (algo == "fixed") {
                if (n == 3 && lo == 2 && hi == 3) {
                    return run.template operator()<V, a_fixed<3, 2, 3>>(name, d_plain);
                }
                if (n == 4 && lo == 3 && hi == 4) {
                    return run.template operator()<V, a_fixed<4, 3, 4>>(name, d_plain);
                }
                if (n == 4 && lo == 2 && hi == 4) {
                    return run.template operator()<V, a_fixed<4, 2, 4>>(name, d_plain);
                }
                if (n == 5 && lo == 3 && hi == 5) {
                    return run.template operator()<V, a_fixed<5, 3, 5>>(name, d_plain);
                }
                if (n == 6 && lo == 2 && hi == 6) {
                    return run.template operator()<V, a_fixed<6, 2, 6>>(name, d_plain);
                }
                if (n == 8 && lo == 4 && hi == 8) {
                    return run.template operator()<V, a_fixed<8, 4, 8>>(name, d_plain);
                }
                if (n == 3 && lo == 0 && hi == 1) {
                    return run.template operator()<V, a_fixed<3, 0, 1>>(name, d_plain);
                }
                if (n == 4 && lo == 0 && hi == 2) {
                    return run.template operator()<V, a_fixed<4, 0, 2>>(name, d_plain);
                }
                if (n == 3 && lo == 1 && hi == 1) {
                    return run.template operator()<V, a_fixed<3, 1, 1>>(name, d_plain);
                }
                if (n == 4 && lo == 2 && hi == 2) {
                    return run.template operator()<V, a_fixed<4, 2, 2>>(name, d_plain);
                }
            }
            return false;
        };
        {
            std::size_t a0 = 0;
            while (a0 <= o.variants.size()) {
                const auto b0 = o.variants.find(',', a0);
                const std::string v = o.variants.substr(a0, b0 == std::string::npos ? std::string::npos : b0 - a0);
                a0 = b0 == std::string::npos ? o.variants.size() + 1 : b0 + 1;
                if (v.empty() || v == "prod" || v == "lowered") {
                    continue;
                }
                const auto dot = v.find('.');
                const std::string algo = v.substr(0, dot);
                const std::string lanes = dot == std::string::npos ? "" : v.substr(dot + 1);
                if (lanes == "v4") {
                    dispatch.template operator()<card::v4>(algo, v);
                } else if (lanes == "v8") {
                    dispatch.template operator()<card::v8>(algo, v);
                } else {
                    dispatch.template operator()<word>(algo, v);
                }
            }
        }
        for (void *p : owned) {
            sycl::free(p, q);
        }
        sycl::free(out, q);
        sycl::free(scrA, q);
        sycl::free(scrN, q);
    }
    sycl::free(pool, q);
    return 0;
}
