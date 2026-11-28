// Philox Bernoulli sampler microbenchmark: launches each sampler variant of samplers.h and the production
// forms of src/mc/prng/bernoulli/ over events x bitpacks, reports the median launch time and checks
// every variant's bits against a host reference. Build and run: see lab/kernels/README.md.
//
// The production forms (pstrided, plockstep) sample kStream at the counters of src/mc/prng/stream.h:
// launch(it) is pass `it`, bitpacks [it·B, (it + 1)·B) of each event's run. The variants of samplers.h keep
// lab::counter, the counter of the samplers they replicate, so their bitpacks differ from the production forms':
// each group is compared with its own first output, and the two bit-sliced host references (lab::bs_reference,
// prng::bernoulli::sample) are checked against each other at both groups' counters.
#include "mc/prng/bernoulli/bernoulli.h"
#include "mc/prng/stream.h"
#include "samplers.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace {

namespace prng = fuzzybits::mc::prng;

// The production forms' stream: the key lab::philox has built in (lab::kK0, lab::kK1, the engine's fixed key before
// runs had seeds), which the cross-checks of the host references need, on device 0; not the default stream (seed
// stream::kDefaultSeed).
constexpr prng::stream kStream{.key = {.k0 = lab::kK0, .k1 = lab::kK1}, .device = 0};

// kernel::sampler::stream_argument, repeated because build.sh copies only src/mc/prng: the expanded stream (the key's ten
// round keys and the device) is specialized where AdaptiveCpp offers it, so the generic JIT folds the round keys into
// immediates as in production.
#if defined(ACPP_EXT_SPECIALIZED)
using stream_argument = sycl::specialized<prng::expanded_stream>;
#else
using stream_argument = prng::expanded_stream;
#endif

std::uint64_t probability_to_threshold(double p) {
    if (!(p > 0.0)) {
        return 0;
    }
    if (p >= 1.0) {
        return ~0ull;
    }
    const auto bits = std::bit_cast<std::uint64_t>(p);
    const auto exp_bits = (bits >> 52U) & 0x7FFU;
    const auto frac = bits & 0xFFFFFFFFFFFFFULL;
    if (exp_bits == 0) {
        return 0;
    }
    const int shift = static_cast<int>(exp_bits) - 1023 + 12;
    const auto sig = (std::uint64_t{1} << 52U) | frac;
    if (shift <= -64) {
        return 0;
    }
    if (shift < 0) {
        return sig >> static_cast<unsigned>(-shift);
    }
    return sig << static_cast<unsigned>(shift);
}

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

struct Options {
    std::size_t events = 122;
    std::size_t bitpacks = 131072;
    int reps = 12;
    int warm = 2;
    std::string pmode = "0.01";
    std::vector<std::string> variants;
    std::size_t wg = 256;
    std::size_t persist = 16;
    std::size_t simd_wg = 64;
    bool verify = true;
};

struct EventData {
    std::uint32_t *pdag = nullptr;
    std::uint64_t *thr32 = nullptr;
    std::uint64_t *T = nullptr;
    std::uint8_t *always = nullptr;
};

} // namespace

int main(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--events") {
            o.events = std::stoul(next());
        } else if (a == "--bitpacks") {
            o.bitpacks = std::stoul(next());
        } else if (a == "--reps") {
            o.reps = std::stoi(next());
        } else if (a == "--warm") {
            o.warm = std::stoi(next());
        } else if (a == "--p") {
            o.pmode = next();
        } else if (a == "--wg") {
            o.wg = std::stoul(next());
        } else if (a == "--persist") {
            o.persist = std::stoul(next());
        } else if (a == "--simd-wg") {
            o.simd_wg = std::stoul(next());
        } else if (a == "--no-verify") {
            o.verify = false;
        } else {
            o.variants.push_back(a);
        }
    }
    if (o.variants.empty()) {
        o.variants = {"prod", "prod16", "bs3d", "bs", "bs16", "bsp4", "pers", "simd8"};
    }

    sycl::queue q{sycl::property::queue::enable_profiling{}};
    const auto dev = q.get_device();
    const std::size_t maxwg = dev.get_info<sycl::info::device::max_work_group_size>();
    std::printf("device: %s | cpu=%d | max_wg=%zu | events=%zu bitpacks=%zu reps=%d p=%s\n", dev.get_info<sycl::info::device::name>().c_str(),
                dev.is_cpu() ? 1 : 0, maxwg, o.events, o.bitpacks, o.reps, o.pmode.c_str());

    // Probabilities.
    std::vector<double> p(o.events);
    if (o.pmode == "mix") {
        // Plant-model-like spread: log-uniform over [1e-7, 0.3] plus special values.
        const double special[] = {1.0, 0.5, 0.1, 1e-3, 0.0, 2.0e-10, 1e-12};
        for (std::size_t e = 0; e < o.events; ++e) {
            if (e % 16 < 7 && e < 16) {
                p[e] = special[e % 16];
            } else {
                const double u = static_cast<double>((e * 2654435761u) % 10007) / 10007.0;
                p[e] = std::pow(10.0, -7.0 + u * (7.0 + std::log10(0.3)));
            }
        }
    } else {
        std::fill(p.begin(), p.end(), std::stod(o.pmode));
    }

    std::vector<std::uint32_t> h_pdag(o.events);
    std::vector<std::uint64_t> h_thr32(o.events), h_T(o.events);
    std::vector<std::uint8_t> h_always(o.events);
    for (std::size_t e = 0; e < o.events; ++e) {
        h_pdag[e] = static_cast<std::uint32_t>(e + 2);
        h_thr32[e] = static_cast<std::uint64_t>(p[e] * 4294967296.0);
        h_always[e] = p[e] >= 1.0 ? 1 : 0;
        h_T[e] = h_always[e] ? ~0ull : probability_to_threshold(p[e]);
    }
    EventData ev;
    ev.pdag = sycl::malloc_device<std::uint32_t>(o.events, q);
    ev.thr32 = sycl::malloc_device<std::uint64_t>(o.events, q);
    ev.T = sycl::malloc_device<std::uint64_t>(o.events, q);
    ev.always = sycl::malloc_device<std::uint8_t>(o.events, q);
    q.memcpy(ev.pdag, h_pdag.data(), o.events * 4).wait();
    q.memcpy(ev.thr32, h_thr32.data(), o.events * 8).wait();
    q.memcpy(ev.T, h_T.data(), o.events * 8).wait();
    q.memcpy(ev.always, h_always.data(), o.events).wait();

    const std::size_t E = o.events, B = o.bitpacks;
    std::uint64_t *out = sycl::malloc_device<std::uint64_t>(E * B, q);

    // Production 3D launch geometry (compute_optimal_local_range_3d_for_gpu).
    const auto log2maxwg = static_cast<unsigned>(std::log2(static_cast<double>(maxwg)));
    const auto wis = dev.get_info<sycl::info::device::max_work_item_sizes<3>>();
    const unsigned lx2 = std::min<unsigned>(static_cast<unsigned>(std::log2(closest_power_of_2(std::min<std::size_t>(E, wis[0])))), log2maxwg);
    const unsigned lz2 = std::min<unsigned>(static_cast<unsigned>(std::log2(closest_power_of_2(std::min<std::size_t>(B, wis[2])))), log2maxwg - lx2);
    const sycl::range<3> local3{std::size_t{1} << lx2, 1, std::size_t{1} << lz2};
    const sycl::range<3> global3{(E + local3[0] - 1) / local3[0] * local3[0], 1, (B + local3[2] - 1) / local3[2] * local3[2]};

    std::printf("prod local3=(%zu,%zu,%zu) global3=(%zu,%zu,%zu)\n", local3[0], local3[1], local3[2], global3[0], global3[1], global3[2]);
    using Launch = std::function<sycl::event(std::uint32_t)>;
    std::map<std::string, Launch> launches;

    const auto pdag = ev.pdag;
    const auto thr32 = ev.thr32;
    const auto Tdev = ev.T;
    const auto always = ev.always;

    auto prod3d = [&](auto split) -> Launch {
        constexpr bool S = decltype(split)::value;
        return [=, &q](std::uint32_t it) {
            return q.parallel_for(sycl::nd_range<3>{global3, local3}, [=](sycl::nd_item<3> item) {
                const auto e = static_cast<std::uint32_t>(item.get_global_id(0));
                const auto s = static_cast<std::uint32_t>(item.get_global_id(2));
                if (e >= E || s >= B) {
                    return;
                }
                const lab::state128 base = lab::counter(pdag[e], e, s, it);
                out[e * B + s] = lab::sample_production<S>(base, thr32[e]);
            });
        };
    };
    launches["prod"] = prod3d(std::false_type{});
    launches["prod16"] = prod3d(std::true_type{});

    launches["bs3d"] = [=, &q](std::uint32_t it) {
        return q.parallel_for(sycl::nd_range<3>{global3, local3}, [=](sycl::nd_item<3> item) {
            const auto e = static_cast<std::uint32_t>(item.get_global_id(0));
            const auto s = static_cast<std::uint32_t>(item.get_global_id(2));
            if (e >= E || s >= B) {
                return;
            }
            const lab::state128 base = lab::counter(pdag[e], e, s, it);
            out[e * B + s] = always[e] ? ~0ull : lab::sample_bitsliced<false>(base, Tdev[e]);
        });
    };

    const std::size_t wg = o.wg;
    const sycl::range<2> local2{1, wg};
    const sycl::range<2> global2{E, (B + wg - 1) / wg * wg};
    auto bs2d = [&](auto split, auto prefix) -> Launch {
        constexpr bool S = decltype(split)::value;
        constexpr int P = decltype(prefix)::value;
        return [=, &q](std::uint32_t it) {
            return q.parallel_for(sycl::nd_range<2>{global2, local2}, [=](sycl::nd_item<2> item) {
                const auto e = static_cast<std::uint32_t>(item.get_global_id(0));
                const auto s = static_cast<std::uint32_t>(item.get_global_id(1));
                if (s >= B) {
                    return;
                }
                const lab::state128 base = lab::counter(pdag[e], e, s, it);
                std::uint64_t v;
                if (always[e]) {
                    v = ~0ull;
                } else if constexpr (P == 0) {
                    v = lab::sample_bitsliced<S>(base, Tdev[e]);
                } else {
                    v = lab::sample_bitsliced_prefix<S, P>(base, Tdev[e]);
                }
                out[e * B + s] = v;
            });
        };
    };
    launches["bs"] = bs2d(std::false_type{}, std::integral_constant<int, 0>{});
    launches["bs16"] = bs2d(std::true_type{}, std::integral_constant<int, 0>{});
    launches["bsp4"] = bs2d(std::false_type{}, std::integral_constant<int, 4>{});
    launches["bsp4s"] = bs2d(std::true_type{}, std::integral_constant<int, 4>{});
    launches["bsp2"] = bs2d(std::false_type{}, std::integral_constant<int, 2>{});

    // Persistent: per event, S threads each walking ceil(B/S) bitpacks with stride S.
    const std::size_t K = o.persist;
    const std::size_t Sthreads = std::max<std::size_t>(1, (B + K - 1) / K);
    const sycl::range<2> globalP{E, (Sthreads + wg - 1) / wg * wg};
    auto pers = [&](auto split) -> Launch {
        constexpr bool S = decltype(split)::value;
        return [=, &q](std::uint32_t it) {
            return q.parallel_for(sycl::nd_range<2>{globalP, local2}, [=](sycl::nd_item<2> item) {
                const auto e = static_cast<std::uint32_t>(item.get_global_id(0));
                const auto t = static_cast<std::uint32_t>(item.get_global_id(1));
                if (t >= Sthreads) {
                    return;
                }
                const auto stride = static_cast<std::uint32_t>(Sthreads);
                const auto count = static_cast<std::uint32_t>((B - t + Sthreads - 1) / Sthreads);
                std::uint64_t *row = out + e * B;
                if (always[e]) {
                    for (std::uint32_t i = 0; i < count; ++i) {
                        row[t + i * stride] = ~0ull;
                    }
                    return;
                }
                lab::sample_bitsliced_persistent<S>(row, pdag[e], e, t, stride, count, it, Tdev[e]);
            });
        };
    };
    launches["pers"] = pers(std::false_type{});
    launches["pers16"] = pers(std::true_type{});

    auto simd = [&](auto width) -> Launch {
        constexpr int W = decltype(width)::value;
        const std::size_t items = (B + W - 1) / W;
        const std::size_t swg = o.simd_wg;
        const sycl::range<2> lS{1, swg};
        const sycl::range<2> gS{E, (items + swg - 1) / swg * swg};
        return [=, &q](std::uint32_t it) {
            return q.parallel_for(sycl::nd_range<2>{gS, lS}, [=](sycl::nd_item<2> item) {
                const auto e = static_cast<std::uint32_t>(item.get_global_id(0));
                const auto j = static_cast<std::uint32_t>(item.get_global_id(1));
                if (j >= items) {
                    return;
                }
                std::uint64_t *dst = out + e * B + static_cast<std::size_t>(j) * W;
                if (always[e]) {
#pragma unroll
                    for (int i = 0; i < W; ++i) {
                        dst[i] = ~0ull;
                    }
                    return;
                }
                lab::sample_bitsliced_simd<W>(dst, pdag[e], e, j * W, it, Tdev[e]);
            });
        };
    };
    auto simd64 = [&](auto width) -> Launch {
        constexpr int W = decltype(width)::value;
        const std::size_t items = (B + W - 1) / W;
        const std::size_t swg = o.simd_wg;
        const sycl::range<2> lS{1, swg};
        const sycl::range<2> gS{E, (items + swg - 1) / swg * swg};
        return [=, &q](std::uint32_t it) {
            return q.parallel_for(sycl::nd_range<2>{gS, lS}, [=](sycl::nd_item<2> item) {
                const auto e = static_cast<std::uint32_t>(item.get_global_id(0));
                const auto j = static_cast<std::uint32_t>(item.get_global_id(1));
                if (j >= items) {
                    return;
                }
                std::uint64_t *dst = out + e * B + static_cast<std::size_t>(j) * W;
                if (always[e]) {
#pragma unroll
                    for (int i = 0; i < W; ++i) {
                        dst[i] = ~0ull;
                    }
                    return;
                }
                lab::sample_bitsliced_simd64<W>(dst, pdag[e], e, j * W, it, Tdev[e]);
            });
        };
    };
    launches["v64x4"] = simd64(std::integral_constant<int, 4>{});
    launches["v64x8"] = simd64(std::integral_constant<int, 8>{});
    launches["v64x16"] = simd64(std::integral_constant<int, 16>{});
    auto simdp = [&](auto width) -> Launch {
        constexpr int W = decltype(width)::value;
        // Each work-item covers W*K consecutive bitpacks (K = --persist).
        const std::size_t per_item = static_cast<std::size_t>(W) * o.persist;
        const std::size_t items = B / per_item;
        const std::size_t swg = o.simd_wg;
        const sycl::range<2> lS{1, swg};
        const sycl::range<2> gS{E, (items + swg - 1) / swg * swg};
        const auto K32 = static_cast<std::uint32_t>(o.persist);
        return [=, &q](std::uint32_t it) {
            return q.parallel_for(sycl::nd_range<2>{gS, lS}, [=](sycl::nd_item<2> item) {
                const auto e = static_cast<std::uint32_t>(item.get_global_id(0));
                const auto j = static_cast<std::uint32_t>(item.get_global_id(1));
                if (j >= items) {
                    return;
                }
                std::uint64_t *row = out + e * B;
                const auto s0 = static_cast<std::uint32_t>(j * per_item);
                if (always[e]) {
                    for (std::size_t i = 0; i < per_item; ++i) {
                        row[s0 + i] = ~0ull;
                    }
                    return;
                }
                lab::sample_bitsliced_simd64_persistent<W>(row, pdag[e], e, s0, K32, it, Tdev[e]);
            });
        };
    };
    launches["vp8"] = simdp(std::integral_constant<int, 8>{});
    launches["vp16"] = simdp(std::integral_constant<int, 16>{});
    // The production forms (src/mc/prng/bernoulli/) in the 1-D layout of kernel::sampler::geometry (events x work-groups
    // per event) with the production stream argument. pstrided gives each work-item a run as kernel::sampler::strided
    // does; plockstep runs the lockstep form 16 bitpacks per work-item in work-groups of 64, the geometry of the
    // pre-split CPU kernel, which is not kernel::sampler::lockstep's (chunks of up to kMaxChunk, work-groups of 4).
    const stream_argument stream_arg{prng::expanded_stream::of(kStream)};
    auto production = [&](auto form) -> Launch {
        constexpr bool lockstep_form = decltype(form)::value;
        const std::size_t units = std::max<std::size_t>(dev.get_info<sycl::info::device::max_compute_units>(), 1);
        const std::size_t wg_p = lockstep_form ? 64 : 128;
        std::uint32_t items;
        if (!lockstep_form) {
            const auto run = static_cast<std::uint32_t>(std::clamp<std::size_t>(E * B / (units * 2048), 1, 32));
            items = static_cast<std::uint32_t>((B + run - 1) / run);
        } else {
            items = static_cast<std::uint32_t>((B + 15) / 16);
        }
        const auto padded = static_cast<std::uint32_t>((items + wg_p - 1) / wg_p * wg_p);
        const auto B32 = static_cast<std::uint32_t>(B);
        std::printf("production %s: items/event=%u padded=%u\n", lockstep_form ? "lockstep" : "strided", items, padded);
        return [=, &q](std::uint32_t pass) {
            const auto groups = static_cast<std::uint32_t>(padded / wg_p);
            return q.parallel_for(sycl::nd_range<1>{E * padded, wg_p}, [=](sycl::nd_item<1> item) {
                // Event from the work-group index, so the compiler sees it uniform across the group.
                const auto group = static_cast<std::uint32_t>(item.get_group(0));
                const auto e = group / groups;
                const auto t = static_cast<std::uint32_t>((group - e * groups) * wg_p + item.get_local_id(0));
                if (t >= items) {
                    return;
                }
                std::uint64_t *row = out + e * B;
                if (always[e] || Tdev[e] == 0) {
                    return; // not in this benchmark's p sets except mix; skip
                }
                const prng::expanded_stream run = stream_arg;
                const prng::bernoulli::pass_stream source(run, pdag[e], std::uint64_t{pass} * B32);
                if constexpr (!lockstep_form) {
                    prng::bernoulli::sample_strided(row, t, items, B32, source, Tdev[e]);
                } else {
                    const std::uint32_t first = t * 16;
                    const std::uint32_t valid = std::min<std::uint32_t>(16, B32 - first);
                    prng::bernoulli::sample_lockstep<16>(row, first, valid, source, Tdev[e]);
                }
            });
        };
    };
    launches["pstrided"] = production(std::false_type{});
    launches["plockstep"] = production(std::true_type{});
    launches["simd4"] = simd(std::integral_constant<int, 4>{});
    launches["simd8"] = simd(std::integral_constant<int, 8>{});
    launches["simd16"] = simd(std::integral_constant<int, 16>{});

    // ---------------------------------------------------------------- run
    std::vector<std::uint64_t> host(E * B);
    std::vector<std::uint64_t> ref_bs;              // first bit-sliced variant of samplers.h, iteration 0
    std::vector<std::uint64_t> ref_prod;            // first 32-bit sampler (prod, prod16), iteration 0
    std::vector<std::uint64_t> ref_production_form; // first production form (pstrided, plockstep), pass 0
    for (const auto &name : o.variants) {
        auto itl = launches.find(name);
        if (itl == launches.end()) {
            std::printf("unknown variant %s\n", name.c_str());
            continue;
        }
        if (B % 16 != 0 && (name.rfind("simd", 0) == 0 || name.rfind("v64", 0) == 0 || name.rfind("vp", 0) == 0)) {
            std::printf("%s needs bitpacks %% 16 == 0\n", name.c_str());
            continue;
        }
        auto &launch = itl->second;
        for (int w = 0; w < o.warm; ++w) {
            launch(1000 + w).wait();
        }
        std::vector<double> ms;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < o.reps; ++r) {
            sycl::event e = launch(static_cast<std::uint32_t>(r + 1));
            e.wait();
            const auto s = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            const auto f = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            ms.push_back(static_cast<double>(f - s) * 1e-6);
        }
        const auto t1 = std::chrono::steady_clock::now();
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        const double wall = std::chrono::duration<double, std::milli>(t1 - t0).count() / o.reps;
        const double ns_per_bp = med * 1e6 / static_cast<double>(E * B);

        std::string check;
        if (o.verify) {
            launch(0).wait();
            q.memcpy(host.data(), out, E * B * 8).wait();
            const bool is_prod = name.rfind("prod", 0) == 0;
            // The production forms' bitpacks are at prng::counter, the other variants' at lab::counter.
            const bool is_production_form = name == "pstrided" || name == "plockstep";
            auto &ref = is_prod ? ref_prod : is_production_form ? ref_production_form : ref_bs;
            if (ref.empty()) {
                ref = host;
                check = "ref";
            } else {
                std::size_t diff = 0;
                for (std::size_t i = 0; i < E * B; ++i) {
                    diff += host[i] != ref[i];
                }
                check = diff == 0 ? "identical" : ("DIFF=" + std::to_string(diff));
            }
            // Host reference on a sample of bitpacks.
            std::size_t bad = 0, checked = 0;
            for (std::size_t e = 0; e < E; e += std::max<std::size_t>(1, E / 16)) {
                for (std::size_t s = 0; s < B; s += std::max<std::size_t>(1, B / 64)) {
                    const auto s32 = static_cast<std::uint32_t>(s);
                    std::uint64_t want;
                    if (is_production_form) {
                        // Bitpack s of pass 0, which launch(0) sampled, from the production reference.
                        // lab::bs_reference at the same counter, written independently, must agree.
                        const prng::bernoulli::pass_stream source(kStream, h_pdag[e], 0);
                        want = h_always[e] ? ~0ull : prng::bernoulli::sample(source, s32, h_T[e]);
                        const prng::state128 c = source.counter_of(s32);
                        const lab::state128 lc{{c.x[0], c.x[1], c.x[2], c.x[3]}};
                        if (!h_always[e] && lab::bs_reference(lc, h_T[e]) != want) {
                            ++bad;
                        }
                    } else {
                        const lab::state128 base = lab::counter(h_pdag[e], static_cast<std::uint32_t>(e), s32, 0);
                        if (is_prod) {
                            want = lab::sample_production<false>(base, h_thr32[e]);
                        } else {
                            want = h_always[e] ? ~0ull : lab::bs_reference(base, h_T[e]);
                            // The production reference, called from host code, must agree.
                            const prng::state128 pb{{base.x[0], base.x[1], base.x[2], base.x[3]}};
                            if (!h_always[e] && prng::bernoulli::sample(pb, kStream.key, h_T[e]) != want) {
                                ++bad;
                            }
                        }
                    }
                    bad += want != host[e * B + s];
                    ++checked;
                }
            }
            check += " host_ref_bad=" + std::to_string(bad) + "/" + std::to_string(checked);
            // Frequency z-score per event (vs threshold / 2^bits).
            double zmax = 0;
            for (std::size_t e = 0; e < E; ++e) {
                std::uint64_t ones = 0;
                for (std::size_t s = 0; s < B; ++s) {
                    ones += static_cast<std::uint64_t>(std::popcount(host[e * B + s]));
                }
                const double n = 64.0 * static_cast<double>(B);
                double pe;
                if (h_always[e]) {
                    pe = 1.0;
                } else if (is_prod) {
                    pe = std::min(1.0, static_cast<double>(h_thr32[e]) / 4294967296.0);
                } else {
                    pe = std::ldexp(static_cast<double>(h_T[e]), -64);
                }
                const double sd = std::sqrt(n * pe * (1 - pe));
                if (sd > 0) {
                    zmax = std::max(zmax, std::fabs((static_cast<double>(ones) - n * pe) / sd));
                } else if (static_cast<double>(ones) != n * pe) {
                    zmax = 1e9;
                }
            }
            char buf[64];
            std::snprintf(buf, sizeof buf, " max|z|=%.2f", zmax);
            check += buf;
        }
        std::printf("%-8s kernel_med=%9.3f ms  min=%9.3f  max=%9.3f  wall=%9.3f ms  %7.3f ns/bitpack  %s\n", name.c_str(), med, ms.front(), ms.back(), wall,
                    ns_per_bp, check.c_str());
        std::fflush(stdout);
    }
    sycl::free(out, q);
    sycl::free(ev.pdag, q);
    sycl::free(ev.thr32, q);
    sycl::free(ev.T, q);
    sycl::free(ev.always, q);
    return 0;
}
