// Names one MC kernel at a bitpack width it does not support. Built only by the mc.compile_fail.* tests, which pass
// iff compilation fails and every diagnostic is the expected one. The kernels take their bitpack type from
// event::node_types, whose bitpack_word constraint (event/node.h) refuses a word wider than 64 bits, so the refusal
// comes before any kernel is instantiated. Configure with FZB_BITS (128 | 256) and exactly one of
// FZB_KERNEL_CARDINALITY / FZB_KERNEL_TALLY.

#include "mc/event/node.h"
#include "mc/kernel/cardinality.h"
#include "mc/kernel/tally.h"

#include <cstdint>

namespace {

// Clang caps _BitInt at 128 bits on AArch64, so 256 bits is a struct of limbs: a 256-bit word that is no integer
// type.
struct uint256 {
    std::uint64_t limb[4] = {0, 0, 0, 0};
};

} // namespace

#if FZB_BITS == 128
using bitpack_t = unsigned __int128;
#elif FZB_BITS == 256
using bitpack_t = uint256;
#else
#error "FZB_BITS must be 128 or 256"
#endif

static_assert(sizeof(bitpack_t) * 8 == FZB_BITS);

#if defined(FZB_KERNEL_CARDINALITY)
using kernel_t = fuzzybits::mc::kernel::cardinality<fuzzybits::mc::event::node_types<bitpack_t, std::int32_t, std::uint32_t>,
                                                    fuzzybits::mc::kernel::cardinality_form::fixed, 4, false>;
#elif defined(FZB_KERNEL_TALLY)
using kernel_t = fuzzybits::mc::kernel::tally<fuzzybits::mc::event::node_types<bitpack_t, std::int32_t, std::uint32_t>>;
#else
#error "define FZB_KERNEL_CARDINALITY or FZB_KERNEL_TALLY"
#endif
