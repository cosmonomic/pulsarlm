#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace pulsarlm {

struct digest {
    std::uint64_t hi;
    std::uint64_t lo;

    friend bool operator==(const digest& a, const digest& b) noexcept = default;
};

struct digest_hash {
    std::size_t operator()(const digest& d) const noexcept {
        return d.lo;
    }
};

digest make_digest(std::span<const std::byte> seed) noexcept;
digest make_digest(const digest& parent, std::span<const std::int64_t> tokens) noexcept;

// The digest of each full page of tokens, chained from parent.
std::vector<digest>
make_page_digests(const digest& parent, std::span<const std::int64_t> tokens, std::size_t page_size);

// The hash chosen at build time: "sha256" or "blake3".
constexpr std::string_view digest_function() noexcept {
    return PULSARLM_DIGEST_FUNCTION;
}

}  // namespace pulsarlm
