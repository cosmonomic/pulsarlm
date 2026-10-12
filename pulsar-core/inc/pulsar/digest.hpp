#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace pulsar {

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

// The root input is seed; the page input is parent.hi, parent.lo, then tokens, each as 8 little-endian bytes. The
// domain is "pulsar root digest" or "pulsar page digest". SHA-256 hashes domain followed by input; BLAKE3 hashes input
// in derive-key mode with domain as context. The digest is the first 16 output bytes read as hi, then lo, each
// little-endian. Every service computing digests must match this encoding and the hash chosen at build time.
digest make_digest(std::span<const std::byte> seed) noexcept;
digest make_digest(const digest& parent, std::span<const std::int64_t> tokens) noexcept;

// The digest of each full page of tokens, chained from parent: the first page from parent, each next page from the
// page before it. tokens start at a page boundary; the partial tail page is not hashed.
std::vector<digest>
make_page_digests(const digest& parent, std::span<const std::int64_t> tokens, std::size_t page_size);

// The hash chosen at build time: "sha256" or "blake3".
std::string_view digest_function() noexcept;

}  // namespace pulsar
