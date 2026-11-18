#include "pulsarlm/digest.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace pulsarlm {

std::vector<digest>
make_page_digests(const digest& parent, std::span<const std::int64_t> tokens, std::size_t page_size) {
    if (page_size == 0) {
        throw std::invalid_argument("make_page_digests takes a page_size above 0");
    }
    std::vector<digest> digests;
    digests.reserve(tokens.size() / page_size);
    digest previous = parent;
    for (std::size_t begin = 0; tokens.size() - begin >= page_size; begin += page_size) {
        previous = make_digest(previous, tokens.subspan(begin, page_size));
        digests.push_back(previous);
    }
    return digests;
}

}  // namespace pulsarlm
