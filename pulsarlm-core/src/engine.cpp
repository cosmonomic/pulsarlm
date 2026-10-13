#include "pulsarlm/engine.hpp"

#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <stdexcept>

namespace pulsarlm {

namespace {

void check_tokens(std::span<const std::int64_t> tokens, std::size_t vocab_size) {
    for (std::int64_t token : tokens) {
        if (token < 0 || static_cast<std::uint64_t>(token) >= vocab_size) {
            throw std::invalid_argument(std::format("token id {} is outside [0, {})", token, vocab_size));
        }
    }
}

}  // namespace

std::size_t line_count(const engine_config& config) {
    std::size_t bytes = line_bytes(config.shape);
    if (config.kv_cache_bytes < bytes) {
        throw std::invalid_argument(
            std::format("a KV cache of {} bytes holds no line; a line takes {} bytes", config.kv_cache_bytes, bytes)
        );
    }
    return config.kv_cache_bytes / bytes;
}

void validate(const request& r, std::size_t page_size, std::size_t vocab_size) {
    if (r.initial_state.tail.size() >= page_size) {
        throw std::invalid_argument(std::format(
            "the initial state's tail holds {} tokens; it must be shorter than a page ({})",
            r.initial_state.tail.size(),
            page_size
        ));
    }
    if (r.initial_state.tail.empty() && r.tokens.empty()) {
        throw std::invalid_argument("the request has no token to forward");
    }
    check_tokens(r.initial_state.tail, vocab_size);
    check_tokens(r.tokens, vocab_size);
    if (r.eviction.n_working < 1) {
        throw std::invalid_argument("eviction.n_working must be at least 1");
    }
    if (r.eviction.cycle_tokens < 1) {
        throw std::invalid_argument("eviction.cycle_tokens must be at least 1");
    }
    if (!(r.eviction.attention_mass_decay > 0.0 && r.eviction.attention_mass_decay <= 1.0)) {
        throw std::invalid_argument(
            std::format("eviction.attention_mass_decay {} is outside (0, 1]", r.eviction.attention_mass_decay)
        );
    }
}

}  // namespace pulsarlm
