#include "pulsarlm/engine/request_callbacks.hpp"

#include <expected>
#include <memory>
#include <utility>

namespace pulsarlm {

void end_with_loop_exited(std::unique_ptr<request_callbacks>& callbacks) noexcept {
    std::unique_ptr<request_callbacks> ended = std::move(callbacks);
    ended->on_end(std::unexpected(request_error::loop_exited), request_statistics{});
}

}  // namespace pulsarlm
