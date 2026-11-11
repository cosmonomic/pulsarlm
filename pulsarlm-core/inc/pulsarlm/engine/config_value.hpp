#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace pulsarlm {

struct config_value;
using config_list = std::vector<config_value>;
using config_dict = std::map<std::string, config_value, std::less<>>;

// A value of a JSON config such as a checkpoint's config.json.
struct config_value {
    std::variant<std::nullptr_t, bool, std::int64_t, double, std::string, config_list, config_dict> value;
};

}  // namespace pulsarlm
