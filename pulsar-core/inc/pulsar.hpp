#pragma once

#include "pulsar/ops.hpp"
#include "pulsar/profiling.hpp"
#include "pulsar/rope.hpp"

#include "pulsar/model/compiled_model.hpp"
#include "pulsar/model/interface.hpp"
#include "pulsar/model/paged_forward.hpp"
#include "pulsar/model/qwen2.hpp"
#include "pulsar/model/qwen3.hpp"

#include "pulsar/digest.hpp"
#include "pulsar/engine.hpp"
#include "pulsar/event_loop.hpp"
#include "pulsar/eviction.hpp"
#include "pulsar/kv_cache.hpp"
#include "pulsar/request_callbacks.hpp"
#include "pulsar/sampling.hpp"
#include "pulsar/scheduler.hpp"

// Umbrella header for the engine's public C++ API: the engine and the components
// it is built from, the model definitions (model/), and the op declarations, RoPE
// math and profiling macros they're built on. Excludes kernels/: those are .cuh,
// device code compiled only from .cu translation units, not headers a host-side
// engine consumer includes.
