#pragma once

#include "pulsarlm/ops.hpp"
#include "pulsarlm/profiling.hpp"
#include "pulsarlm/rope.hpp"

#include "pulsarlm/model/compiled_model.hpp"
#include "pulsarlm/model/interface.hpp"
#include "pulsarlm/model/paged_forward.hpp"
#include "pulsarlm/model/qwen2.hpp"
#include "pulsarlm/model/qwen3.hpp"

#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine.hpp"
#include "pulsarlm/event_loop.hpp"
#include "pulsarlm/eviction.hpp"
#include "pulsarlm/kv_cache.hpp"
#include "pulsarlm/request_callbacks.hpp"
#include "pulsarlm/sampling.hpp"
#include "pulsarlm/scheduler.hpp"

// Umbrella header for the engine's public C++ API: the engine and the components
// it is built from, the model definitions (model/), and the op declarations, RoPE
// math and profiling macros they're built on. Excludes kernels/: those are .cuh,
// device code compiled only from .cu translation units, not headers a host-side
// engine consumer includes.
