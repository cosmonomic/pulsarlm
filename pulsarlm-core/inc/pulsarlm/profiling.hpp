#pragma once

// Profiling turned off by default; enable with CMake PULSARLM_PROF=ON.
//
// PULSARLM_PROF_EXPR forwards its expression's value while timing it. A
// summary also prints to stderr at process exit.
//
// These are host wall-clock timers, not GPU timers: CUDA launches
// asynchronously, so the phase containing the first blocking call (a D2H
// copy, .item(), a synchronize) absorbs all device time launched before it.

#ifdef PULSARLM_PROF

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>

namespace pulsarlm::prof {

enum Phase {
    STEP = 0,
    RECALL_CYCLE,
    EVICT,
    ONEVICT,
    MLC_RECALL,
    RERANK,
    RERANK_SCORE,
    RERANK_STAGE,
    RERANK_BMM,
    RECOMPACT_GPU,
    NPHASE
};

struct Acc {
    std::atomic<uint64_t> ns{0};
    std::atomic<uint64_t> calls{0};
};

inline Acc& acc(int i) {
    static Acc a[NPHASE];
    return a[i];
}

struct Timer {
    int idx;
    std::chrono::steady_clock::time_point t0;
    explicit Timer(int i)
        : idx(i),
          t0(std::chrono::steady_clock::now()) {}
    ~Timer() {
        auto d = std::chrono::steady_clock::now() - t0;
        acc(idx).ns += std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
        acc(idx).calls++;
    }
};

inline const char* name(int i) {
    static const char* n[NPHASE] = {
        "STEP",
        "RECALL_CYCLE",
        "EVICT",
        "ONEVICT",
        "MLC_RECALL",
        "RERANK",
        "RERANK_SCORE",
        "RERANK_STAGE",
        "RERANK_BMM",
        "RECOMPACT_GPU"
    };
    return n[i];
}

inline void dump(const char* tag) {
    std::fprintf(stderr, "\n=== PULSARLM_PROF %s ===\n", tag ? tag : "");
    std::fprintf(stderr, "%-16s %12s %10s %12s\n", "phase", "total_ms", "calls", "us/call");
    for (int i = 0; i < NPHASE; ++i) {
        uint64_t ns = acc(i).ns.load();
        uint64_t c = acc(i).calls.load();
        double ms = ns / 1e6;
        double uspc = c ? (ns / 1e3) / static_cast<double>(c) : 0.0;
        std::fprintf(stderr, "%-16s %12.2f %10llu %12.2f\n", name(i), ms, static_cast<unsigned long long>(c), uspc);
    }
    std::fprintf(stderr, "=====================\n");
}

// Single instance across all translation units (inline variable). Its destructor
// runs at process exit and prints the accumulated summary.
struct AtExitDump {
    ~AtExitDump() {
        dump("atexit");
    }
};
inline AtExitDump g_atexit_dump;

}  // namespace pulsarlm::prof

#define PULSARLM_PROF_CAT2(a, b) a##b
#define PULSARLM_PROF_CAT(a, b) PULSARLM_PROF_CAT2(a, b)
#define PULSARLM_PROF_SCOPE(phase) \
    ::pulsarlm::prof::Timer PULSARLM_PROF_CAT(pulsarlm_prof_timer_, __LINE__)(::pulsarlm::prof::phase)
#define PULSARLM_PROF_EXPR(phase, ...) \
    ([&]() -> decltype(auto) { \
        ::pulsarlm::prof::Timer PULSARLM_PROF_CAT(pulsarlm_prof_timer_, __LINE__)(::pulsarlm::prof::phase); \
        return (__VA_ARGS__); \
    }())
#define PULSARLM_PROF_DUMP(tag) ::pulsarlm::prof::dump(tag)

#else  // PULSARLM_PROF not defined: complete no-ops.

#define PULSARLM_PROF_SCOPE(phase) ((void)0)
#define PULSARLM_PROF_EXPR(phase, ...) (__VA_ARGS__)
#define PULSARLM_PROF_DUMP(tag) ((void)0)

#endif  // PULSARLM_PROF
