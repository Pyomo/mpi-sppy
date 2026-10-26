// utils/timer.cpp
#include "timer.hpp"
#include <mutex>

namespace {
    const GlobalTimer* g_timer = nullptr;
    std::mutex g_mutex;
}

void install_global_timer(const GlobalTimer* t) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_timer = t;
}

void global_toc(std::string_view msg) {
    const GlobalTimer* local = nullptr;
    {   // tiny critical section
        std::lock_guard<std::mutex> lock(g_mutex);
        local = g_timer;
    }
    if (local) {
        local->global_toc(msg);
    } else {
        // Fallback: avoid UB if someone calls before install
        static GlobalTimer fallback;
        fallback.global_toc(msg);
    }
}

// New: get wall-clock seconds from the same global timer
double global_seconds() {
    const GlobalTimer* local = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        local = g_timer;
    }
    if (local) {
        return local->seconds();
    } else {
        static GlobalTimer fallback;
        return fallback.seconds();
    }
}
