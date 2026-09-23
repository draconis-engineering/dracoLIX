#pragma once
// Memory profiling — Phase 3 final
// Licensed under GPL-3.0-only
#include <cstddef>
#include <string>
#include <fstream>
#include <sstream>

namespace dracolix::mem {

inline size_t parse_proc_status(const char* key) {
#if defined(__linux__)
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind(key, 0) == 0) {
            std::istringstream iss(line);
            std::string k; size_t val; std::string unit;
            iss >> k >> val >> unit;
            // VmRSS is in kB
            return val * 1024;
        }
    }
#endif
    return 0;
}

inline size_t current_rss_bytes() {
#if defined(__linux__)
    return parse_proc_status("VmRSS:");
#elif defined(_WIN32)
    // fallback: no Windows API here to keep header-only without <windows.h>
    return 0;
#elif defined(__APPLE__)
    return 0;
#else
    return 0;
#endif
}

inline size_t peak_rss_bytes() {
#if defined(__linux__)
    return parse_proc_status("VmHWM:");
#else
    return 0;
#endif
}

inline std::string format_bytes(size_t b) {
    const char* units[] = {"B","KB","MB","GB"};
    double v = (double)b;
    int u = 0;
    while (v >= 1024 && u < 3) { v /= 1024; ++u; }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.1f %s", v, units[u]);
    return std::string(buf);
}

} // namespace dracolix::mem
