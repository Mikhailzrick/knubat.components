// BatteryPlus — battery percentage daemon for handheld Linux systems
// Version 1.2.1
// Copyright (c) 2025 Mikhailzrick
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2
// as published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License v2
// along with this program. If not, see https://www.gnu.org/licenses/.
//
//
// Purpose:
//   BatteryPlus provides stable battery UI behavior for handheld Linux systems.
//   It can use voltage-based estimation or PMIC capacity while still providing
//   step limiting, hooks, restore handling, and charge-state events.
//
// Modes:
//   • voltage
//       - Calculates percentage from V_EMPTY and the learned V_FULL_DIS anchor
//       - Uses median filtering and EMA smoothing
//       - Compensates charging voltage before applying the discharge curve
//       - Learns charging inflation after plug-in and tapers it to the full-anchor gap
//       - Holds charging at 99% until full is reported or inferred
//
//   • pmic
//       - Uses the PMIC-reported capacity directly
//       - Reuses the normal UI, hooks, state handling, and update timing
//
// Full calibration:
//   - V_FULL_CHG is the observed charging peak
//   - V_FULL_DIS is learned from the settled voltage after unplugging from full
//   - Full is accepted from PMIC Full, a stable voltage dwell, or a confirmed peak drop
//   - Every new filtered charging peak restarts the 45-minute dwell
//   - Voltage can drop from its charging peak when charging finishes.
//     Treat raw and filtered voltage staying at least 10 mV below that peak
//     for 120 seconds as a fallback full signal when PMIC Full isn't reported.
//   - Post-unplug voltage is sampled adaptively for up to 60 seconds
//   - One valid pair is saved exactly, then automatic calibration is locked
//   - Removing batteryplus-calibrated and restarting allows a new calibration cycle
//
// Restore and resume:
//   - Clean shutdown saves charge state, voltage, and visible percentage
//   - Startup and resume use voltage windows to restore, blend, or recalculate
//   - Resume waits for charge state confirmation before reconciliation
//   - Established charging compensation survives Charging -> Charging resume
//
// Visible percentage:
//   - Exported through /tmp/battery.percent
//   - Normal updates never count down while charging or up while discharging
//   - Normal changes are limited to one percentage point per update
//   - Larger internal/UI gaps use shorter update intervals
//   - Charging doubles the adaptive interval timing
//   - Reported or inferred Full restores normal timing
//
// Hooks:
//   /etc/batteryplus/charging.d/
//   /etc/batteryplus/discharging.d/
//   /etc/batteryplus/state.d/
//
//   Charging/discharging hooks are launched on visible percentage changes,
//   charge-direction changes, startup, and resume. Charging hooks are launched
//   in reverse filename order; discharging hooks in filename order.
//   Hooks run asynchronously, so completion order is not guaranteed.
//   State hooks are launched at startup and on confirmed charge-state changes.
//   Hooks receive visible percentage as $1 and charge state as $2.
//
// Persistent files:
//   <data_dir>/batteryplus-voltage.map     Learned full-voltage anchors
//   <data_dir>/batteryplus-calibrated      Calibration lock flag
//   <data_dir>/batteryplus-restore.state   One-shot startup restore state
//   <data_dir>/batteryplus-debug.log       Debug log when debug=true
//
// Config:
//   /etc/batteryplus/batteryplus.conf
//
//   mode=voltage|pmic
//   data_dir=<absolute persistent directory>
//   V_EMPTY=<mV>
//   debug=true|false
//
//   <data_dir>/batteryplus.conf may override settings except data_dir.
//   If data_dir is empty, the normal XDG/HOME config location is used.
//
// Signals:
//   SIGTERM / SIGINT   Stop cleanly and save restore state
//   SIGUSR1            Ignored
//
// Build:
//
//   aarch64 cross-build:
//       aarch64-linux-gnu-g++ -O2 -flto=auto -ffunction-sections -fdata-sections -std=gnu++20 -Wall -Wextra -Wpedantic batteryplus.cpp -Wl,--gc-sections,-s -o batteryplus
//
//   native x86-64:
//       g++ -O2 -flto=auto -ffunction-sections -fdata-sections -std=gnu++20 -Wall -Wextra -Wpedantic batteryplus.cpp -Wl,--gc-sections,-s -o batteryplus

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

// ========================= Config (constants) =========================
static constexpr const char* MAP_FILENAME = "batteryplus-voltage.map";
static constexpr const char* RESTORE_STATE_FILENAME = "batteryplus-restore.state";
static constexpr const char* OVERRIDE_CONFIG_FILENAME = "batteryplus.conf";
static constexpr const char* CONFIG_FILE = "/etc/batteryplus/batteryplus.conf";
static constexpr const char* PERCENT_FILE = "/tmp/battery.percent";
static constexpr const char* STATUS_FILE = "/tmp/battery.status";
static constexpr const char* DEBUG_LOG_FILENAME = "batteryplus-debug.log";
static constexpr const char* ROOT = "/etc/batteryplus";

// Update timing
static constexpr int INTERNAL_INTERVAL_S = 15;
static constexpr int WRITE_INTERVAL_S = 120;
static constexpr int WRITE_INTERVAL_DELTA_SMALL_S = 60;
static constexpr int WRITE_INTERVAL_DELTA_LARGE_S = 30;
static constexpr int CHARGE_VISIBLE_INTERVAL_MULTIPLIER = 2;
static constexpr int RESUME_SUSPEND_GAP_S = 2;

// Charging
static constexpr int CHARGE_COMP_REFINE_WINDOW_S = 30;
static constexpr int CHARGE_COMP_MAX_INFLATION_MV = 300;

// Full learning
static constexpr int PEAK_DWELL_S = 45 * 60;
static constexpr int PEAK_TRACK_RANGE_MV = 50;
static constexpr int PEAK_FULL_DROP_MV = 10;
static constexpr int PEAK_FULL_DROP_CONFIRM_S = 120;
static constexpr int VFULL_DIS_INITIAL_SETTLE_S = 8;
static constexpr int VFULL_DIS_CHECK_S = 8;
static constexpr int VFULL_DIS_STABLE_MV = 5;
static constexpr int VFULL_DIS_STABLE_COUNT = 2;
static constexpr int VFULL_DIS_MAX_WAIT_S = 60;

// A new charge state has to remain stable this long before we accept it.
static constexpr int STATUS_CONFIRM_S = 1;

// Voltage mapping
static constexpr double DISCHARGE_100_WINDOW_PERCENT = 2.0;
static constexpr int MIN_RANGE_MV = 500;

struct RestoreWindow {
    int down_mv;
    int up_mv;
};

// Allowed voltage movement for each saved -> current charge-state transition.
static constexpr RestoreWindow RESTORE_DIS_TO_DIS = { 50, 35 };
static constexpr RestoreWindow RESTORE_CHG_TO_CHG = { 35, 75 };
static constexpr RestoreWindow RESTORE_DIS_TO_CHG = { 35, 300 };
static constexpr RestoreWindow RESTORE_CHG_TO_DIS = { 300, 35 };

// Blend outward until this multiple of the exact-restore window.
static constexpr double RESTORE_BLEND_SCALE = 1.5;

// Small startup correction for discharge load immediately after boot.
static constexpr int BOOT_COMP_MAX_UPTIME_S = 60;
static constexpr int BOOT_COMP_MV = 25;

// Near empty, prefer a fresh startup estimate over a potentially stale restore.
static constexpr int BOOT_EMPTY_FRESH_MARGIN_MV = 50;

// EMA alpha = 0.2
static constexpr int ALPHA_NUM = 2;
static constexpr int ALPHA_DEN = 10;

// Used until a valid learned pair exists.
static constexpr int DEFAULT_V_FULL_CHG = 4050;
static constexpr int DEFAULT_V_FULL_DIS = 4000;
static constexpr int DEFAULT_V_EMPTY = 3250;

// Mode parameters
enum class BatteryMode {
    Voltage,
    Pmic
};

// ========================= States =========================
struct ConfigVals {
    BatteryMode mode = BatteryMode::Voltage;
    fs::path data_dir;
    int V_EMPTY = DEFAULT_V_EMPTY;
    bool debug = false;
};

// Calibration is enabled while the lock flag is absent.
struct CalibrationState {
    bool enabled = true;
};

// Written on clean exit and consumed once at the next startup.
struct RestoreState {
    bool charging = false;
    int saved_mv = -1;
    int visible_percent = -1;
};

struct StartupInitResult {
    bool restore_charging = false;
    bool force_visible_write = false;
};

struct StatePaths {
    fs::path data_dir;
    fs::path map_file;
    fs::path calibrated_flag;
    fs::path restore_state_file;
    fs::path debug_log_file;
};

// ========================= Globals =========================
static std::atomic<bool> g_running { true };
static StatePaths g_paths;
static ConfigVals g_cfg;

// ========================= Utilities =========================
static void handle_signal(int) {
    g_running = false;
}

static bool sleep_interruptible_s(int seconds) {
    for (int i = 0; i < seconds; ++i) {
        if (!g_running) {
            return false;
        }

        std::this_thread::sleep_for(1s);
    }

    return g_running;
}

static int64_t boottime_s() {
    struct timespec ts{};
    if (::clock_gettime(CLOCK_BOOTTIME, &ts) != 0) {
        return 0;
    }
    return static_cast<int64_t>(ts.tv_sec);
}

static int64_t clock_ms(clockid_t clock_id) {
    struct timespec ts{};

    if (::clock_gettime(clock_id, &ts) != 0) {
        return -1;
    }

    return static_cast<int64_t>(ts.tv_sec) * 1000 +
           static_cast<int64_t>(ts.tv_nsec) / 1000000;
}

static void debug_log(const char* event, const char* format = nullptr, ...) {
    if (!g_cfg.debug || g_paths.debug_log_file.empty()) {
        return;
    }

    std::FILE* log = std::fopen(g_paths.debug_log_file.c_str(), "a");
    if (!log) {
        return;
    }

    std::fprintf(
        log,
        "uptime=%lld,event=%s",
        static_cast<long long>(boottime_s()),
        event
    );

    if (format && *format) {
        std::fputc(',', log);

        va_list args;
        va_start(args, format);
        std::vfprintf(log, format, args);
        va_end(args);
    }

    std::fputc('\n', log);
    std::fclose(log);
}

#define DEBUG_LOG(...) \
    do { \
        if (g_cfg.debug) { \
            debug_log(__VA_ARGS__); \
        } \
    } while (0)

static std::optional<std::string> slurp(const fs::path& p) {
    std::ifstream f(p);
    if (!f) return std::nullopt;
    std::string s;
    std::getline(f, s);
    if (!f && !f.eof()) return std::nullopt;
    // Trim trailing whitespace.
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
    return s;
}

static std::optional<int> slurp_int(const fs::path& p) {
    auto s = slurp(p);
    if (!s) return std::nullopt;
    char* end = nullptr;
    long v = std::strtol(s->c_str(), &end, 10);
    if (end == s->c_str()) return std::nullopt;
    return static_cast<int>(v);
}

static bool write_atomic(const fs::path& path, const std::string& data, mode_t mode = 0644) {
    fs::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) return false;
        f << data;
        f.close();
        if (!f.good()) return false;
    }
    ::chmod(tmp.c_str(), mode);
    return ::rename(tmp.c_str(), path.c_str()) == 0;
}

static bool save_restore_state(const RestoreState& st) {
    std::string data;
    data += "charging=" + std::to_string(st.charging ? 1 : 0) + "\n";
    data += "saved_mv=" + std::to_string(st.saved_mv) + "\n";
    data += "visible_percent=" + std::to_string(st.visible_percent) + "\n";
    return write_atomic(g_paths.restore_state_file, data, 0644);
}

static std::optional<RestoreState> consume_restore_state() {
    RestoreState st;
    bool saw_charging = false;
    bool saw_saved = false;
    bool saw_visible = false;

    std::ifstream f(g_paths.restore_state_file);
    if (f) {
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("charging=", 0) == 0) {
                st.charging = (std::atoi(line.c_str() + 9) != 0);
                saw_charging = true;
            } else if (line.rfind("saved_mv=", 0) == 0) {
                st.saved_mv = std::atoi(line.c_str() + 9);
                saw_saved = true;
            } else if (line.rfind("visible_percent=", 0) == 0) {
                st.visible_percent = std::atoi(line.c_str() + 16);
                saw_visible = true;
            }
        }
    }

    // Always consume the file, even if the saved state is invalid.
    std::error_code ec;
    fs::remove(g_paths.restore_state_file, ec);

    if (!(saw_charging && saw_saved && saw_visible)) {
        return std::nullopt;
    }

    return st;
}

static int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static void to_lower_inplace(char* s) {
    for (; *s; ++s) {
        *s = static_cast<char>(std::tolower((unsigned char)*s));
    }
}

static std::optional<bool> parse_bool(char* value) {
    to_lower_inplace(value);

    if (std::strcmp(value, "true") == 0) {
        return true;
    }

    if (std::strcmp(value, "false") == 0) {
        return false;
    }

    return std::nullopt;
}

static int median3(int a, int b, int c) {
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
    return b;
}

static int median_values(const int* values, int count) {
    if (!values || count <= 0) {
        return -1;
    }

    std::vector<int> sorted(values, values + count);
    std::sort(sorted.begin(), sorted.end());

    if ((count % 2) != 0) {
        return sorted[count / 2];
    }

    return static_cast<int>(
        (static_cast<int64_t>(sorted[count / 2 - 1]) +
         sorted[count / 2] + 1) / 2
    );
}

template <std::size_t N>
static int median(std::array<int, N> samples) {
    static_assert(N > 0, "At least one sample is required");

    std::sort(samples.begin(), samples.end());

    if constexpr (N % 2 != 0) {
        return samples[N / 2];
    } else {
        return static_cast<int>(
            (static_cast<int64_t>(samples[N / 2 - 1]) +
             samples[N / 2] + 1) / 2
        );
    }
}

enum class ChargeStatus {
    Unknown,
    Charging,
    Discharging,
    Full
};

static bool is_charging_status(ChargeStatus status) {
    return status == ChargeStatus::Charging ||
           status == ChargeStatus::Full;
}

static ChargeStatus read_charge_status(const fs::path& status_path) {
    auto s = slurp(status_path);
    if (!s) return ChargeStatus::Unknown;

    if (s->rfind("Charging", 0) == 0) return ChargeStatus::Charging;
    if (s->rfind("Discharging", 0) == 0) return ChargeStatus::Discharging;
    if (s->rfind("Full", 0) == 0) return ChargeStatus::Full;
    return ChargeStatus::Unknown;
}
struct ChargeStatusTracker {
    ChargeStatus confirmed = ChargeStatus::Unknown;
    ChargeStatus candidate = ChargeStatus::Unknown;
    std::chrono::steady_clock::time_point candidate_since{};
};

// Debounce charge-state changes before accepting them.
static void update_confirmed_charge_status(
    ChargeStatusTracker& tracker,
    ChargeStatus observed,
    std::chrono::steady_clock::time_point now,
    bool reset_candidate
) {
    if (reset_candidate) {
        tracker.candidate = ChargeStatus::Unknown;
    }

    // Don't let a failed read replace a confirmed state.
    if (observed == ChargeStatus::Unknown) {
        tracker.candidate = ChargeStatus::Unknown;
        return;
    }

    if (observed == tracker.confirmed) {
        tracker.candidate = ChargeStatus::Unknown;
        return;
    }

    if (observed != tracker.candidate) {
        tracker.candidate = observed;
        tracker.candidate_since = now;
        return;
    }

    if (now - tracker.candidate_since < std::chrono::seconds(STATUS_CONFIRM_S)) {
        return;
    }

    tracker.confirmed = tracker.candidate;
    tracker.candidate = ChargeStatus::Unknown;
}

static const char* charge_status_arg(ChargeStatus st) {
    switch (st) {
        case ChargeStatus::Charging: return "Charging";
        case ChargeStatus::Discharging: return "Discharging";
        case ChargeStatus::Full: return "Full";
        default: return "";
    }
}

static bool write_status_file(ChargeStatus status) {
    const char* value = charge_status_arg(status);

    if (!value || !*value)
        return false;

    // Ensure inotify watches work. (no write_atomic)
    std::ofstream f(STATUS_FILE, std::ios::trunc);
    if (!f)
        return false;

    f << value << '\n';
    f.close();

    return f.good();
}

enum class VisiblePolicy {
    Normal,
    Hold,
    Snap
};

struct LoopState {
    ChargeStatus observed_status = ChargeStatus::Unknown;
    ChargeStatus status = ChargeStatus::Unknown;

    bool resume_detected = false;
    bool resume_transition = false;
    bool first_visible = false;
    bool hooks_fired = false;
    bool hold_voltage = false;

    VisiblePolicy visible_policy = VisiblePolicy::Normal;

    bool resume_restore_now = false;

    bool charging_changed = false;
    bool status_changed = false;

    bool recalculate = false;

    std::optional<int> resume_visible_percent;

    bool charging() const {
        return is_charging_status(status);
    }

    bool full() const {
        return status == ChargeStatus::Full;
    }
};

// ========================= Config =========================
static void ensure_config() {
    struct stat st{};
    if (::stat(CONFIG_FILE, &st) == 0) return; // already exists

    fs::create_directories(fs::path(ROOT));

    std::FILE* f = std::fopen(CONFIG_FILE, "w");
    if (!f) return;

    std::fprintf(f,
        "# data_dir should be an absolute persistent directory\n"
        "# if blank BatteryPlus uses $XDG_CONFIG_HOME/batteryplus\n"
        "# or $HOME/.config/batteryplus when XDG_CONFIG_HOME is not valid\n"
        "# <data_dir>/batteryplus.conf may override settings (except data_dir)\n"
        "# possible modes: voltage(default) and pmic\n"
        "[Config]\n"
        "mode=voltage\n"
        "data_dir=\n"
        "V_EMPTY=%d\n"
        "debug=false\n",
        DEFAULT_V_EMPTY
    );
    std::fclose(f);
}

static fs::path resolve_default_data_dir() {
    const char* xdg_config_home = std::getenv("XDG_CONFIG_HOME");

    if (xdg_config_home != nullptr && xdg_config_home[0] != '\0') {
        fs::path xdg_path(xdg_config_home);

        if (xdg_path.is_absolute()) {
            return xdg_path / "batteryplus";
        }
    }

    const char* home = std::getenv("HOME");

    if (home != nullptr && home[0] != '\0') {
        fs::path home_path(home);

        if (home_path.is_absolute()) {
            return home_path / ".config" / "batteryplus";
        }
    }

    return {};
}

static void read_config_file(
    const fs::path& config_path,
    ConfigVals& cfg,
    bool allow_data_dir
) {
    std::FILE* f = std::fopen(config_path.c_str(), "r");
    if (!f) {
        return;
    }

    char line[512];

    while (std::fgets(line, sizeof(line), f)) {
        char* p = line;

        while (*p == ' ' || *p == '\t') {
            ++p;
        }

        if (*p == '\0' || *p == '\n' || *p == '#' || *p == '[') {
            continue;
        }

        char* eq = std::strchr(p, '=');
        if (!eq) {
            continue;
        }

        *eq = '\0';

        char* key = p;
        char* val = eq + 1;

        for (char* t = key + std::strlen(key);
             t > key &&
             (t[-1] == ' ' || t[-1] == '\t' ||
              t[-1] == '\r' || t[-1] == '\n');
             --t) {
            t[-1] = '\0';
        }

        while (*val == ' ' || *val == '\t') {
            ++val;
        }

        for (char* t = val + std::strlen(val);
             t > val &&
             (t[-1] == ' ' || t[-1] == '\t' ||
              t[-1] == '\r' || t[-1] == '\n');
             --t) {
            t[-1] = '\0';
        }

        if (std::strcmp(key, "mode") == 0) {
            to_lower_inplace(val);

            if (std::strcmp(val, "pmic") == 0) {
                cfg.mode = BatteryMode::Pmic;
            } else if (std::strcmp(val, "voltage") == 0) {
                cfg.mode = BatteryMode::Voltage;
            }

        } else if (std::strcmp(key, "data_dir") == 0) {
            if (allow_data_dir) {
                cfg.data_dir = val;
            }

        } else if (std::strcmp(key, "V_EMPTY") == 0) {
            char* end = nullptr;
            long v = std::strtol(val, &end, 10);

            if (end != val && *end == '\0') {
                cfg.V_EMPTY = static_cast<int>(v);
            }
        } else if (std::strcmp(key, "debug") == 0) {
            if (auto parsed = parse_bool(val)) {
                cfg.debug = *parsed;
            }
        }
    }

    std::fclose(f);
}

static void validate_config(
    ConfigVals& cfg,
    const ConfigVals& fallback
) {
    // Invalid values fall back to the previous config layer.
    if (cfg.V_EMPTY < 3000 || cfg.V_EMPTY > 3400) {
        cfg.V_EMPTY = fallback.V_EMPTY;
    }
}

static ConfigVals read_config() {
    ConfigVals cfg;
    const ConfigVals defaults;

    // Load the system config first.
    read_config_file(CONFIG_FILE, cfg, true);
    validate_config(cfg, defaults);

    // data_dir must come from the system config and be absolute.
    if (!cfg.data_dir.empty() && !cfg.data_dir.is_absolute()) {
        cfg.data_dir.clear();
    }

    // Use the normal user config location if data_dir wasn't set.
    if (cfg.data_dir.empty()) {
        cfg.data_dir = resolve_default_data_dir();
    }

    if (cfg.data_dir.empty()) {
        return cfg;
    }

    // The data-dir config can override everything except data_dir.
    const ConfigVals base_cfg = cfg;

    read_config_file(
        cfg.data_dir / OVERRIDE_CONFIG_FILENAME,
        cfg,
        false
    );

    validate_config(cfg, base_cfg);

    return cfg;
}

// ========================= Hook System =========================
static bool is_executable(const fs::directory_entry& de) {
    if (!de.is_regular_file()) return false;
    return ::access(de.path().c_str(), X_OK) == 0;
}

static int run_hook_file(const fs::path& file, int visible_percent, ChargeStatus status) {
    pid_t pid = ::fork();
    if (pid < 0) return -1;

    if (pid == 0) {
        // Silence hook stdio; hooks are fire-and-forget.
        int nullfd = ::open("/dev/null", O_RDWR);
        if (nullfd >= 0) {
            ::dup2(nullfd, STDIN_FILENO);
            ::dup2(nullfd, STDOUT_FILENO);
            ::dup2(nullfd, STDERR_FILENO);
            ::close(nullfd);
        }
        std::string pct = std::to_string(clampi(visible_percent, 0, 100));
        const char* state = charge_status_arg(status);

        const char* argv[] = {
            file.c_str(),
            pct.c_str(),
            state,
            nullptr
        };
        ::execv(argv[0], (char* const*)argv);
        _exit(127);
    }

    return 0;
}

struct HookCache {
    std::vector<fs::path> charging;
    std::vector<fs::path> discharging;
    std::vector<fs::path> state;
    bool loaded = false;
};

static void scan_hook_dir(const fs::path& dir, std::vector<fs::path>& paths) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) return;

    for (auto& de : fs::directory_iterator(dir)) {
        if (is_executable(de)) {
            paths.push_back(de.path());
        }
    }

    std::sort(paths.begin(), paths.end());
}

static void load_hook_cache(HookCache& hc) {
    fs::create_directories(fs::path(ROOT) / "charging.d");
    fs::create_directories(fs::path(ROOT) / "discharging.d");
    fs::create_directories(fs::path(ROOT) / "state.d");
    scan_hook_dir(fs::path(ROOT) / "charging.d",    hc.charging);
    scan_hook_dir(fs::path(ROOT) / "discharging.d", hc.discharging);
    scan_hook_dir(fs::path(ROOT) / "state.d",       hc.state);
    hc.loaded = true;
}

static void run_paths(
    const std::vector<fs::path>& paths,
    int visible_percent,
    ChargeStatus status,
    bool reverse = false
) {
    if (reverse) {
        for (auto it = paths.rbegin(); it != paths.rend(); ++it) {
            run_hook_file(*it, visible_percent, status);
        }
    } else {
        for (const auto& p : paths) {
            run_hook_file(p, visible_percent, status);
        }
    }
}

static void run_direction_hooks_cached(
    const HookCache& hc,
    bool charging,
    int visible_percent,
    ChargeStatus status
) {
    if (!hc.loaded || visible_percent < 0) return;

    const auto& paths = charging ? hc.charging : hc.discharging;

    // Launch charging hooks high-to-low; discharging hooks low-to-high.
    run_paths(paths, visible_percent, status, charging);
}

static void run_state_hooks_cached(const HookCache& hc, int visible_percent, ChargeStatus status) {
    if (!hc.loaded) return;
    run_paths(hc.state, visible_percent, status);
}

// ========================= Battery discovery =========================
struct BatteryPaths {
    fs::path base_dir;
    fs::path status;
    fs::path voltage_now;
    fs::path capacity;
};

static std::optional<BatteryPaths> find_battery() {
    auto has_status = [](const fs::path& d) {
        return fs::exists(d / "status");
    };

    auto has_battery_signal = [](const fs::path& d) {
        return fs::exists(d / "voltage_now") ||
               fs::exists(d / "voltage_avg") ||
               fs::exists(d / "capacity");
    };

    // Prefer battery/fuel-gauge supplies that expose something we can actually use.
    std::vector<std::string> patterns = {
        "BAT", "bat", "FUEL", "fuel"
    };

    fs::path base("/sys/class/power_supply");
    if (!fs::exists(base)) return std::nullopt;

    auto build_paths = [](const fs::path& d) -> BatteryPaths {
        BatteryPaths battery;
        battery.base_dir = d;
        battery.status = d / "status";

        // Prefer voltage_now, with voltage_avg as a fallback.
        if (fs::exists(d / "voltage_now")) {
            battery.voltage_now = d / "voltage_now";
        } else if (fs::exists(d / "voltage_avg")) {
            battery.voltage_now = d / "voltage_avg";
        }

        if (fs::exists(d / "capacity"))
            battery.capacity = d / "capacity";

        return battery;
    };

    for (auto& de : fs::directory_iterator(base)) {
        std::string name = de.path().filename().string();
        bool match = false;

        for (auto& p : patterns) {
            if (name.find(p) != std::string::npos) {
                match = true;
                break;
            }
        }

        if (!match) continue;

        if (has_status(de.path()) && has_battery_signal(de.path())) {
            return build_paths(de.path());
        }
    }

    return std::nullopt;
}

// ========================= Map file =========================
// Persistent learned voltage anchors.
struct VoltageAnchors {
    int V_FULL_CHG = DEFAULT_V_FULL_CHG;
    int V_FULL_DIS = DEFAULT_V_FULL_DIS;
};

static bool save_map_atomic(const fs::path& path, const VoltageAnchors& m) {
    std::string data;
    data += "V_FULL_CHG=" + std::to_string(m.V_FULL_CHG) + "\n";
    data += "V_FULL_DIS=" + std::to_string(m.V_FULL_DIS) + "\n";

    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    if (ec) return false;

    return write_atomic(path, data, 0644);
}

// Fall back to defaults if the saved anchors aren't usable.
static VoltageAnchors load_map(const fs::path& path, bool& valid_pair) {
    valid_pair = false;
    VoltageAnchors m;
    bool need_save = false;
    bool found_vfull_chg = false;
    bool found_vfull_dis = false;

    std::ifstream f(path);
    if (f) {
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("V_FULL_CHG=", 0) == 0) {
                m.V_FULL_CHG = std::atoi(line.c_str() + 11);
                found_vfull_chg = true;
            } else if (line.rfind("V_FULL_DIS=", 0) == 0) {
                m.V_FULL_DIS = std::atoi(line.c_str() + 11);
                found_vfull_dis = true;
            }
        }
    } else {
        return m;
    }

    if (!found_vfull_chg) {
        m.V_FULL_CHG = DEFAULT_V_FULL_CHG;
        need_save = true;
    }

    if (!found_vfull_dis) {
        m.V_FULL_DIS = DEFAULT_V_FULL_DIS;
        need_save = true;
    }

    if (m.V_FULL_CHG < 3600 || m.V_FULL_CHG > 4600) {
        m.V_FULL_CHG = DEFAULT_V_FULL_CHG;
        need_save = true;
    }

    if (m.V_FULL_DIS < 3600 || m.V_FULL_DIS > 4600) {
        m.V_FULL_DIS = DEFAULT_V_FULL_DIS;
        need_save = true;
    }

    if (m.V_FULL_CHG < m.V_FULL_DIS) {
        m.V_FULL_CHG = DEFAULT_V_FULL_CHG;
        m.V_FULL_DIS = DEFAULT_V_FULL_DIS;
        need_save = true;
    }

    // Repaired defaults don't count as a learned pair.
    valid_pair = !need_save;

    return m;
}

// Save one calibration pair exactly and lock further learning.
static bool commit_full_pair(
    int charge_mv,
    int discharge_mv,
    VoltageAnchors& anchors,
    CalibrationState& calibration
) {
    if (charge_mv < 3600 || charge_mv > 4600 ||
        discharge_mv < 3600 || discharge_mv > 4600 ||
        discharge_mv > charge_mv) {

        DEBUG_LOG(
            "calibration",
            "result=rejected,reason=invalid_pair,"
            "charge_mv=%d,discharge_mv=%d",
            charge_mv,
            discharge_mv
        );

        return false;
    }

    VoltageAnchors next;
    next.V_FULL_CHG = charge_mv;
    next.V_FULL_DIS = discharge_mv;

    if (!save_map_atomic(g_paths.map_file, next)) {
        DEBUG_LOG(
            "calibration",
            "result=failed,reason=map_write,"
            "charge_mv=%d,discharge_mv=%d",
            charge_mv,
            discharge_mv
        );

        return false;
    }

    if (!write_atomic(g_paths.calibrated_flag, "", 0644)) {
        DEBUG_LOG(
            "calibration",
            "result=failed,reason=flag_write,"
            "charge_mv=%d,discharge_mv=%d",
            charge_mv,
            discharge_mv
        );

        return false;
    }

    DEBUG_LOG(
        "calibration",
        "result=saved,mode=exact,"
        "previous_charge_mv=%d,previous_discharge_mv=%d,"
        "saved_charge_mv=%d,saved_discharge_mv=%d",
        anchors.V_FULL_CHG,
        anchors.V_FULL_DIS,
        next.V_FULL_CHG,
        next.V_FULL_DIS
    );

    anchors = next;
    calibration.enabled = false;

    return true;
}

// ========================= Percent calc =========================
struct VoltageFilterState {
    int prev1 = -1;
    int prev2 = -1;
    double ema = -1.0;
};

// Restart the filter from a known voltage.
static void seed_smoothed_voltage(VoltageFilterState& voltage_state, int voltage_mv) {
    voltage_state.prev1 = voltage_mv;
    voltage_state.prev2 = voltage_mv;
    voltage_state.ema = voltage_mv;
}

// Tracks the charging peak until a full event is confirmed.
struct FullLearningState {
    bool tracking_charge_peak = false;
    int peak_charge_ema_mv = -1;

    bool full_event_active = false;

    bool peak_timer_active = false;
    int64_t peak_start_bt_s = 0;
    int64_t peak_deadline_bt_s = 0;
    int64_t peak_drop_start_bt_s = -1;
};

static void reset_full_learn_state(
    FullLearningState& full_learning,
    const char* reason = "reset"
) {
    if (full_learning.tracking_charge_peak && !full_learning.full_event_active) {
        DEBUG_LOG(
            "dwell",
            "result=cancelled,reason=%s,peak_mv=%d,active=%d",
            reason,
            full_learning.peak_charge_ema_mv,
            full_learning.peak_timer_active ? 1 : 0
        );
    }

    full_learning.tracking_charge_peak = false;
    full_learning.peak_charge_ema_mv = -1;
    full_learning.full_event_active = false;
    full_learning.peak_timer_active = false;
    full_learning.peak_start_bt_s = 0;
    full_learning.peak_deadline_bt_s = 0;
    full_learning.peak_drop_start_bt_s = -1;
}

static double smootherstep(double x) {
    x = std::clamp(x, 0.0, 1.0);
    return x*x*x*(x*(x*6 - 15) + 10);
}

static double shape_scurve(double x, double strength) {
    x = std::clamp(x, 0.0, 1.0);
    const double s = smootherstep(x);
    strength = std::clamp(strength, 0.0, 1.0);
    return x + (s - x) * strength;
}

static int read_voltage_mv(const fs::path& voltage_now) {
    auto vopt = slurp_int(voltage_now);
    if (!vopt) return -1;
    int raw = *vopt;
    // Sysfs may report either microvolts or millivolts.
    if (raw >= 100000) {
        raw /= 1000;
    }
    return raw;
}

static int update_smoothed_voltage(int voltage_raw_mv, const VoltageAnchors& anchors, VoltageFilterState& voltage_state) {
    if (voltage_state.prev1 < 0)
        voltage_state.prev1 = (voltage_raw_mv > 0 ? voltage_raw_mv : anchors.V_FULL_DIS);
    if (voltage_state.prev2 < 0)
        voltage_state.prev2 = voltage_state.prev1;

    int v_med = median3(
        voltage_state.prev2,
        voltage_state.prev1,
        (voltage_raw_mv > 0 ? voltage_raw_mv : voltage_state.prev1)
    );

    voltage_state.prev2 = voltage_state.prev1;
    voltage_state.prev1 = (voltage_raw_mv > 0 ? voltage_raw_mv : voltage_state.prev1);

    if (voltage_state.ema < 0)
        voltage_state.ema = v_med;
    else
        voltage_state.ema = (ALPHA_NUM * v_med + (ALPHA_DEN - ALPHA_NUM) * voltage_state.ema) / ALPHA_DEN;

    return static_cast<int>(std::lround(voltage_state.ema));
}

static int burst_sample_voltage(const fs::path& voltage_now) {
    int a = read_voltage_mv(voltage_now);

    if (!sleep_interruptible_s(1)) return -1;
    int b = read_voltage_mv(voltage_now);

    if (!sleep_interruptible_s(1)) return -1;
    int c = read_voltage_mv(voltage_now);

    // Replace a failed read with one of the valid samples if possible.
    if (a <= 0) a = (b > 0 ? b : c);
    if (b <= 0) b = (a > 0 ? a : c);
    if (c <= 0) c = (a > 0 ? a : b);

    if (a <= 0 || b <= 0 || c <= 0) return -1;
    return median3(a, b, c);
}

// Session-specific correction learned from the voltage jump after plug-in.
struct ChargeCompensation {
    int preplug_voltage_mv = -1;
    int learned_inflation_mv = 0;

    std::array<int, 3> samples{};
    int sample_count = -1; // -1 means no initial measurement is pending.
    int64_t next_sample_bt_s = 0;

    int64_t next_refine_bt_s = 0;
    int64_t refine_until_bt_s = 0;

    int observed_voltage_mv = -1; // Voltage at the strongest observed inflation.
};

static bool has_established_charge_compensation(
    const ChargeCompensation& compensation
) {
    return compensation.learned_inflation_mv > 0 &&
           compensation.observed_voltage_mv > 0 &&
           compensation.sample_count < 0 &&
           compensation.refine_until_bt_s == 0;
}

// Estimate the normal charging offset from the learned full-voltage difference.
static double base_charge_offset(double voltage_mv, const VoltageAnchors& anchors) {
    constexpr double CHARGE_OFFSET_MULTIPLIER = 4.0;

    const double full_delta_mv =
        std::max(0, anchors.V_FULL_CHG - anchors.V_FULL_DIS);

    // Clamp to the same limit as measured charging inflation.
    const double max_offset_mv = std::min(
        full_delta_mv * CHARGE_OFFSET_MULTIPLIER,
        static_cast<double>(CHARGE_COMP_MAX_INFLATION_MV)
    );

    const double charge_range_mv =
        std::max(1, anchors.V_FULL_CHG - g_cfg.V_EMPTY);

    // Start tapering after accounting for the offset itself.
    double taper_range_mv = charge_range_mv - max_offset_mv;

    // Fall back to the full range if the adjusted range is unusable.
    if (taper_range_mv <= 0.0) {
        taper_range_mv = charge_range_mv;
    }

    const double taper = std::clamp(
        (anchors.V_FULL_CHG - voltage_mv) / taper_range_mv,
        0.0,
        1.0
    );

    // Retain the full-anchor gap even when charging voltage reaches its peak.
    return full_delta_mv + (max_offset_mv - full_delta_mv) * taper;
}

// Learn charging inflation from the observed plug-in voltage jump.
static void observe_charge_transition(
    ChargeCompensation& compensation,
    int previous_voltage_mv,
    int current_voltage_mv
) {
    compensation.learned_inflation_mv = 0;
    compensation.observed_voltage_mv = -1;

    if (previous_voltage_mv <= 0 ||
        current_voltage_mv <= previous_voltage_mv) {
        return;
    }

    compensation.learned_inflation_mv = std::clamp(
        current_voltage_mv - previous_voltage_mv,
        0,
        CHARGE_COMP_MAX_INFLATION_MV
    );

    if (compensation.learned_inflation_mv > 0) {
        compensation.observed_voltage_mv = current_voltage_mv;
    }
}

static void refine_charge_compensation(
    ChargeCompensation& compensation,
    int current_voltage_mv,
    const VoltageAnchors& anchors,
    int64_t now_bt_s
) {
    if (compensation.next_refine_bt_s <= 0 ||
        compensation.refine_until_bt_s <= 0 ||
        now_bt_s < compensation.next_refine_bt_s) {
        return;
    }

    const int observed_inflation_mv =
        current_voltage_mv - compensation.preplug_voltage_mv;

    const int clamped_inflation_mv = std::clamp(
        observed_inflation_mv,
        0,
        CHARGE_COMP_MAX_INFLATION_MV
    );

    bool updated = false;

    // Only increase the learned inflation during the refinement window.
    if (compensation.preplug_voltage_mv > 0 &&
        current_voltage_mv > compensation.preplug_voltage_mv &&
        clamped_inflation_mv > compensation.learned_inflation_mv) {

        compensation.learned_inflation_mv =
            clamped_inflation_mv;

        compensation.observed_voltage_mv = current_voltage_mv;

        updated = true;
    }

    DEBUG_LOG(
        "charge_compensation",
        "result=refine,"
        "current_mv=%d,observed_inflation_mv=%d,"
        "learned_inflation_mv=%d,"
        "base_offset_mv=%.1f,updated=%d,remaining_s=%lld",
        current_voltage_mv,
        observed_inflation_mv,
        compensation.learned_inflation_mv,
        base_charge_offset(current_voltage_mv, anchors),
        updated ? 1 : 0,
        static_cast<long long>(
            std::max<int64_t>(
                0,
                compensation.refine_until_bt_s - now_bt_s
            )
        )
    );

    if (now_bt_s >= compensation.refine_until_bt_s) {
        compensation.next_refine_bt_s = 0;
        compensation.refine_until_bt_s = 0;

        DEBUG_LOG(
            "charge_compensation",
            "result=curve_ready,observed_mv=%d,learned_inflation_mv=%d,"
            "full_offset_mv=%d",
            compensation.observed_voltage_mv,
            compensation.learned_inflation_mv,
            std::max(0, anchors.V_FULL_CHG - anchors.V_FULL_DIS)
        );

        return;
    }

    compensation.next_refine_bt_s += INTERNAL_INTERVAL_S;

    if (compensation.next_refine_bt_s >
        compensation.refine_until_bt_s) {
        compensation.next_refine_bt_s =
            compensation.refine_until_bt_s;
    }
}

// Use the measured offset during refinement, then taper it to the full-anchor gap.
static double charging_offset_mv(
    int voltage_raw_mv,
    const VoltageAnchors& anchors,
    const ChargeCompensation& compensation
) {
    if (compensation.learned_inflation_mv <= 0) {
        return base_charge_offset(voltage_raw_mv, anchors);
    }

    const double learned_mv = compensation.learned_inflation_mv;

    // Keep the measured offset unchanged while refinement is active.
    if (!has_established_charge_compensation(compensation)) {
        return learned_mv;
    }

    const int observed_mv = compensation.observed_voltage_mv;
    const int full_charge_mv = anchors.V_FULL_CHG;
    const int full_offset_mv =
        std::max(0, anchors.V_FULL_CHG - anchors.V_FULL_DIS);

    // Avoid an invalid taper range if the default full anchor is too low.
    if (observed_mv >= full_charge_mv) {
        return learned_mv;
    }

    const double progress = std::clamp(
        (voltage_raw_mv - observed_mv) /
            static_cast<double>(full_charge_mv - observed_mv),
        0.0,
        1.0
    );

    return learned_mv + (full_offset_mv - learned_mv) * progress;
}

// Convert terminal voltage into its estimated non-charging equivalent.
static int correct_voltage_for_percent(
    int voltage_raw_mv,
    const VoltageAnchors& anchors,
    bool charging,
    const ChargeCompensation& compensation
) {
    if (!charging || voltage_raw_mv <= 0) {
        return voltage_raw_mv;
    }

    const double offset_mv = std::max(
        0.0,
        charging_offset_mv(
            voltage_raw_mv,
            anchors,
            compensation
        )
    );

    return static_cast<int>(std::lround(
        voltage_raw_mv - offset_mv
    ));
}

// Collect plug-in samples without delaying hooks or the main loop.
// Return true when sampling finishes or is cancelled so we can recalculate.
static bool update_charge_compensation(
    const BatteryPaths& battery,
    const VoltageAnchors& anchors,
    VoltageFilterState& voltage_state,
    VoltageFilterState& percent_voltage_state,
    ChargeCompensation& charge_compensation,
    bool charging,
    bool charging_changed,
    ChargeStatus last_charge_status,
    bool resume_detected,
    int visible_percent,
    int64_t now_bt_s,
    long gap_s
) {

    bool charge_measurement_finished = false;
    const bool measurement_pending = charge_compensation.sample_count >= 0;

    const bool preserve_resume_compensation =
        resume_detected &&
        charging &&
        is_charging_status(last_charge_status) &&
        !measurement_pending &&
        has_established_charge_compensation(charge_compensation);

    // Preserve established compensation across Charging -> Charging resume.
    if ((resume_detected && !preserve_resume_compensation) ||
        charging_changed ||
        !charging ||
        (measurement_pending && gap_s > 2)) {

        if (measurement_pending) {
            DEBUG_LOG(
                "charge_compensation",
                "result=cancelled,reason=%s,samples=%d,gap_s=%ld",
                resume_detected ? "resume" :
                !charging ? "unplug" :
                gap_s > 2 ? "loop_gap" :
                "state_change",
                charge_compensation.sample_count,
                gap_s
            );
        }

        charge_compensation = {};
        charge_measurement_finished = measurement_pending;
    } else if (preserve_resume_compensation) {
        DEBUG_LOG(
            "charge_compensation",
            "result=resume_preserved,"
            "learned_inflation_mv=%d,observed_mv=%d",
            charge_compensation.learned_inflation_mv,
            charge_compensation.observed_voltage_mv
        );
    }

    // Compare the pre-plug voltage with three post-plug samples.
    if (charging_changed &&
        charging &&
        last_charge_status == ChargeStatus::Discharging &&
        !resume_detected &&
        gap_s <= 2 &&
        (voltage_state.prev1 > 0 || voltage_state.ema > 0.0) &&
        visible_percent >= 0) {
        charge_compensation.preplug_voltage_mv =
            voltage_state.prev1 > 0
                ? voltage_state.prev1
                : static_cast<int>(std::lround(voltage_state.ema));
        charge_compensation.sample_count = 0;
        charge_compensation.next_sample_bt_s = now_bt_s + 1;


        DEBUG_LOG(
            "charge_compensation",
            "result=started,preplug_mv=%d,visible_percent=%d",
            charge_compensation.preplug_voltage_mv,
            visible_percent
        );
    }

    if (charge_compensation.sample_count >= 0 &&
        now_bt_s >= charge_compensation.next_sample_bt_s) {
        const int sample_mv = read_voltage_mv(battery.voltage_now);

        if (sample_mv <= 0) {
            DEBUG_LOG(
                "charge_compensation",
                "result=cancelled,reason=voltage_read_failed,samples=%d",
                charge_compensation.sample_count
            );

            // Stop waiting if voltage can't be read; use the base offset instead.
            charge_compensation = {};
            charge_measurement_finished = true;
        } else {
            charge_compensation.samples[charge_compensation.sample_count++] = sample_mv;
            charge_compensation.next_sample_bt_s = now_bt_s + 1;

            if (charge_compensation.sample_count == 3) {
                const int settled_voltage_mv = median3(
                    charge_compensation.samples[0],
                    charge_compensation.samples[1],
                    charge_compensation.samples[2]
                );

                observe_charge_transition(
                    charge_compensation,
                    charge_compensation.preplug_voltage_mv,
                    settled_voltage_mv
                );

                // Keep refining the observed charging inflation during the bounded window.
                charge_compensation.next_refine_bt_s = now_bt_s + INTERNAL_INTERVAL_S;
                charge_compensation.refine_until_bt_s = now_bt_s + CHARGE_COMP_REFINE_WINDOW_S;

                DEBUG_LOG(
                    "charge_compensation",
                    "result=complete,"
                    "preplug_mv=%d,"
                    "samples=%d/%d/%d,"
                    "settled_mv=%d,observed_inflation_mv=%d,"
                    "base_offset_mv=%.1f,learned_inflation_mv=%d,accepted=%d",
                    charge_compensation.preplug_voltage_mv,
                    charge_compensation.samples[0],
                    charge_compensation.samples[1],
                    charge_compensation.samples[2],
                    settled_voltage_mv,
                    settled_voltage_mv - charge_compensation.preplug_voltage_mv,
                    base_charge_offset(settled_voltage_mv, anchors),
                    charge_compensation.learned_inflation_mv,
                    charge_compensation.learned_inflation_mv > 0 ? 1 : 0
                );

                seed_smoothed_voltage(voltage_state, settled_voltage_mv);

                const int corrected_voltage_mv =
                    correct_voltage_for_percent(
                        settled_voltage_mv,
                        anchors,
                        charging,
                        charge_compensation
                    );

                seed_smoothed_voltage(percent_voltage_state, corrected_voltage_mv);

                charge_compensation.sample_count = -1;
                charge_measurement_finished = true;
            }
        }
    }

    // Reseed from current voltage if sampling didn't already give us one.
    if ((charging_changed ||
         (charge_measurement_finished &&
          charge_compensation.learned_inflation_mv <= 0)) &&
        !resume_detected &&
        charge_compensation.sample_count < 0) {
        const int current_voltage_mv = read_voltage_mv(battery.voltage_now);

        if (current_voltage_mv > 0) {
            seed_smoothed_voltage(
                voltage_state,
                current_voltage_mv
            );

            const int corrected_voltage_mv =
                correct_voltage_for_percent(
                    current_voltage_mv,
                    anchors,
                    charging,
                    charge_compensation
                );

            seed_smoothed_voltage(
                percent_voltage_state,
                corrected_voltage_mv
            );
        }
    }

    return charge_measurement_finished;
}

// Convert filtered, charging-corrected voltage to percent.
static int voltage_to_percent(
    int voltage_now_mv,
    const VoltageAnchors& anchors,
    bool charging
) {
    if (voltage_now_mv <= 0) {
        return 1;
    }

    const int v_empty = g_cfg.V_EMPTY;
    const int v_full = anchors.V_FULL_DIS;

    if (v_full <= v_empty) {
        return 1;
    }

    // Scale the full-voltage plateau to the voltage range.
    const int discharge_100_window_mv =
        static_cast<int>(std::lround(
            (v_full - v_empty) *
            (DISCHARGE_100_WINDOW_PERCENT / 100.0)
        ));

    const int v_100_start = std::max(
        v_full - discharge_100_window_mv,
        v_empty + MIN_RANGE_MV
    );

    const double x = std::clamp(
        (voltage_now_mv - v_empty) /
            static_cast<double>(v_100_start - v_empty),
        0.0,
        1.0
    );

    // Blend linear voltage mapping with a smooth S-curve.
    constexpr double SCURVE_STRENGTH_LOW = 0.75;
    constexpr double SCURVE_STRENGTH_HIGH = 0.50;

    const double strength = SCURVE_STRENGTH_LOW - (SCURVE_STRENGTH_LOW - SCURVE_STRENGTH_HIGH) * smootherstep(x);

    const double shaped = shape_scurve(x, strength);

    return clampi(static_cast<int>(std::lround(shaped * 100.0)), 0, charging ? 99 : 100);
}

// Visible percent stays monotonic within the current charge direction.
// Charging never counts down; discharging never counts up.
static int step_limit(int last, int target, bool charging) {
    if (last < 0) return target;
    if (charging) {
        if (target <= last) return last;
        if (target <= last + 1) return target;
        return last + 1;
    } else {
        if (target >= last) return last;
        if (target >= last - 1) return target;
        return last - 1;
    }
}

// Decide whether visible percent should update this loop.
static bool check_update_visible_percent(
    bool first_visible,
    bool snap_now,
    bool resume_update,
    int internal_percent,
    int visible_percent,
    ChargeStatus status,
    std::chrono::steady_clock::time_point now,
    std::chrono::steady_clock::time_point last_visible_write
) {
    const bool charging = is_charging_status(status);

    if (first_visible) {
        return true;
    }

    if (snap_now) {
        return true;
    }

    if (resume_update && internal_percent != visible_percent) {
        return true;
    }

    // Don't reset the write timer when monotonic output prevents a change.
    if ((charging && internal_percent <= visible_percent) ||
        (!charging && internal_percent >= visible_percent)) {
        return false;
    }

    auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(now - last_visible_write).count();

    int required_interval = WRITE_INTERVAL_S;
    int delta = std::abs(internal_percent - visible_percent);

    // Larger gaps catch up sooner.
    if (delta >= 6) {
        required_interval = WRITE_INTERVAL_DELTA_LARGE_S;
    } else if (delta >= 3) {
        required_interval = WRITE_INTERVAL_DELTA_SMALL_S;
    }

    // Slow charging updates a little; Full uses normal catch-up timing.
    if (status == ChargeStatus::Charging) {
        required_interval *= CHARGE_VISIBLE_INTERVAL_MULTIPLIER;
    }

    return elapsed_s >= required_interval;
}

static int compute_new_visible_percent(
    bool first_visible,
    bool snap_now,
    int visible_percent,
    int internal_percent,
    bool charging
) {
    if (first_visible || snap_now) {
        return clampi(internal_percent, 0, 100);
    }

    return clampi(step_limit(visible_percent, internal_percent, charging), 0, 100);
}

// Tracks adaptive post-unplug sampling for V_FULL_DIS.
struct UnplugLearnState {
    bool pending = false;
    int charge_mv = -1;

    int64_t start_bt_s = 0;
    int64_t next_sample_bt_s = 0;

    std::array<int, 3> samples{};
    int sample_count = 0;

    int last_candidate_mv = -1;
    int stable_count = 0;

    std::array<int, 6> candidates{};
    int candidate_count = 0;
};

// After unplugging from full, let voltage settle and learn the discharge anchor.
static bool update_unplug_full_learning(
    const BatteryPaths& battery,
    VoltageAnchors& anchors,
    VoltageFilterState& voltage_state,
    VoltageFilterState& percent_voltage_state,
    int visible_percent,
    FullLearningState& full_learning,
    CalibrationState& calibration,
    UnplugLearnState& unplug_learning,
    ChargeStatus status,
    bool charging_changed,
    bool resume_detected,
    int64_t now_bt_s,
    long gap_s
) {
    if (resume_detected || gap_s > 2 || status == ChargeStatus::Unknown) {
        if (unplug_learning.pending) {
            DEBUG_LOG(
                "unplug_learning",
                "result=cancelled,reason=%s,"
                "charge_mv=%d,samples=%d",
                resume_detected ? "resume" :
                gap_s > 2 ? "loop_gap" :
                "status_unknown",
                unplug_learning.charge_mv,
                unplug_learning.sample_count
            );
        }

        unplug_learning = {};
        reset_full_learn_state(
            full_learning,
            resume_detected ? "resume" : gap_s > 2 ? "loop_gap" : "status_unknown"
        );
        return false;
    }

    if (status != ChargeStatus::Discharging) {
        // Replugging cancels the pending unplug capture.
        if (unplug_learning.pending) {
            DEBUG_LOG(
                "unplug_learning",
                "result=cancelled,reason=replug,"
                "charge_mv=%d,samples=%d",
                unplug_learning.charge_mv,
                unplug_learning.sample_count
            );

            unplug_learning = {};
            reset_full_learn_state(full_learning);
        }
        return false;
    }

    // Full detection remains active after calibration. Learning is enabled
    // again by removing the calibration flag and restarting the daemon.
    if (!calibration.enabled) {
        unplug_learning = {};
        reset_full_learn_state(full_learning, "discharging");
        return false;
    }

    if (charging_changed && full_learning.full_event_active) {
        // Only calibrate after Full has actually reached the visible UI.
        if (visible_percent != 100) {
            reset_full_learn_state(full_learning);
            return false;
        }

        unplug_learning = {};
        unplug_learning.pending = true;
        unplug_learning.charge_mv = full_learning.peak_charge_ema_mv;
        unplug_learning.start_bt_s = now_bt_s;
        unplug_learning.next_sample_bt_s =
            now_bt_s + VFULL_DIS_INITIAL_SETTLE_S;

        DEBUG_LOG(
            "unplug_learning",
            "result=started,charge_mv=%d,"
            "initial_settle_s=%d,check_s=%d,"
            "stable_mv=%d,stable_count=%d,max_wait_s=%d",
            unplug_learning.charge_mv,
            VFULL_DIS_INITIAL_SETTLE_S,
            VFULL_DIS_CHECK_S,
            VFULL_DIS_STABLE_MV,
            VFULL_DIS_STABLE_COUNT,
            VFULL_DIS_MAX_WAIT_S
        );

        reset_full_learn_state(full_learning);
    }

    // The unplug samples keep their own copy of the charging peak.
    reset_full_learn_state(full_learning, "discharging");

    if (!unplug_learning.pending) {
        return false;
    }

    int settled_voltage_mv = -1;
    const char* settle_reason = nullptr;

    if (now_bt_s < unplug_learning.next_sample_bt_s) {
        return false;
    }

    const int voltage_mv =
        read_voltage_mv(battery.voltage_now);

    if (voltage_mv <= 0) {
        DEBUG_LOG(
            "unplug_learning",
            "result=cancelled,reason=voltage_read_failed,"
            "charge_mv=%d,samples=%d",
            unplug_learning.charge_mv,
            unplug_learning.sample_count
        );

        unplug_learning = {};
        return false;
    }

    unplug_learning.samples[
        unplug_learning.sample_count++
    ] = voltage_mv;

    if (unplug_learning.sample_count < 3) {
        unplug_learning.next_sample_bt_s =
            now_bt_s + 1;
        return false;
    }

    const int candidate_mv =
        median(unplug_learning.samples);

    if (unplug_learning.candidate_count <
        static_cast<int>(
            unplug_learning.candidates.size()
        )) {

        unplug_learning.candidates[
            unplug_learning.candidate_count++
        ] = candidate_mv;
    }

    int candidate_delta_mv = -1;

    if (unplug_learning.last_candidate_mv > 0) {
        candidate_delta_mv = std::abs(
            candidate_mv -
            unplug_learning.last_candidate_mv
        );

        if (candidate_delta_mv <=
            VFULL_DIS_STABLE_MV) {

            ++unplug_learning.stable_count;
        } else {
            unplug_learning.stable_count = 0;
        }
    }

    const int64_t completed_elapsed_s =
        now_bt_s - unplug_learning.start_bt_s;

    DEBUG_LOG(
        "unplug_learning",
        "result=candidate,"
        "elapsed_s=%lld,"
        "samples=%d/%d/%d,"
        "candidate_mv=%d,previous_mv=%d,"
        "delta_mv=%d,stable_count=%d,"
        "candidate_count=%d",
        static_cast<long long>(completed_elapsed_s),
        unplug_learning.samples[0],
        unplug_learning.samples[1],
        unplug_learning.samples[2],
        candidate_mv,
        unplug_learning.last_candidate_mv,
        candidate_delta_mv,
        unplug_learning.stable_count,
        unplug_learning.candidate_count
    );

    // On timeout, ignore the first burst and use the median of later candidates.
    if (completed_elapsed_s >=
        VFULL_DIS_MAX_WAIT_S) {

        if (unplug_learning.candidate_count > 1) {
            settled_voltage_mv = median_values(
                unplug_learning.candidates.data() + 1,
                unplug_learning.candidate_count - 1
            );
        } else {
            settled_voltage_mv = candidate_mv;
        }

        settle_reason = "timeout";
    } else if (
        unplug_learning.stable_count >=
        VFULL_DIS_STABLE_COUNT) {

        settled_voltage_mv = candidate_mv;
        settle_reason = "stable";
    } else {
        unplug_learning.last_candidate_mv =
            candidate_mv;

        unplug_learning.sample_count = 0;

        // A 2-second burst plus an 8-second wait keeps medians about 10 seconds apart.
        unplug_learning.next_sample_bt_s =
            now_bt_s + VFULL_DIS_CHECK_S;

        return false;
    }


    DEBUG_LOG(
        "unplug_learning",
        "result=complete,reason=%s,"
        "charge_mv=%d,settled_mv=%d,"
        "elapsed_s=%lld,candidates=%d",
        settle_reason,
        unplug_learning.charge_mv,
        settled_voltage_mv,
        static_cast<long long>(
            now_bt_s - unplug_learning.start_bt_s
        ),
        unplug_learning.candidate_count
    );

    // A failure aborts this calibration attempt.
    if (!commit_full_pair(
            unplug_learning.charge_mv,
            settled_voltage_mv,
            anchors,
            calibration)) {
        unplug_learning = {};
        reset_full_learn_state(full_learning);
        return false;
    }

    seed_smoothed_voltage(voltage_state, settled_voltage_mv);
    seed_smoothed_voltage(percent_voltage_state, settled_voltage_mv);
    unplug_learning = {};
    reset_full_learn_state(full_learning);
    return true;
}

// Restart dwell whenever a new filtered charging peak is observed.
static void update_peak_tracking(
    FullLearningState& full_learning,
    const VoltageAnchors& anchors,
    bool charging,
    bool status_full,
    int voltage_ema_mv,
    int64_t now_bt_s
) {
    if (full_learning.full_event_active || voltage_ema_mv <= 0) {
        return;
    }

    if (!charging) {
        reset_full_learn_state(full_learning, "discharging");
        return;
    }

    if (status_full) {
        return;
    }

    const bool first_peak = !full_learning.tracking_charge_peak;
    if (first_peak) {
        if (voltage_ema_mv < anchors.V_FULL_CHG - PEAK_TRACK_RANGE_MV) {
            return;
        }
    } else if (voltage_ema_mv <= full_learning.peak_charge_ema_mv) {
        return;
    }

    full_learning.tracking_charge_peak = true;
    full_learning.peak_charge_ema_mv = voltage_ema_mv;
    full_learning.peak_timer_active = true;
    full_learning.peak_start_bt_s = now_bt_s;
    full_learning.peak_deadline_bt_s = now_bt_s + PEAK_DWELL_S;

    DEBUG_LOG(
        "dwell",
        "result=%s,reason=%s,ema_mv=%d,peak_mv=%d,remaining_s=%d",
        first_peak ? "started" : "restarted",
        first_peak ? "near_full" : "new_peak",
        voltage_ema_mv,
        full_learning.peak_charge_ema_mv,
        PEAK_DWELL_S
    );
}

// Check whether the dwell timer has finished.
static bool check_peak_dwell_met(
    const FullLearningState& full_learning,
    int64_t now_bt_s
) {
    return full_learning.peak_timer_active &&
           now_bt_s >= full_learning.peak_deadline_bt_s;
}

// Lock in the charge-side peak. It isn't saved until unplug learning completes.
static void check_charge_full_event(
    FullLearningState& full_learning,
    int voltage_raw_mv,
    bool full_event_triggered
) {
    if (full_learning.full_event_active || !full_event_triggered || voltage_raw_mv <= 0 ||
        !full_learning.tracking_charge_peak || full_learning.peak_charge_ema_mv < 3600 ||
        full_learning.peak_charge_ema_mv > 4600) {
        return;
    }

    full_learning.full_event_active = true;
}

// ========================= Percent Calculation Modes =========================
struct PercentResult {
    int percent = -1;
    bool valid = false;
};

// Update voltage state, track full learning, and calculate internal percent.
static PercentResult run_voltage_mode(
    const BatteryPaths& battery,
    VoltageAnchors& anchors,
    VoltageFilterState& voltage_state,
    VoltageFilterState& percent_voltage_state,
    FullLearningState& full_learning,
    bool charging,
    ChargeCompensation& compensation,
    bool status_full,
    int64_t now_bt_s
) {
    PercentResult result;

    int voltage_raw_mv = read_voltage_mv(battery.voltage_now);

    if (voltage_raw_mv <= 0) {
        full_learning.peak_drop_start_bt_s = -1;
        return result;
    }

    const int voltage_raw_ema_mv =
        update_smoothed_voltage(
            voltage_raw_mv,
            anchors,
            voltage_state
        );

    if (charging) {
        refine_charge_compensation(
            compensation,
            voltage_raw_mv,
            anchors,
            now_bt_s
        );
    }

    const int corrected_voltage_mv =
        correct_voltage_for_percent(
            voltage_raw_mv,
            anchors,
            charging,
            compensation
        );

    const int voltage_percent_ema_mv =
        update_smoothed_voltage(
            corrected_voltage_mv,
            anchors,
            percent_voltage_state
        );

    update_peak_tracking(
        full_learning,
        anchors,
        charging,
        status_full,
        voltage_raw_ema_mv,
        now_bt_s
    );

    bool peak_dwell_met = check_peak_dwell_met(full_learning, now_bt_s);

    // Cancel confirmation if raw voltage recovers before the filter catches up.
    const bool peak_drop_present =
        charging && full_learning.peak_timer_active &&
        full_learning.peak_charge_ema_mv - voltage_raw_ema_mv >= PEAK_FULL_DROP_MV &&
        full_learning.peak_charge_ema_mv - voltage_raw_mv >= PEAK_FULL_DROP_MV;

    if (!peak_drop_present) {
        full_learning.peak_drop_start_bt_s = -1;
    } else if (full_learning.peak_drop_start_bt_s < 0) {
        full_learning.peak_drop_start_bt_s = now_bt_s;
    }

    const bool peak_drop_met =
        peak_drop_present &&
        now_bt_s - full_learning.peak_drop_start_bt_s >= PEAK_FULL_DROP_CONFIRM_S;

    bool full_event_triggered = status_full || peak_dwell_met || peak_drop_met;

    bool allow_visible_100 = full_event_triggered || full_learning.full_event_active;

    int percent = voltage_to_percent(
        voltage_percent_ema_mv,
        anchors,
        charging
    );

    // Hold charging at 99% until full is reported or inferred.
    if (charging) {
        if (allow_visible_100) {
            percent = 100;
        } else if (percent > 99) {
            percent = 99;
        }
    }

    const bool full_event_was_active = full_learning.full_event_active;

    check_charge_full_event(
        full_learning,
        voltage_raw_mv,
        charging && full_event_triggered
    );

    if (!full_event_was_active &&
        full_learning.full_event_active) {

        DEBUG_LOG(
            "full_event",
            "source=%s,raw_mv=%d,ema_mv=%d,"
            "peak_mv=%d,dwell_s=%lld",
            status_full && peak_dwell_met ? "pmic+dwell" :
            status_full ? "pmic" :
            peak_drop_met ? "peak_drop" :
            "dwell",
            voltage_raw_mv,
            voltage_raw_ema_mv,
            full_learning.peak_charge_ema_mv,
            static_cast<long long>(
                full_learning.peak_timer_active
                    ? now_bt_s - full_learning.peak_start_bt_s
                    : 0
            )
        );
    }

    result.percent = percent;
    result.valid = true;
    return result;
}

// PMIC mode uses the reported capacity and the same visible-percent pipeline.
static PercentResult run_pmic_mode(
    const fs::path& capacity_path
) {
    PercentResult result;

    auto pct_opt = slurp_int(capacity_path);
    if (!pct_opt) {
        return result;
    }

    int percent = clampi(*pct_opt, 0, 100);

    result.percent = percent;
    result.valid = true;
    return result;
}

// Decide how much of the saved percentage is still trustworthy.
enum class RestoreMode {
    Fresh,
    Blend,
    Exact
};

struct RestoreDecision {
    RestoreMode mode = RestoreMode::Fresh;
    int visible_percent = -1;
    double saved_weight = 0.0;
};

static const char* restore_mode_arg(RestoreMode mode) {
    switch (mode) {
        case RestoreMode::Exact: return "exact";
        case RestoreMode::Blend: return "blend";
        default: return "fresh";
    }
}

static RestoreDecision reconcile_restore_percent(
    const RestoreState& saved,
    int current_voltage_mv,
    bool charging,
    int fresh_percent
) {
    RestoreDecision result;

    if (current_voltage_mv <= 0 ||
        saved.saved_mv <= 0 ||
        saved.visible_percent < 0 ||
        saved.visible_percent > 100 ||
        fresh_percent < 0 ||
        fresh_percent > 100) {
        return result;
    }

    result.visible_percent = fresh_percent;

    const RestoreWindow& window = saved.charging
        ? (charging ? RESTORE_CHG_TO_CHG : RESTORE_CHG_TO_DIS)
        : (charging ? RESTORE_DIS_TO_CHG : RESTORE_DIS_TO_DIS);

    const int delta_mv = current_voltage_mv - saved.saved_mv;
    const int distance_mv = std::abs(delta_mv);
    const int exact_limit_mv = delta_mv < 0 ? window.down_mv : window.up_mv;

    // Close enough to keep the saved percentage exactly.
    if (distance_mv <= exact_limit_mv) {
        result.mode = RestoreMode::Exact;
        result.visible_percent = saved.visible_percent;
        result.saved_weight = 1.0;
        return result;
    }

    const double blend_limit_mv = exact_limit_mv * RESTORE_BLEND_SCALE;

    // Too far away to trust the saved state.
    if (distance_mv >= blend_limit_mv) {
        return result;
    }

    // Between the two limits, fade from saved percent toward the fresh result.
    result.mode = RestoreMode::Blend;
    result.saved_weight = std::clamp(
        (blend_limit_mv - distance_mv) /
            (blend_limit_mv - exact_limit_mv),
        0.0,
        1.0
    );

    int blended_percent = static_cast<int>(std::lround(
        saved.visible_percent * result.saved_weight +
        fresh_percent * (1.0 - result.saved_weight)
    ));

    // Don't let rounding make a blend look identical to the saved value when
    // normal monotonic stepping couldn't move toward the fresh result.
    const bool blocked_direction =
        (charging && fresh_percent < saved.visible_percent) ||
        (!charging && fresh_percent > saved.visible_percent);

    if (blocked_direction &&
        blended_percent == saved.visible_percent &&
        fresh_percent != saved.visible_percent) {
        blended_percent +=
            fresh_percent > saved.visible_percent ? 1 : -1;
    }

    result.visible_percent = clampi(blended_percent, 0, 100);
    return result;
}

// Calculate a fresh restore percentage without carrying over session compensation.
static int calculate_restore_fresh_percent(
    int voltage_mv,
    const VoltageAnchors& anchors,
    bool charging,
    bool status_full
) {
    ChargeCompensation no_extra_comp;

    const int corrected_voltage_mv =
        correct_voltage_for_percent(
            voltage_mv,
            anchors,
            charging,
            no_extra_comp
        );

    int percent = voltage_to_percent(
        corrected_voltage_mv,
        anchors,
        charging
    );

    if (charging && status_full) {
        percent = 100;
    }

    return percent;
}

struct ResumeRestoreResult {
    VisiblePolicy visible_policy = VisiblePolicy::Normal;
    std::optional<int> visible_percent;
};

// Reconcile the saved percentage and restart filtering from current voltage.
static ResumeRestoreResult restore_voltage_on_resume(
    const BatteryPaths& battery,
    const VoltageAnchors& anchors,
    VoltageFilterState& voltage_state,
    VoltageFilterState& percent_voltage_state,
    const RestoreState& saved,
    ChargeStatus current_status
) {
    ResumeRestoreResult result;

    if (current_status == ChargeStatus::Unknown) {
        result.visible_policy = VisiblePolicy::Hold;
        return result;
    }

    // Give voltage a moment to settle after resume, then use a short burst.
    if (!sleep_interruptible_s(1)) {
        result.visible_policy = VisiblePolicy::Hold;
        return result;
    }

    const int current_voltage_mv =
        burst_sample_voltage(battery.voltage_now);

    if (current_voltage_mv <= 0) {
        result.visible_policy = VisiblePolicy::Hold;

        DEBUG_LOG(
            "resume_restore",
            "result=failed,reason=voltage_read_failed,"
            "saved_mv=%d,saved_percent=%d",
            saved.saved_mv,
            saved.visible_percent
        );

        return result;
    }

    const bool charging = is_charging_status(current_status);

    const bool status_full = current_status == ChargeStatus::Full;

    const int fresh_percent = calculate_restore_fresh_percent(
        current_voltage_mv,
        anchors,
        charging,
        status_full
    );

    const RestoreDecision decision = reconcile_restore_percent(
        saved,
        current_voltage_mv,
        charging,
        fresh_percent
    );

    result.visible_percent = decision.visible_percent;

    // Blend/fresh results snap once; an exact restore keeps the existing UI value.
    if (decision.mode != RestoreMode::Exact) {
        result.visible_policy = VisiblePolicy::Snap;
    }

    DEBUG_LOG(
        "resume_restore",
        "previous=%s,current=%s,"
        "saved_mv=%d,current_mv=%d,delta_mv=%d,"
        "saved_percent=%d,fresh_percent=%d,"
        "mode=%s,saved_weight=%.3f,result_percent=%d",
        saved.charging ? "Charging" : "Discharging",
        charge_status_arg(current_status),
        saved.saved_mv,
        current_voltage_mv,
        current_voltage_mv - saved.saved_mv,
        saved.visible_percent,
        fresh_percent,
        restore_mode_arg(decision.mode),
        decision.saved_weight,
        decision.visible_percent
    );

    seed_smoothed_voltage(voltage_state, current_voltage_mv);

    ChargeCompensation no_extra_comp;

    const int corrected_voltage_mv =
        correct_voltage_for_percent(
            current_voltage_mv,
            anchors,
            charging,
            no_extra_comp
        );

    seed_smoothed_voltage(
        percent_voltage_state,
        corrected_voltage_mv
    );

    return result;
}

// Reconcile the saved percentage with the current startup state.
static StartupInitResult init_voltage_startup_state(
    const BatteryPaths& battery,
    const VoltageAnchors& anchors,
    VoltageFilterState& voltage_state,
    VoltageFilterState& percent_voltage_state,
    int& internal_percent,
    int& visible_percent
) {
    StartupInitResult result;

    ChargeStatus startup_status = read_charge_status(battery.status);

    const int v_now = burst_sample_voltage(battery.voltage_now);

    // Re-read status after sampling in case it changed during the burst.
    const ChargeStatus settled_status = read_charge_status(battery.status);

    if (settled_status != ChargeStatus::Unknown) {
        startup_status = settled_status;
    }

    result.restore_charging = is_charging_status(startup_status);

    const bool status_full = startup_status == ChargeStatus::Full;

    const auto saved = consume_restore_state();

    if (v_now <= 0) {
        if (saved) {
            DEBUG_LOG(
                "startup_restore",
                "result=skipped,reason=voltage_read_failed,"
                "saved_mv=%d,saved_percent=%d",
                saved->saved_mv,
                saved->visible_percent
            );
        }

        return result;
    }

    // Boot compensation only affects the fresh calculation.
    // Restore confidence compares the startup voltage with the saved voltage.
    int fresh_voltage_mv = v_now;

    if (startup_status == ChargeStatus::Discharging &&
        boottime_s() < BOOT_COMP_MAX_UPTIME_S) {
        fresh_voltage_mv += BOOT_COMP_MV;
    }

    seed_smoothed_voltage(voltage_state, v_now);

    ChargeCompensation no_extra_comp;

    const int corrected_fresh_voltage_mv =
        correct_voltage_for_percent(
            fresh_voltage_mv,
            anchors,
            result.restore_charging,
            no_extra_comp
        );

    seed_smoothed_voltage(
        percent_voltage_state,
        corrected_fresh_voltage_mv
    );

    const int fresh_percent = calculate_restore_fresh_percent(
        fresh_voltage_mv,
        anchors,
        result.restore_charging,
        status_full
    );

    // Internal percent always uses the fresh calculation.
    internal_percent = fresh_percent;

    // PMIC Full at startup is authoritative.
    if (status_full) {
        visible_percent = 100;
        result.force_visible_write = true;
        return result;
    }

    // Prefer a fresh calculation near V_EMPTY, bypassing a potentially stale restore.
    if (startup_status == ChargeStatus::Discharging && v_now <= g_cfg.V_EMPTY + BOOT_EMPTY_FRESH_MARGIN_MV) {
        return result;
    }

    if (startup_status == ChargeStatus::Unknown || !saved) {
        if (saved) {
            DEBUG_LOG(
                "startup_restore",
                "result=skipped,reason=status_unknown,"
                "saved_mv=%d,current_mv=%d,saved_percent=%d",
                saved->saved_mv,
                v_now,
                saved->visible_percent
            );
        }

        return result;
    }

    const RestoreDecision decision = reconcile_restore_percent(
        *saved,
        v_now,
        result.restore_charging,
        fresh_percent
    );

    DEBUG_LOG(
        "startup_restore",
        "previous=%s,current=%s,"
        "saved_mv=%d,current_mv=%d,delta_mv=%d,"
        "saved_percent=%d,fresh_percent=%d,"
        "mode=%s,saved_weight=%.3f,result_percent=%d",
        saved->charging ? "Charging" : "Discharging",
        charge_status_arg(startup_status),
        saved->saved_mv,
        v_now,
        v_now - saved->saved_mv,
        saved->visible_percent,
        fresh_percent,
        restore_mode_arg(decision.mode),
        decision.saved_weight,
        decision.visible_percent
    );

    // A fresh result is published normally by the first main-loop update.
    if (decision.mode == RestoreMode::Fresh) {
        return result;
    }

    visible_percent = decision.visible_percent;

    // Always publish the restored percentage even if an older output file exists.
    result.force_visible_write = true;

    return result;
}

// ========================= Main =========================
int main() {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    std::signal(SIGUSR1, SIG_IGN); // Ignore for now.

    // Hooks are fire-and-forget, so reap children automatically.
    std::signal(SIGCHLD, SIG_IGN);

    fs::create_directories(fs::path(ROOT));

    ensure_config();

    // Load system config, then any persistent override.
    g_cfg = read_config();

    if (g_cfg.data_dir.empty()) {
        std::fprintf(stderr,
            "batteryplus: Error: unable to resolve persistent data directory\n");
        return 1;
    }

    g_paths.data_dir = g_cfg.data_dir;
    g_paths.map_file = g_paths.data_dir / MAP_FILENAME;
    g_paths.calibrated_flag = g_paths.data_dir / "batteryplus-calibrated";
    g_paths.restore_state_file = g_paths.data_dir / RESTORE_STATE_FILENAME;
    g_paths.debug_log_file = g_paths.data_dir / DEBUG_LOG_FILENAME;

    std::error_code data_dir_ec;
    fs::create_directories(g_paths.data_dir, data_dir_ec);

    if (data_dir_ec) {
        std::fprintf(stderr,
            "batteryplus: Error: unable to create persistent data directory\n");
        return 1;
    }

    HookCache hooks;
    load_hook_cache(hooks);

    auto battery_opt = find_battery();
    if (!battery_opt) {
        std::fprintf(stderr, "batteryplus: Error: No battery detected!\n");
        return 1;
    }
    BatteryPaths battery = *battery_opt;

    if (g_cfg.mode == BatteryMode::Voltage) {
        if (battery.voltage_now.empty()) {
            std::fprintf(stderr, "batteryplus: Error: No battery voltage path detected!\n");
            return 1;
        }
    } else if (g_cfg.mode == BatteryMode::Pmic) {
        if (battery.capacity.empty()) {
            std::fprintf(stderr, "batteryplus: Error: No PMIC capacity path detected!\n");
            return 1;
        }
    }

    bool valid_saved_pair = false;
    VoltageAnchors anchors = load_map(g_paths.map_file, valid_saved_pair);

    VoltageFilterState voltage_state; // raw physical voltage
    VoltageFilterState percent_voltage_state; // charging-corrected voltage
    int internal_percent = -1; // current calculated percentage
    int visible_percent = -1; // percentage exposed to the UI
    FullLearningState full_learning;
    CalibrationState calibration;
    calibration.enabled = !fs::exists(g_paths.calibrated_flag);
    ChargeCompensation charge_compensation;
    UnplugLearnState unplug_learn;

    // Restore previous UI state if it still matches startup conditions.
    bool restore_charging = false;
    bool force_startup_visible_write = false;

    if (g_cfg.mode == BatteryMode::Voltage) {
        StartupInitResult startup_init = init_voltage_startup_state(
            battery,
            anchors,
            voltage_state,
            percent_voltage_state,
            internal_percent,
            visible_percent
        );

        restore_charging = startup_init.restore_charging;
        force_startup_visible_write =
            startup_init.force_visible_write;
    }

    // Seed the initial status; later changes have to pass confirmation.
    ChargeStatusTracker charge_status_tracker;
    charge_status_tracker.confirmed = read_charge_status(battery.status);

    if (charge_status_tracker.confirmed == ChargeStatus::Unknown) {
        charge_status_tracker.confirmed = restore_charging
            ? ChargeStatus::Charging
            : ChargeStatus::Discharging;
    }

    write_status_file(charge_status_tracker.confirmed);

    if (g_cfg.debug) {
        if (g_cfg.mode == BatteryMode::Voltage) {
            const int raw_voltage_mv =
                read_voltage_mv(battery.voltage_now);

            DEBUG_LOG(
                "startup",
                "mode=voltage,status=%s,"
                "raw_mv=%d,ema_mv=%d,"
                "internal_percent=%d,visible_percent=%d,"
                "V_EMPTY=%d,V_FULL_CHG=%d,V_FULL_DIS=%d,"
                "calibrated=%d",
                charge_status_arg(charge_status_tracker.confirmed),
                raw_voltage_mv,
                static_cast<int>(std::lround(voltage_state.ema)),
                internal_percent,
                visible_percent,
                g_cfg.V_EMPTY,
                anchors.V_FULL_CHG,
                anchors.V_FULL_DIS,
                calibration.enabled ? 0 : 1
            );
        } else {
            const auto capacity = slurp_int(battery.capacity);

            DEBUG_LOG(
                "startup",
                "mode=pmic,status=%s,capacity=%d,visible_percent=%d",
                charge_status_arg(charge_status_tracker.confirmed),
                capacity.value_or(-1),
                visible_percent
            );
        }
    }

    bool first_direction_hook = true;
    bool first_state_hook = true;
    ChargeStatus last_charge_status = charge_status_tracker.confirmed;

    // Check boot time against monotonic time to see if we resumed from suspend
    auto last_visible_write = std::chrono::steady_clock::now();
    int64_t last_loop_bt_ms = clock_ms(CLOCK_BOOTTIME);
    int64_t last_loop_mono_ms = clock_ms(CLOCK_MONOTONIC);

    long gap_s = 0;

    bool resume_restore_pending = false;
    RestoreState resume_restore_saved;

    int calc_tick = INTERNAL_INTERVAL_S;
    while (g_running) {
        LoopState loop;

        auto now = std::chrono::steady_clock::now();

        const int64_t now_bt_ms = clock_ms(CLOCK_BOOTTIME);
        const int64_t now_mono_ms = clock_ms(CLOCK_MONOTONIC);

        int64_t bt_gap_ms = 0;
        int64_t mono_gap_ms = 0;

        if (now_bt_ms >= 0 && last_loop_bt_ms >= 0) {
            bt_gap_ms =
                std::max<int64_t>(0, now_bt_ms - last_loop_bt_ms);
        }

        if (now_mono_ms >= 0 && last_loop_mono_ms >= 0) {
            mono_gap_ms =
                std::max<int64_t>(0, now_mono_ms - last_loop_mono_ms);
        }

        const int64_t suspend_gap_ms =
            std::max<int64_t>(0, bt_gap_ms - mono_gap_ms);

        // Keep the normal loop gap for existing interruption/reset logic.
        gap_s = static_cast<long>(bt_gap_ms / 1000);

        const int64_t now_bt_s =
            now_bt_ms >= 0
                ? now_bt_ms / 1000
                : boottime_s();

        last_loop_bt_ms = now_bt_ms;
        last_loop_mono_ms = now_mono_ms;

        loop.resume_detected =
            suspend_gap_ms >=
            static_cast<int64_t>(RESUME_SUSPEND_GAP_S) * 1000;

        if (loop.resume_detected) {
            DEBUG_LOG(
                "resume",
                "trigger=clock_delta,gap_s=%ld,suspend_gap_ms=%lld,"
                "previous_status=%s,"
                "internal_percent=%d,visible_percent=%d,ema_mv=%d",
                gap_s,
                static_cast<long long>(suspend_gap_ms),
                charge_status_arg(last_charge_status),
                internal_percent,
                visible_percent,
                static_cast<int>(std::lround(voltage_state.ema))
            );
        }

        // Read status now, but don't accept a change until it stays stable.
        loop.observed_status = read_charge_status(battery.status);

        update_confirmed_charge_status(
            charge_status_tracker,
            loop.observed_status,
            now,
            loop.resume_detected
        );

        loop.status = charge_status_tracker.confirmed;

        loop.first_visible = (visible_percent < 0);

        if (loop.resume_detected) {
            reset_full_learn_state(full_learning, "resume");

            if (g_cfg.mode == BatteryMode::Voltage) {
                // Save the pre-resume state before current voltage changes it.
                resume_restore_saved.charging = is_charging_status(last_charge_status);
                resume_restore_saved.saved_mv =
                    voltage_state.prev1 > 0
                        ? voltage_state.prev1
                        : static_cast<int>(std::lround(voltage_state.ema));
                resume_restore_saved.visible_percent = visible_percent;

                resume_restore_pending = true;
            }
        }

        // Wait for any pending state change to confirm before choosing a restore window.
        if (resume_restore_pending &&
            loop.observed_status != ChargeStatus::Unknown &&
            charge_status_tracker.candidate == ChargeStatus::Unknown) {
            loop.resume_restore_now = true;
            resume_restore_pending = false;
        }

        // Keep the confirmed transition part of resume handling.
        loop.resume_transition = loop.resume_detected || loop.resume_restore_now;

        // Preserve the discharge voltage while a possible plug-in is confirmed.
        // Resume also waits for confirmation before updating voltage or percent.
        loop.hold_voltage = g_cfg.mode == BatteryMode::Voltage &&
            (resume_restore_pending ||
             (loop.status == ChargeStatus::Discharging &&
              is_charging_status(charge_status_tracker.candidate)));

        if (loop.hold_voltage) {
            loop.visible_policy = VisiblePolicy::Hold;
        }

        if (loop.resume_restore_now) {
            const ResumeRestoreResult restored =
                restore_voltage_on_resume(
                    battery,
                    anchors,
                    voltage_state,
                    percent_voltage_state,
                    resume_restore_saved,
                    loop.status
                );

            loop.visible_policy = restored.visible_policy;
            loop.resume_visible_percent = restored.visible_percent;

            // Resume restore ignores session compensation; reapply it afterward.
            if (loop.charging() &&
                has_established_charge_compensation(charge_compensation)) {
                const int current_voltage_mv = voltage_state.prev1;

                if (current_voltage_mv > 0) {
                    const int corrected_voltage_mv =
                        correct_voltage_for_percent(
                            current_voltage_mv,
                            anchors,
                            true,
                            charge_compensation
                        );

                    seed_smoothed_voltage(
                        percent_voltage_state,
                        corrected_voltage_mv
                    );

                    DEBUG_LOG(
                        "charge_compensation",
                        "result=resume_reseeded,"
                        "raw_mv=%d,corrected_mv=%d",
                        current_voltage_mv,
                        corrected_voltage_mv
                    );
                }
            }
        }

        loop.charging_changed = loop.charging() != is_charging_status(last_charge_status);
        loop.status_changed = loop.status != last_charge_status;

        if (loop.status_changed) {
            write_status_file(loop.status);

            if (g_cfg.mode == BatteryMode::Voltage) {
                DEBUG_LOG(
                    "status_change",
                    "previous=%s,current=%s,observed=%s,"
                    "raw_mv=%d,ema_mv=%d,"
                    "internal_percent=%d,visible_percent=%d",
                    charge_status_arg(last_charge_status),
                    charge_status_arg(loop.status),
                    charge_status_arg(loop.observed_status),
                    read_voltage_mv(battery.voltage_now),
                    static_cast<int>(std::lround(voltage_state.ema)),
                    internal_percent,
                    visible_percent
                );
            } else {
                DEBUG_LOG(
                    "status_change",
                    "previous=%s,current=%s,observed=%s,"
                    "capacity=%d,internal_percent=%d,visible_percent=%d",
                    charge_status_arg(last_charge_status),
                    charge_status_arg(loop.status),
                    charge_status_arg(loop.observed_status),
                    slurp_int(battery.capacity).value_or(-1),
                    internal_percent,
                    visible_percent
                );
            }
        }

        if (g_cfg.mode == BatteryMode::Voltage && !loop.hold_voltage) {
            if (update_unplug_full_learning(
                    battery,
                    anchors,
                    voltage_state,
                    percent_voltage_state,
                    visible_percent,
                    full_learning,
                    calibration,
                    unplug_learn,
                    loop.status,
                    loop.charging_changed,
                    loop.resume_transition,
                    now_bt_s,
                    gap_s)) {
                loop.recalculate = true;
            }

            if (update_charge_compensation(
                    battery,
                    anchors,
                    voltage_state,
                    percent_voltage_state,
                    charge_compensation,
                    loop.charging(),
                    loop.charging_changed,
                    last_charge_status,
                    loop.resume_transition,
                    visible_percent,
                    now_bt_s,
                    gap_s)) {
                loop.recalculate = true;
            }
        }

        loop.recalculate =
            !loop.hold_voltage &&
            !(g_cfg.mode == BatteryMode::Voltage &&
            charge_compensation.sample_count >= 0) &&
            (
                loop.recalculate ||
                loop.first_visible ||
                loop.resume_detected ||
                loop.resume_restore_now ||
                loop.charging_changed ||
                loop.status_changed ||
                calc_tick >= INTERNAL_INTERVAL_S
            );

        if (loop.recalculate) {
            calc_tick = 0;

            PercentResult percent_result;

            if (g_cfg.mode == BatteryMode::Voltage) {
                percent_result = run_voltage_mode(
                    battery,
                    anchors,
                    voltage_state,
                    percent_voltage_state,
                    full_learning,
                    loop.charging(),
                    charge_compensation,
                    loop.full(),
                    now_bt_s
                );
            } else {
                percent_result = run_pmic_mode(
                    battery.capacity
                );
            }

            if (percent_result.valid) {
                internal_percent = percent_result.percent;
            }

            // Wait for a valid initial percentage before publishing or running hooks.
            if (internal_percent < 0) {
                DEBUG_LOG(
                    "initial_percent",
                    "result=deferred,reason=reading_unavailable"
                );

                last_charge_status = loop.status;
                if (!sleep_interruptible_s(1)) {
                    break;
                }
                ++calc_tick;
                continue;
            }

            int visible_target = internal_percent;

            const bool full_for_pacing =
                loop.full() ||
                (g_cfg.mode == BatteryMode::Voltage &&
                 loop.charging() && full_learning.full_event_active);

            bool percent_file_exists = fs::exists(PERCENT_FILE);

            bool need_visible_update =
                force_startup_visible_write ||
                loop.resume_visible_percent.has_value() ||
                !percent_file_exists || (
                    loop.visible_policy != VisiblePolicy::Hold &&
                    check_update_visible_percent(
                        loop.first_visible,
                        loop.visible_policy == VisiblePolicy::Snap,
                        loop.resume_detected || loop.resume_restore_now,
                        visible_target,
                        visible_percent,
                        full_for_pacing ? ChargeStatus::Full : loop.status,
                        now,
                        last_visible_write
                    )
                );

            if (need_visible_update) {
                int new_visible;

                if (loop.resume_visible_percent.has_value()) {
                    new_visible = *loop.resume_visible_percent;
                } else if (force_startup_visible_write) {
                    new_visible = visible_percent;
                } else if (loop.visible_policy == VisiblePolicy::Hold && visible_percent >= 0) {
                    new_visible = visible_percent;
                } else {
                    new_visible = compute_new_visible_percent(
                        loop.first_visible,
                        loop.visible_policy == VisiblePolicy::Snap,
                        visible_percent,
                        visible_target,
                        loop.charging()
                    );
                }

                fs::create_directories(
                    fs::path(PERCENT_FILE).parent_path()
                );

                const bool write_ok = write_atomic(
                    PERCENT_FILE,
                    std::to_string(new_visible) + "\n",
                    0644
                );

                if (write_ok) {
                    const bool visible_changed = new_visible != visible_percent;
                    DEBUG_LOG(
                        "visible_update",
                        "reason=%s,status=%s,"
                        "old_percent=%d,new_percent=%d,"
                        "internal_percent=%d,target_percent=%d",
                        loop.resume_visible_percent.has_value() ? "resume_restore" :
                        force_startup_visible_write ? "startup_restore" :
                        !percent_file_exists ? "file_missing" :
                        loop.first_visible ? "initial" :
                        loop.visible_policy == VisiblePolicy::Snap ? "snap" :
                        "scheduled",
                        charge_status_arg(loop.status),
                        visible_percent,
                        new_visible,
                        internal_percent,
                        visible_target
                    );

                    visible_percent = new_visible;
                    last_visible_write = now;

                    if (visible_changed) {
                        run_direction_hooks_cached(
                            hooks,
                            loop.charging(),
                            visible_percent,
                            loop.status
                        );
                        loop.hooks_fired = true;
                    }

                    force_startup_visible_write = false;
                } else {
                    DEBUG_LOG(
                        "visible_update",
                        "result=failed,reason=write_failed,"
                        "attempted_percent=%d,current_percent=%d",
                        new_visible,
                        visible_percent
                    );
                }
            }
        }

        // Run the current direction hooks once for startup, direction changes,
        // or resume when a visible percentage change didn't already run them.
        const bool direction_hook_event =
            first_direction_hook ||
            loop.charging_changed ||
            loop.resume_restore_now ||
            (g_cfg.mode == BatteryMode::Pmic && loop.resume_detected);

        if (!loop.hooks_fired && direction_hook_event && visible_percent >= 0) {
            run_direction_hooks_cached(
                hooks,
                loop.charging(),
                visible_percent,
                loop.status
            );
            loop.hooks_fired = true;
        }

        if (visible_percent >= 0) {
            first_direction_hook = false;
        }

        // State hooks run once at startup and again on confirmed state changes.
        if ((first_state_hook || loop.status_changed) && visible_percent >= 0) {
            run_state_hooks_cached(
                hooks,
                visible_percent,
                loop.status
            );
            first_state_hook = false;
        }

        last_charge_status = loop.status;

        if (g_running) {
            std::this_thread::sleep_for(1s);
            ++calc_tick;
        }
    }

    // Save one-shot state so a clean restart can preserve UI continuity.
    if (g_cfg.mode == BatteryMode::Voltage) {
        // Use the current hardware status at shutdown instead of a state that may
        // still be waiting on the normal confirmation delay.
        const ChargeStatus shutdown_status = read_charge_status(battery.status);
        const int shutdown_voltage_mv = read_voltage_mv(battery.voltage_now);

        bool shutdown_charging = is_charging_status(last_charge_status);

        if (shutdown_status != ChargeStatus::Unknown) {
            shutdown_charging = is_charging_status(shutdown_status);
        }

        RestoreState st;
        st.charging = shutdown_charging;
        st.saved_mv = shutdown_voltage_mv > 0
            ? shutdown_voltage_mv
            : static_cast<int>(std::lround(voltage_state.ema));
        st.visible_percent = visible_percent;

        const bool valid_restore =
            st.saved_mv > 0 &&
            st.visible_percent >= 0 &&
            st.visible_percent <= 100;

        bool restore_saved = false;

        if (valid_restore) {
            restore_saved = save_restore_state(st);
        }

        DEBUG_LOG(
            "shutdown",
            "status=%s,confirmed_status=%s,"
            "raw_mv=%d,ema_mv=%d,saved_mv=%d,"
            "internal_percent=%d,visible_percent=%d,"
            "restore_valid=%d,restore_saved=%d",
            charge_status_arg(shutdown_status),
            charge_status_arg(last_charge_status),
            shutdown_voltage_mv,
            static_cast<int>(std::lround(voltage_state.ema)),
            st.saved_mv,
            internal_percent,
            visible_percent,
            valid_restore ? 1 : 0,
            restore_saved ? 1 : 0
        );
    } else {
        DEBUG_LOG(
            "shutdown",
            "status=%s,capacity=%d,"
            "internal_percent=%d,visible_percent=%d",
            charge_status_arg(read_charge_status(battery.status)),
            slurp_int(battery.capacity).value_or(-1),
            internal_percent,
            visible_percent
        );
    }

    return 0;
}
