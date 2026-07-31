// BatteryPlus — battery percentage daemon for handheld Linux systems
//
// Copyright (c) 2025 Mikhailzrick
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2
// as published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License v2
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
//
// Purpose:
//   BatteryPlus is an alternative battery reporting daemon designed for
//   handheld Linux systems whose built-in PMIC battery percentage is
//   inaccurate, unstable, or just poorly implemented.
//
//   In voltage mode, BatteryPlus derives percent from battery voltage using
//   median-of-3 filtering, EMA smoothing, shaped charge/discharge curves,
//   step-limited visible output, and automatic full-voltage learning.
//
//   In PMIC mode, BatteryPlus reads the PMIC-reported capacity directly while
//   still using the same visible-output, step-limiting, hook, and daemon
//   infrastructure.
//
// Core behavior:
//
//   • Voltage mode
//       - Reads /sys/class/power_supply/*/voltage_now
//       - Converts smoothed voltage to percent using separate charge and
//         discharge curves
//       - Uses separate empty anchors for charging and discharging
//       - Learns separate full anchors:
//             V_FULL_CHG: peak/full voltage while charging
//             V_FULL_DIS: settled voltage shortly after unplugging from full
//       - Caps charging display at 99% until a full event is detected
//
//   • PMIC mode
//       - Reads /sys/class/power_supply/*/capacity
//       - Reuses BatteryPlus output and hook handling
//
//   • Smoothing and startup behavior
//       - Uses median-of-3 plus EMA smoothing for voltage noise reduction
//       - Optionally restores previous EMA and visible percent after daemon
//         restart if voltage and charge state still match
//
//   • Visible percent behavior
//       - Internal percent is recalculated every INTERNAL_INTERVAL_S
//       - Visible percent is written to /tmp/battery.percent less often
//       - Larger internal/visible deltas shorten the write interval
//       - Percent changes are step-limited unless a long resume gap triggers
//         a snap update.
//
//   • Resume handling
//       - SIGUSR1 acts as an explicit resume hint
//       - Long loop gaps are treated as resume/suspend gaps
//       - In-progress full-voltage calibration is aborted after resume
//       - Long gaps may burst-sample voltage and snap visible percent
//
//   • Hook system
//       - /etc/batteryplus/charging.d/
//       - /etc/batteryplus/discharging.d/
//       - /etc/batteryplus/state.d/
//       - Bucket hooks run on exact 5% visible-percent transitions
//       - Non-numeric hook names in charging.d/discharging.d act as wildcards
//       - Hooks receive:
//             $1 = visible percent
//             $2 = charge state string
//
// Files:
//   /tmp/battery.percent
//       Exported visible battery percent for UI polling.
//
//   <data_dir>/batteryplus-voltage.map
//       Stores learned voltage anchors:
//             V_FULL_CHG
//             V_FULL_DIS
//
//   <data_dir>/batteryplus-calibrated
//       Presence-only flag created after both charge-side and discharge-side
//       full anchors have been learned in the current calibration flow.
//
//   <data_dir>/batteryplus-restore.state
//       Temporary restore file written on clean daemon exit and consumed on
//       next startup if voltage/state still match.
//
// Config:
//   /etc/batteryplus/batteryplus.conf
//
//   Data directory:
//       data_dir=<absolute persistent directory>
//
//       If data_dir is blank, BatteryPlus uses:
//           $XDG_CONFIG_HOME/batteryplus
//       or:
//           $HOME/.config/batteryplus
//
//       <data_dir>/batteryplus.conf may override configuration settings (except data_dir).
//
//   Optional:
//       mode=voltage|pmic
//       V_EMPTY_CHG=<mV> (Default: 3400)
//       V_EMPTY_DIS=<mV> (Default: 3250)
//
// Signals:
//   SIGTERM / SIGINT
//       Stop daemon and save restore state when possible.
//
//   SIGUSR1
//       Resume hint. Forces an immediate recalculation path, aborts active
//       calibration state, and may trigger wildcard hooks if no bucket hook
//       already ran.
//
// Build:
//   aarch64 cross-build: aarch64-linux-gnu-g++ -O3 -flto -std=gnu++20 -Wall -Wextra -pedantic batteryplus.cpp -o batteryplus
//   native x86-64: g++ -O3 -flto -std=gnu++20 -Wall -Wextra -pedantic batteryplus.cpp -o batteryplus

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <csignal>
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
static constexpr const char* ROOT = "/etc/batteryplus"; // hook/config root: charging.d, discharging.d, state.d

// Timers / Thresholds / Parameters
static constexpr int INTERNAL_INTERVAL_S = 15; // how often internal calculations are done in seconds
static constexpr int WRITE_INTERVAL_S = 120; // how often visible percent is written to the battery percent file in seconds
static constexpr int WRITE_INTERVAL_DELTA_SMALL_S = 60; // how often visible percent updates(in seconds) when there's a small delta
static constexpr int WRITE_INTERVAL_DELTA_LARGE_S = 30; // how often visible percent updates(in seconds) when there's a large delta
static constexpr int PEAK_DWELL_S = 30 * 60; // seconds spent with no new peak before full is determined. Fallback in case full never reported.
static constexpr int PEAK_STABILITY_WINDOW_MV = 20; // abort peak-based calibration if charging EMA drops more than this below the observed peak
static constexpr int VFULL_DIS_SETTLE_S = 15; // settle time in seconds after unplug before recording V_FULL_DIS
static constexpr int PEAK_TRACK_START_MV = 4000; // voltage threshold to begin top-of-charge tracking(in mv)
static constexpr int MIN_RANGE_MV = 400; // minimum usable voltage span between empty and full(in mv)
static constexpr int DISCHARGE_100_WINDOW_MV = 20; // reports 100% when discharging if within this range (in mv) from learned V_FULL_DIS

// Startup restore voltage windows (mv)
static constexpr int RESTORE_CHG_DOWN_MV = 50;
static constexpr int RESTORE_CHG_UP_MV = 100;
static constexpr int RESTORE_DIS_DOWN_MV = 50;
static constexpr int RESTORE_DIS_UP_MV = 25;

// Startup discharge compensation
static constexpr int BOOT_COMP_MAX_UPTIME_S = 60;
static constexpr int BOOT_COMP_MV = 15;

// EMA parameters
static constexpr int ALPHA_NUM = 2;
static constexpr int ALPHA_DEN = 10;

// Defaults for map if missing
static constexpr int DEFAULT_V_FULL_CHG = 4050;// mV (absolute voltage ceiling when charging)
static constexpr int DEFAULT_V_FULL_DIS = 4000; // mV (100% anchor while discharging)
static constexpr int DEFAULT_V_EMPTY_CHG = 3400; // mV (0% anchor while charging)
static constexpr int DEFAULT_V_EMPTY_DIS = 3250; // mV (0% anchor when discharging)

// Mode parameters
enum class BatteryMode {
    Voltage,
    Pmic
};

// ========================= States =========================
struct ConfigVals {
    BatteryMode mode = BatteryMode::Voltage;
    fs::path data_dir;
    int V_EMPTY_DIS = DEFAULT_V_EMPTY_DIS;
    int V_EMPTY_CHG = DEFAULT_V_EMPTY_CHG;
};

// Tracks whether both full anchors were learned during this session.
// Used only to create the calibrated flag file.
struct CalibrationState {
    bool learned_vfull_chg = false;
    bool learned_vfull_dis = false;
};

// One-shot startup restore state.
// Written on clean exit, consumed and deleted on next daemon start.
struct RestoreState {
    bool charging = false;
    int ema_mv = -1;
    int visible_percent = -1;
};

struct StartupInitResult {
    bool restore_charging = false;
    bool restored_state = false;
};

struct StatePaths {
    fs::path data_dir;
    fs::path map_file;
    fs::path calibrated_flag;
    fs::path restore_state_file;
};

// ========================= Globals =========================
static std::atomic<bool> g_running { true };
static std::atomic<bool> g_resume_hint{false};
static StatePaths g_paths;
static ConfigVals g_cfg;

// ========================= Utilities =========================
static void handle_signal(int) {
    g_running = false;
}

static void handle_resume_signal(int) {
    g_resume_hint = true;
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

static std::optional<std::string> slurp(const fs::path& p) {
    std::ifstream f(p);
    if (!f) return std::nullopt;
    std::string s;
    std::getline(f, s);
    if (!f && !f.eof()) return std::nullopt;
    // strip CR/LF/space
    while (!s.empty() && (s.back()=='\n' || s.back()=='\r' || s.back()==' ' || s.back()=='\t')) s.pop_back();
    return s;
}

static std::optional<int> slurp_int(const fs::path& p) {
    auto s = slurp(p);
    if (!s) return std::nullopt;
    char* end=nullptr;
    long v = std::strtol(s->c_str(), &end, 10);
    if (end==s->c_str()) return std::nullopt;
    return static_cast<int>(v);
}

static bool write_atomic(const fs::path& path, const std::string& data, mode_t mode = 0644) {
    fs::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) return false;
        f << data;
        if (!f.good()) return false;
    }
    ::chmod(tmp.c_str(), mode);
    return ::rename(tmp.c_str(), path.c_str()) == 0;
}

static bool save_restore_state(const RestoreState& st) {
    std::string data;
    data += "charging=" + std::to_string(st.charging ? 1 : 0) + "\n";
    data += "ema_mv=" + std::to_string(st.ema_mv) + "\n";
    data += "visible_percent=" + std::to_string(st.visible_percent) + "\n";
    return write_atomic(g_paths.restore_state_file, data, 0644);
}

static std::optional<RestoreState> consume_restore_state() {
    RestoreState st;
    bool saw_charging = false;
    bool saw_ema = false;
    bool saw_visible = false;

    std::ifstream f(g_paths.restore_state_file);
    if (f) {
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("charging=", 0) == 0) {
                st.charging = (std::atoi(line.c_str() + 9) != 0);
                saw_charging = true;
            } else if (line.rfind("ema_mv=", 0) == 0) {
                st.ema_mv = std::atoi(line.c_str() + 7);
                saw_ema = true;
            } else if (line.rfind("visible_percent=", 0) == 0) {
                st.visible_percent = std::atoi(line.c_str() + 16);
                saw_visible = true;
            }
        }
    }

    // One-shot restore: delete it even if validation later rejects it.
    std::error_code ec;
    fs::remove(g_paths.restore_state_file, ec);

    if (!(saw_charging && saw_ema && saw_visible)) {
        return std::nullopt;
    }

    return st;
}

// Create the calibrated flag only after both charge-side and discharge-side
// full anchors have been learned in the current calibration flow.
static void create_calibrated_flag(const CalibrationState& calib) {
    if (!(calib.learned_vfull_chg && calib.learned_vfull_dis)) {
        return;
    }

    if (fs::exists(g_paths.calibrated_flag)) {
        return;
    }

    std::ofstream f(g_paths.calibrated_flag, std::ios::trunc);
    if (f) {
        f << "";
    }
}

// Returns the leading numeric prefix if present.
// Non-numeric names return -1 and are treated as wildcard hooks.
static int parse_leading_bucket(const std::string& fname) {
    if (fname.empty() || !std::isdigit((unsigned char)fname[0])) return -1;
    int i = 0, v = 0;
    while (i < (int)fname.size() && std::isdigit((unsigned char)fname[i]) && i < 3) {
        v = v * 10 + (fname[i] - '0');
        ++i;
    }
    return (v >= 0 && v <= 100) ? v : -1;
}

static int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static void to_lower_inplace(char* s) {
    for (; *s; ++s) {
        *s = static_cast<char>(std::tolower((unsigned char)*s));
    }
}

static int median3(int a, int b, int c) {
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
    return b;
}

enum class ChargeStatus {
    Unknown,
    Charging,
    Discharging,
    Full
};

static ChargeStatus read_charge_status(const fs::path& status_path) {
    auto s = slurp(status_path);
    if (!s) return ChargeStatus::Unknown;

    if (s->rfind("Charging", 0) == 0) return ChargeStatus::Charging;
    if (s->rfind("Discharging", 0) == 0) return ChargeStatus::Discharging;
    if (s->rfind("Full", 0) == 0) return ChargeStatus::Full;
    return ChargeStatus::Unknown;
}

static const char* charge_status_arg(ChargeStatus st) {
    switch (st) {
        case ChargeStatus::Charging: return "Charging";
        case ChargeStatus::Discharging: return "Discharging";
        case ChargeStatus::Full: return "Full";
        default: return "";
    }
}

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
        "V_EMPTY_CHG=%d\n"
        "V_EMPTY_DIS=%d\n",
        DEFAULT_V_EMPTY_CHG,
        DEFAULT_V_EMPTY_DIS
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

        // Strip leading spaces.
        while (*p == ' ' || *p == '\t') {
            ++p;
        }

        // Skip blanks, comments, and section headers.
        if (*p == '\0' || *p == '\n' || *p == '#' || *p == '[') {
            continue;
        }

        // key=value
        char* eq = std::strchr(p, '=');
        if (!eq) {
            continue;
        }

        *eq = '\0';

        char* key = p;
        char* val = eq + 1;

        // Trim trailing key whitespace.
        for (char* t = key + std::strlen(key);
             t > key &&
             (t[-1] == ' ' || t[-1] == '\t' ||
              t[-1] == '\r' || t[-1] == '\n');
             --t) {
            t[-1] = '\0';
        }

        // Trim leading value whitespace.
        while (*val == ' ' || *val == '\t') {
            ++val;
        }

        // Trim trailing value whitespace.
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

        } else if (std::strcmp(key, "V_EMPTY_DIS") == 0) {
            char* end = nullptr;
            long v = std::strtol(val, &end, 10);

            if (end != val && *end == '\0') {
                cfg.V_EMPTY_DIS = static_cast<int>(v);
            }

        } else if (std::strcmp(key, "V_EMPTY_CHG") == 0) {
            char* end = nullptr;
            long v = std::strtol(val, &end, 10);

            if (end != val && *end == '\0') {
                cfg.V_EMPTY_CHG = static_cast<int>(v);
            }
        }
    }

    std::fclose(f);
}

static void validate_config(
    ConfigVals& cfg,
    const ConfigVals& fallback
) {
    // Invalid values fall back to the preceding configuration layer.
    if (cfg.V_EMPTY_DIS < 3000 || cfg.V_EMPTY_DIS > 3400) {
        cfg.V_EMPTY_DIS = fallback.V_EMPTY_DIS;
    }

    if (cfg.V_EMPTY_CHG < 3300 || cfg.V_EMPTY_CHG > 3600) {
        cfg.V_EMPTY_CHG = fallback.V_EMPTY_CHG;
    }

    // Reject an invalid combined pair from the new layer.
    if (cfg.V_EMPTY_CHG <= cfg.V_EMPTY_DIS) {
        cfg.V_EMPTY_CHG = fallback.V_EMPTY_CHG;
        cfg.V_EMPTY_DIS = fallback.V_EMPTY_DIS;
    }
}

static ConfigVals read_config() {
    ConfigVals cfg;
    const ConfigVals defaults;

    // Load and validate the base system configuration.
    read_config_file(CONFIG_FILE, cfg, true);
    validate_config(cfg, defaults);

    // Only accept an absolute data_dir from the base configuration.
    if (!cfg.data_dir.empty() && !cfg.data_dir.is_absolute()) {
        cfg.data_dir.clear();
    }

    // If data_dir was not set, use the standard user config location.
    if (cfg.data_dir.empty()) {
        cfg.data_dir = resolve_default_data_dir();
    }

    if (cfg.data_dir.empty()) {
        return cfg;
    }

    // Preserve the validated base layer so invalid overrides can be ignored.
    const ConfigVals base_cfg = cfg;

    // Load persistent overrides. Ignore data_dir if present.
    read_config_file(
        cfg.data_dir / OVERRIDE_CONFIG_FILENAME,
        cfg,
        false
    );

    // Invalid override values fall back to the validated base layer.
    validate_config(cfg, base_cfg);

    return cfg;
}

// ========================= Hook System =========================
// Hook directories:
//   charging.d / discharging.d:
//     Numeric prefixes select exact 5% buckets, e.g. 50, 050, 50-low-power.
//     Non-numeric filenames are wildcards and run on every bucket/resume event.
//   state.d:
//     All executable files run when charge status changes.
//
// Hooks receive:
//   $1 = visible percent
//   $2 = charge state string

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
            pct.c_str(), // $1 = visible percent
            state, // $2 = charge state
            nullptr
        };
        ::execv(argv[0], (char* const*)argv);
        _exit(127);
    }

    // fire-and-forget
    return 0;
}

static constexpr int NUM_BUCKETS = 21; // 0 -> 100 in 5% increments

struct HookCache {
    std::array<std::vector<fs::path>, NUM_BUCKETS> charging;
    std::vector<fs::path> charging_any;
    std::array<std::vector<fs::path>, NUM_BUCKETS> discharging;
    std::vector<fs::path> discharging_any;
    std::vector<fs::path> state;
    bool loaded = false;
};

static int bucket5(int percent) {
    percent = clampi(percent, 0, 100);
    return (percent / 5) * 5;
}

static int bucket_index(int percent) {
    return bucket5(percent) / 5;
}

static void scan_hook_dir(const fs::path& dir, std::array<std::vector<fs::path>, NUM_BUCKETS>& buckets, std::vector<fs::path>& wildcards)
{
    if (!fs::exists(dir) || !fs::is_directory(dir)) return;

    for (auto& de : fs::directory_iterator(dir)) {
        if (!is_executable(de)) continue;
        const std::string fname = de.path().filename().string();

        int n = parse_leading_bucket(fname); // numeric prefix
        if (n >= 0 && n <= 100 && n % 5 == 0) {
            int bi = bucket_index(n);
            buckets[bi].push_back(de.path());
        } else if (n < 0) {
            wildcards.push_back(de.path()); // non-numeric: wildcard
        } else {
            // Numeric hook names must target exact 5% buckets.
            continue;
        }
    }

    auto sorter = [](auto& v){ std::sort(v.begin(), v.end()); };
    for (auto& v : buckets) sorter(v);
    sorter(wildcards);
}

static void scan_hook_dir_all(const fs::path& dir, std::vector<fs::path>& paths) {
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
    scan_hook_dir(fs::path(ROOT) / "charging.d",    hc.charging,    hc.charging_any);
    scan_hook_dir(fs::path(ROOT) / "discharging.d", hc.discharging, hc.discharging_any);
    scan_hook_dir_all(fs::path(ROOT) / "state.d",   hc.state);
    hc.loaded = true;
}

static void run_paths(const std::vector<fs::path>& paths, int visible_percent, ChargeStatus status) {
    for (const auto& p : paths) {
        if (::access(p.c_str(), X_OK) == 0)
            run_hook_file(p, visible_percent, status);
    }
}

static void run_bucket_hooks_cached(const HookCache& hc, bool charging, int percent_value, ChargeStatus status) {
    if (!hc.loaded) return;
    int bi = bucket_index(percent_value);
    const auto& buckets = charging ? hc.charging : hc.discharging;
    const auto& any = charging ? hc.charging_any : hc.discharging_any;

    run_paths(buckets[bi], percent_value, status); // run all scripts for this 5% bucket
    run_paths(any, percent_value, status); // wildcard scripts every change
}

static void run_state_hooks_cached(const HookCache& hc, int visible_percent, ChargeStatus status) {
    if (!hc.loaded) return;
    run_paths(hc.state, visible_percent, status);
}

// Include charge direction in the key so the same bucket can fire again
// after plug/unplug, e.g. 50% discharging -> 50% charging.
static int make_hook_key(bool charging, int bucket_value) {
    return (charging ? 1000 : 0) + bucket_value;
}

static bool check_run_bucket_hooks(
    const HookCache& hooks,
    bool charging,
    int visible_percent,
    ChargeStatus status,
    int& last_hook_key
) {
    if (visible_percent < 0 || (visible_percent % 5) != 0) {
        return false;
    }

    int key = make_hook_key(charging, visible_percent);
    if (key == last_hook_key) {
        return false;
    }

    run_bucket_hooks_cached(hooks, charging, visible_percent, status);
    last_hook_key = key;
    return true;
}

// On resume, run wildcard hooks once if no bucket hook already fired.
static bool check_resume_wildcard_hooks(
    const HookCache& hooks,
    bool explicit_resume,
    bool hooks_fired,
    bool charging,
    int visible_percent,
    ChargeStatus status
) {
    if (!explicit_resume || hooks_fired) {
        return false;
    }

    const auto& any = charging ? hooks.charging_any : hooks.discharging_any;
    run_paths(any, visible_percent, status);
    return true;
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
        return fs::exists(d / "voltage_now") || fs::exists(d / "capacity");
    };

    // Prefer likely battery/fuel-gauge supplies and require either voltage_now
    // or capacity so both voltage and PMIC modes can share discovery.
    std::vector<std::string> patterns = {
        "BAT", "bat", "FUEL", "fuel"
    };

    fs::path base("/sys/class/power_supply");
    if (!fs::exists(base)) return std::nullopt;

    auto build_paths = [](const fs::path& d) -> BatteryPaths {
        BatteryPaths bp;
        bp.base_dir = d;
        bp.status = d / "status";

        if (fs::exists(d / "voltage_now"))
            bp.voltage_now = d / "voltage_now";

        if (fs::exists(d / "capacity"))
            bp.capacity = d / "capacity";

        return bp;
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
struct MapVals {
    int V_FULL_CHG = DEFAULT_V_FULL_CHG;
    int V_FULL_DIS = DEFAULT_V_FULL_DIS;
};

static void save_map_atomic(const fs::path& path, const MapVals& m) {
    std::string data;
    data += "V_FULL_CHG=" + std::to_string(m.V_FULL_CHG) + "\n";
    data += "V_FULL_DIS=" + std::to_string(m.V_FULL_DIS) + "\n";
    fs::create_directories(path.parent_path());
    (void)write_atomic(path, data, 0644);
}

// Reject bad learned anchors and restore defaults rather than
// trusting a stale/corrupt map file.
static MapVals load_map(const fs::path& path) {
    MapVals m;
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
        // No file yet
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

    if (need_save) {
        save_map_atomic(path, m);
    }

    return m;
}

static bool learn_vfull_chg(int voltage_ema_mv, MapVals& map) {
    int candidate_mv = voltage_ema_mv;
    int old_vfull_mv = map.V_FULL_CHG;

    // Reject bad charge-full candidates.
    if (candidate_mv < 3600 || candidate_mv > 4600) {
        return false;
    }

    // Quantize to up to nearest 5 mV so tiny EMA movement does not churn the map file.
    int quantized_mv = ((candidate_mv + 4) / 5) * 5;

    // Only save if meaningfully changed
    if (std::abs(quantized_mv - old_vfull_mv) >= 5) {
        map.V_FULL_CHG = quantized_mv;
        save_map_atomic(g_paths.map_file, map);
    }

    return true;
}

static bool learn_vfull_dis(int voltage_ema_mv, MapVals& map) {
    int candidate_mv = voltage_ema_mv;
    int old_vfull_mv = map.V_FULL_DIS;

    // Reject bad discharge-full candidates.
    if (candidate_mv < 3600 || candidate_mv > 4600) {
        return false;
    }

    // Quantize down to nearest 5 mV so tiny EMA movement does not churn the map file.
    int quantized_mv = (candidate_mv / 5) * 5;

    // Discharge-side full should remain below charge-side full.
    int max_dis_mv = map.V_FULL_CHG - 10;
    if (max_dis_mv < 3600) {
        return false;
    }

    if (quantized_mv > max_dis_mv) {
        quantized_mv = max_dis_mv;
    }

    // Only save if meaningfully changed
    if (std::abs(quantized_mv - old_vfull_mv) >= 5) {
        map.V_FULL_DIS = quantized_mv;
        save_map_atomic(g_paths.map_file, map);
    }

    return true;
}

// ========================= Percent calc =========================
struct SmoothedV {
    int prev1 = -1;
    int prev2 = -1;
    int ema = -1;
};

// Tracks the charge-side full detection flow.
// A full event can come from PMIC status=="Full" or from peak dwell timing.
struct FullLearnState {
    bool tracking_charge_peak = false;
    int peak_charge_ema_mv = -1;

    bool full_event_active = false;

    bool peak_timer_active = false;
    int64_t peak_start_bt_s = 0;

    bool peak_stability_ok = false;
};

static void reset_full_learn_state(FullLearnState& fls) {
    fls.tracking_charge_peak = false;
    fls.peak_charge_ema_mv = -1;
    fls.full_event_active = false;
    fls.peak_timer_active = false;
    fls.peak_start_bt_s = 0;
    fls.peak_stability_ok = false;
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
    // Unit autodetect: >= 100000 => microvolts
    if (raw >= 100000) {
        raw /= 1000; // to mV
    }
    return raw; // mV
}

static int update_smoothed_voltage(int voltage_raw_mv, const MapVals& map, SmoothedV& sv) {
    if (sv.prev1 < 0)
        sv.prev1 = (voltage_raw_mv > 0 ? voltage_raw_mv : map.V_FULL_DIS);
    if (sv.prev2 < 0)
        sv.prev2 = sv.prev1;

    int v_med = median3(
        sv.prev2,
        sv.prev1,
        (voltage_raw_mv > 0 ? voltage_raw_mv : sv.prev1)
    );

    sv.prev2 = sv.prev1;
    sv.prev1 = (voltage_raw_mv > 0 ? voltage_raw_mv : sv.prev1);

    if (sv.ema < 0)
        sv.ema = v_med;
    else
        sv.ema = (ALPHA_NUM * v_med + (ALPHA_DEN - ALPHA_NUM) * sv.ema) / ALPHA_DEN;

    return sv.ema;
}

static int burst_sample_voltage(const fs::path& voltage_now)
{
    int a = read_voltage_mv(voltage_now);

    if (!sleep_interruptible_s(1)) return -1;
    int b = read_voltage_mv(voltage_now);

    if (!sleep_interruptible_s(1)) return -1;
    int c = read_voltage_mv(voltage_now);

    // Fix samples if possible or needed
    if (a <= 0) a = (b > 0 ? b : c);
    if (b <= 0) b = (a > 0 ? a : c);
    if (c <= 0) c = (a > 0 ? a : b);

    if (a <= 0 || b <= 0 || c <= 0) return -1;
    return median3(a, b, c);
}

// Convert voltage to percent using separate charge/discharge behavior:
//   - charging: compressed curve, capped to 99 until full-event logic promotes it
//   - discharging: shaped S-curve with a small 100% plateau near V_FULL_DIS
//
// Full/empty anchors are intentionally different for charge and discharge
// because lithium voltage relaxes downward after unplug.
static int voltage_to_percent(int voltage_now_mv, const MapVals& m, bool charging) {
    if (voltage_now_mv <= 0) {
        // If we somehow get garbage voltage just return 1% so it's intentionally obvious
        return 1;
    }

    const int v_empty = charging ? g_cfg.V_EMPTY_CHG : g_cfg.V_EMPTY_DIS;
    const int v_full  = charging ? m.V_FULL_CHG : m.V_FULL_DIS;

    if (v_full <= v_empty) {
        return 1;
    }

    int v_100_start = v_full;

    if (!charging) {
        v_100_start = v_full - DISCHARGE_100_WINDOW_MV;
    }

    if (v_100_start < v_empty + MIN_RANGE_MV) {
        v_100_start = v_empty + MIN_RANGE_MV;
    }

    if (!charging && voltage_now_mv >= v_100_start) {
        return 100;
    }

    const int v_clamped = clampi(voltage_now_mv, v_empty, v_100_start);
    const int range_adj = v_100_start - v_empty;

    double x = 0.0;
    if (range_adj > 0) {
        x = static_cast<double>(v_clamped - v_empty) / static_cast<double>(range_adj);
    }
    x = std::clamp(x, 0.0, 1.0);

    double shaped = x;

    if (charging) {
        // Charging curve: exponent fades from MAX -> MIN across the range (0% -> 100%)
        constexpr double CHG_EXPONENT_MAX = 2.50; // strongest compression at 0%
        constexpr double CHG_EXPONENT_MIN = 1.00; // linear at full

        double exponent = CHG_EXPONENT_MAX - (CHG_EXPONENT_MAX - CHG_EXPONENT_MIN) * x;

        shaped = std::pow(x, exponent);
    } else {
        // Discharging curve: blended S-curve strength
        constexpr double DISCHARGE_SCURVE_STRENGTH_LOW  = 0.75;
        constexpr double DISCHARGE_SCURVE_STRENGTH_HIGH = 0.50;

        double high_blend = smootherstep(x);

        double strength =
            DISCHARGE_SCURVE_STRENGTH_LOW -
            (DISCHARGE_SCURVE_STRENGTH_LOW - DISCHARGE_SCURVE_STRENGTH_HIGH) * high_blend;

        shaped = shape_scurve(x, strength);
    }

    const int base_percent = static_cast<int>(std::lround(shaped * 100.0));
    return clampi(base_percent, 0, charging ? 99 : 100);
}

// Visible percent is monotonic within a charge direction.
// Charging never counts down; discharging never counts up.
static int step_limit(int last, int target, bool charging) {
    if (last < 0) return target; // first value
    if (charging) {
        if (target <= last) return last; // never decrease while charging
        if (target <= last + 1) return target; // rise at most +1
        return last + 1;
    } else {
        if (target >= last) return last; // never increase while discharging/unknown
        if (target >= last - 1) return target; // drop at most -1
        return last - 1;
    }
}

// Decide whether the UI-facing percent should be written this loop.
// Resume updates bypass the normal write interval but still use step limiting
// unless snap_now is set.
static bool check_update_visible_percent(
    bool first_visible,
    bool snap_now,
    bool resume_update,
    int internal_percent,
    int visible_percent,
    std::chrono::steady_clock::time_point now,
    std::chrono::steady_clock::time_point last_visible_write
) {
    if (first_visible) {
        return true;
    }

    if (snap_now) {
        return true;
    }

    if (resume_update && internal_percent != visible_percent) {
        return true;
    }

    if (internal_percent == visible_percent) {
        return false;
    }

    auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(now - last_visible_write).count();

    int required_interval = WRITE_INTERVAL_S;
    int delta = std::abs(internal_percent - visible_percent);

    // allow faster catch up if needed.
    if (delta >= 6) {
        required_interval = WRITE_INTERVAL_DELTA_LARGE_S;
    } else if (delta >= 3) {
        required_interval = WRITE_INTERVAL_DELTA_SMALL_S;
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

// After a confirmed full event, wait briefly after unplug and record the
// settled discharge-side full voltage. This avoids using charger-inflated
// voltage as the discharge 100% anchor.
static void check_unplug_full_event(
    const BatteryPaths& bp,
    MapVals& map,
    SmoothedV& sv,
    FullLearnState& fls,
    CalibrationState& calib,
    bool charging,
    bool resume_detected,
    bool charging_changed
) {
    // Only trust unplug-settle learning if we did not resume/suspend in between.
    if (charging_changed && !charging && fls.full_event_active && !resume_detected) {
        if (!sleep_interruptible_s(VFULL_DIS_SETTLE_S)) {
            reset_full_learn_state(fls);
            return;
        }

        int v_burst = burst_sample_voltage(bp.voltage_now);

        // Restart smoothing from the settled post unplug voltage.
        sv.prev1 = v_burst;
        sv.prev2 = v_burst;
        sv.ema = v_burst;

        if (learn_vfull_dis(v_burst, map)) {
            calib.learned_vfull_dis = true;
            create_calibrated_flag(calib);
        }

        reset_full_learn_state(fls);
        return;
    }

    if (charging_changed && !charging && !fls.full_event_active) {
        reset_full_learn_state(fls);
        return;
    }

    if (!charging) {
        reset_full_learn_state(fls);
    }
}

// Track the highest stable charging EMA near the top of the pack.
// Each new peak restarts the dwell timer; dropping too far below the peak
// aborts the calibration attempt.
static void update_peak_tracking(
    FullLearnState& fls,
    bool charging,
    int voltage_ema_mv,
    int64_t now_bt_s
) {
    if (!charging) {
        return;
    }

    if (voltage_ema_mv >= PEAK_TRACK_START_MV) {
        if (!fls.tracking_charge_peak) {
            fls.tracking_charge_peak = true;
            fls.peak_timer_active = true;
            fls.peak_start_bt_s = now_bt_s;
            fls.peak_charge_ema_mv = voltage_ema_mv;
            fls.peak_stability_ok = true;
        } else {
            if (voltage_ema_mv > fls.peak_charge_ema_mv) {
                // New higher peak
                fls.peak_charge_ema_mv = voltage_ema_mv;
                fls.peak_start_bt_s = now_bt_s;
                fls.peak_stability_ok = true;

            } else if (voltage_ema_mv >= (fls.peak_charge_ema_mv - PEAK_STABILITY_WINDOW_MV)) {
                // Still within stability range
                fls.peak_stability_ok = true;

            } else {
                // If it drops below stability range, abort calibration attempt
                reset_full_learn_state(fls);
            }
        }
    } else if (fls.tracking_charge_peak) {
        // Fell below the absolute tracking threshold
        reset_full_learn_state(fls);
    }
}

// Full is inferred when the charging peak has remained stable long enough
// without a new higher EMA peak.
static bool check_peak_dwell_met(
    const FullLearnState& fls,
    int64_t now_bt_s
) {
    if (!fls.peak_timer_active || !fls.peak_stability_ok) {
        return false;
    }

    int64_t peak_elapsed_s = now_bt_s - fls.peak_start_bt_s;
    return peak_elapsed_s >= PEAK_DWELL_S;
}

// Confirm charge-side full once per plug-in/full cycle.
// Prefer the tracked peak EMA over the current EMA so late noise or settling
// does not lower the learned charge-full anchor.
static void check_charge_full_event(
    MapVals& map,
    FullLearnState& fls,
    CalibrationState& calib,
    int voltage_raw_mv,
    int voltage_ema_mv,
    bool full_event_triggered
) {
    // Update V_FULL_CHG on confirmed full event
    if (fls.full_event_active || !full_event_triggered || voltage_raw_mv <= 0) {
        return;
    }

    int candidate_chg = (fls.peak_charge_ema_mv > 0) ? fls.peak_charge_ema_mv : voltage_ema_mv;

    if (learn_vfull_chg(candidate_chg, map)) {
        calib.learned_vfull_chg = true;
        create_calibrated_flag(calib);
    }

    fls.full_event_active = true;
}

// ========================= Percent Calculation Modes =========================
struct PercentResult {
    int percent = -1;
    bool valid = false;
};

// Voltage-mode calculation path:
//   1. handle unplug-after-full learning
//   2. update smoothed voltage
//   3. update charge peak tracking
//   4. compute internal percent
//   5. confirm charge-side full event if PMIC/full-dwell says full
static PercentResult run_voltage_mode(
    const BatteryPaths& bp,
    MapVals& map,
    SmoothedV& sv,
    FullLearnState& fls,
    CalibrationState& calib,
    bool charging,
    bool status_full,
    bool resume_detected,
    bool charging_changed,
    int64_t now_bt_s
) {
    PercentResult result;

    int voltage_raw_mv = read_voltage_mv(bp.voltage_now);

    check_unplug_full_event(
        bp,
        map,
        sv,
        fls,
        calib,
        charging,
        resume_detected,
        charging_changed
    );

    int voltage_ema_mv = update_smoothed_voltage(voltage_raw_mv, map, sv);

    update_peak_tracking(
        fls,
        charging,
        voltage_ema_mv,
        now_bt_s
    );

    bool peak_dwell_met = check_peak_dwell_met(fls, now_bt_s);
    bool full_event_triggered = status_full || peak_dwell_met;
    bool allow_visible_100 = full_event_triggered || fls.full_event_active;

    int percent = voltage_to_percent(voltage_ema_mv, map, charging);

    // Charging is capped at 99% until a real/full-inferred event occurs.
    if (charging) {
        if (allow_visible_100) {
            percent = 100;
        } else if (percent > 99) {
            percent = 99;
        }
    }

    check_charge_full_event(
        map,
        fls,
        calib,
        voltage_raw_mv,
        voltage_ema_mv,
        full_event_triggered
    );

    result.percent = percent;
    result.valid = true;
    return result;
}

// PMIC mode trusts the kernel/PMIC capacity value but still uses the same
// visible-output and hook pipeline as voltage mode.
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

// Initialize voltage smoothing on daemon start.
// If the one-shot restore file still matches the current voltage and charge
// state, reuse its EMA and visible percent to avoid a visible jump.
static StartupInitResult init_voltage_startup_state(
    const BatteryPaths& bp,
    SmoothedV& sv,
    int& internal_percent,
    int& visible_percent
) {
    StartupInitResult result;

    ChargeStatus startup_status = ChargeStatus::Unknown;

    if (!bp.status.empty()) {
        startup_status = read_charge_status(bp.status);
        result.restore_charging = (startup_status == ChargeStatus::Charging || startup_status == ChargeStatus::Full);
    }

    int v_now = burst_sample_voltage(bp.voltage_now);

    auto rs_opt = consume_restore_state();
    if (rs_opt) {
        const auto& rs = *rs_opt;

        int restore_delta_mv = v_now - rs.ema_mv;

        bool voltage_matches = rs.charging
            ? (restore_delta_mv >= -RESTORE_CHG_DOWN_MV &&
            restore_delta_mv <=  RESTORE_CHG_UP_MV)
            : (restore_delta_mv >= -RESTORE_DIS_DOWN_MV &&
            restore_delta_mv <=  RESTORE_DIS_UP_MV);

        if (v_now > 0 &&
            rs.ema_mv > 0 &&
            rs.visible_percent >= 0 &&
            rs.visible_percent <= 100 &&
            rs.charging == result.restore_charging &&
            voltage_matches)
        {
            sv.prev1 = rs.ema_mv;
            sv.prev2 = rs.ema_mv;
            sv.ema   = rs.ema_mv;

            visible_percent = rs.visible_percent;
            internal_percent = rs.visible_percent;
            result.restored_state = true;
        }
    }

    // If restore was not used, always pre-seed EMA from the startup voltage for additional smoothing
    // Compensate for boot only load voltage suppression while discharging
    if (!result.restored_state && v_now > 0) {
        if (startup_status == ChargeStatus::Discharging &&
            boottime_s() < BOOT_COMP_MAX_UPTIME_S) {
            v_now += BOOT_COMP_MV;
        }

        sv.prev1 = v_now;
        sv.prev2 = v_now;
        sv.ema   = v_now;
    }

    return result;
}

// ========================= Main =========================
int main() {
    // Signals
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    std::signal(SIGUSR1, handle_resume_signal);

    // Fire-and-forget hooks: prevent/cleanup zombies
    std::signal(SIGCHLD, SIG_IGN);

    // Ensure root directory exists
    fs::create_directories(fs::path(ROOT));

    // Ensure config is valid
    ensure_config();

    // Load layered configuration and resolve the persistent data directory.
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

    // Ensure directory exists
    std::error_code data_dir_ec;
    fs::create_directories(g_paths.data_dir, data_dir_ec);

    if (data_dir_ec) {
        std::fprintf(stderr,
            "batteryplus: Error: unable to create persistent data directory\n");
        return 1;
    }

    HookCache hooks;
    load_hook_cache(hooks);

    // Find battery
    auto bp_opt = find_battery();
    if (!bp_opt) {
        std::fprintf(stderr, "batteryplus: Error: No battery detected!\n");
        return 1;
    }
    BatteryPaths bp = *bp_opt;

    if (g_cfg.mode == BatteryMode::Voltage) {
        if (bp.voltage_now.empty()) {
            std::fprintf(stderr, "batteryplus: Error: No battery voltage path detected!\n");
            return 1;
        }
    } else if (g_cfg.mode == BatteryMode::Pmic) {
        if (bp.capacity.empty()) {
            std::fprintf(stderr, "batteryplus: Error: No PMIC capacity path detected!\n");
            return 1;
        }
    }

    // Voltage map
    MapVals map = load_map(g_paths.map_file);

    if (!fs::exists(g_paths.map_file)) {
        save_map_atomic(g_paths.map_file, map);
    }

    // State
    SmoothedV sv;
    int internal_percent = -1; // smoothed percent from voltage
    int visible_percent = -1; // step-limited percent we expose
    FullLearnState fls;
    CalibrationState calib; // used for calibration flag file creation

    // Init/Check if we should restore state saved by a previous state
    bool restore_charging = false;

    if (g_cfg.mode == BatteryMode::Voltage) {
        StartupInitResult startup_init = init_voltage_startup_state(
            bp,
            sv,
            internal_percent,
            visible_percent
        );
        restore_charging = startup_init.restore_charging;
    }

    // Track charge/discharge transitions
    bool first_state_hook = true;
    bool last_charging = restore_charging;
    ChargeStatus last_charge_status = restore_charging ? ChargeStatus::Charging : ChargeStatus::Discharging;

    // Keyed by (mode, bucket) so 0% discharging can fire even if 0% charging fired last loop
    int last_hook_key = -1;

    // Resume detection thresholds
    static constexpr long SHORT_GAP_S = 15 * 60; // threshold for a nudge
    static constexpr long LONG_GAP_S  = 60 * 60; // threshold for burst-sampling after a long suspend gap

    // Long-gap resume snap thresholds.
    // Downward movement is trusted more readily because battery drain during suspend is expected.
    // Upward movement requires a larger delta because voltage can rebound after suspend
    static constexpr int RESUME_SNAP_DOWN_DELTA_MV = 25;
    static constexpr int RESUME_SNAP_UP_DELTA_MV   = 75;

    auto last_visible_write = std::chrono::steady_clock::now();
    int64_t last_loop_bt_s = boottime_s();
    long gap_s = 0;

    int calc_tick = INTERNAL_INTERVAL_S;
    while (g_running) {
        // Get time data
        auto now = std::chrono::steady_clock::now();

        int64_t now_bt_s = boottime_s();
        gap_s = now_bt_s - last_loop_bt_s;
        if (gap_s < 0) {
            gap_s = 0;
        }
        last_loop_bt_s = now_bt_s;

        // Read charge status
        ChargeStatus charge_status = read_charge_status(bp.status);
        bool charging = (charge_status == ChargeStatus::Charging || charge_status == ChargeStatus::Full);
        bool status_full = (charge_status == ChargeStatus::Full);

        bool first_visible = (visible_percent < 0);
        bool hooks_fired = false;

        // Resume/gap detection.
        // Explicit SIGUSR1 or a long loop gap invalidates in-progress calibration.
        bool resume_hint = g_resume_hint.exchange(false);
        bool short_resume_gap = (gap_s >= SHORT_GAP_S);
        bool long_resume_gap = (gap_s >= LONG_GAP_S);

        bool resume_detected = resume_hint || short_resume_gap;

        // Abort any in progress calibration state
        if (resume_detected) {
            reset_full_learn_state(fls);
        }

        bool snap_now = false;
        if (long_resume_gap && g_cfg.mode == BatteryMode::Voltage) {
            // Long resume gaps may reflect real battery movement while asleep, but voltage can
            // also rebound after suspend/load removal. Always burst-sample and reseed EMA so
            // internal calculations restart from a current voltage, but only snap the visible
            // percent if the voltage moved far enough to be meaningful.

            int ema_before_resume = sv.ema;

            int v_stable = burst_sample_voltage(bp.voltage_now);
            if (v_stable > 0) {
                sv.prev1 = sv.prev2 = v_stable;
                sv.ema   = v_stable;

                if (ema_before_resume > 0) {
                    int resume_delta_mv = v_stable - ema_before_resume;

                    if (resume_delta_mv >= RESUME_SNAP_UP_DELTA_MV ||
                        resume_delta_mv <= -RESUME_SNAP_DOWN_DELTA_MV) {
                        snap_now = true;
                    }
                }
            }
        }

        // Detect charging state change
        bool charging_changed = (charging != last_charging);
        bool status_changed = (charge_status != last_charge_status);

        // Recalculate immediately for first write, resume, plug/unplug, or normal interval.
        bool do_calc = first_visible || resume_detected || charging_changed || calc_tick >= INTERNAL_INTERVAL_S;

        if (do_calc) {
            calc_tick = 0;

            PercentResult pr;

            if (g_cfg.mode == BatteryMode::Voltage) {
                pr = run_voltage_mode(
                    bp,
                    map,
                    sv,
                    fls,
                    calib,
                    charging,
                    status_full,
                    resume_detected,
                    charging_changed,
                    now_bt_s
                );
            } else {
                pr = run_pmic_mode(
                    bp.capacity
                );
            }

            if (pr.valid) {
                internal_percent = pr.percent;
            }

            bool percent_file_exists = fs::exists(PERCENT_FILE);

            // Decide if we need to update/write to the battery percent file
            bool need_visible_update = !percent_file_exists || check_update_visible_percent(
                first_visible,
                snap_now,
                resume_detected,
                internal_percent,
                visible_percent,
                now,
                last_visible_write
            );

            if (need_visible_update) {
                int new_visible = compute_new_visible_percent(
                    first_visible,
                    snap_now,
                    visible_percent,
                    internal_percent,
                    charging
                );

                if (new_visible != visible_percent || !percent_file_exists) {
                    visible_percent = new_visible;
                    fs::create_directories(fs::path(PERCENT_FILE).parent_path());
                    (void)write_atomic(PERCENT_FILE, std::to_string(visible_percent) + "\n", 0644);
                    last_visible_write = now;

                    // fire hooks on exact 5% increments
                    hooks_fired = check_run_bucket_hooks(
                        hooks,
                        charging,
                        visible_percent,
                        charge_status,
                        last_hook_key
                    ) || hooks_fired;
                }
            }
        }

        // On plug/unplug(state change), rerun bucket hooks for the new charge state.
        if (charging_changed) {
            hooks_fired = check_run_bucket_hooks(
                hooks,
                charging,
                visible_percent,
                charge_status,
                last_hook_key
            ) || hooks_fired;
        }

        // State hooks are independent from 5% bucket hooks and fire on status changes.
        if (first_state_hook || status_changed) {
            run_state_hooks_cached(hooks, visible_percent, charge_status);
            first_state_hook = false;
        }

        // Run wildcard scripts once on reset only if we didn't already
        hooks_fired = check_resume_wildcard_hooks(
            hooks,
            resume_detected,
            hooks_fired,
            charging,
            visible_percent,
            charge_status
        ) || hooks_fired;

        // Final update for charging/status
        last_charging = charging;
        last_charge_status = charge_status;

        // Sleep
        if (g_running) {
            std::this_thread::sleep_for(1s);
            ++calc_tick;
        }
    }

    // Save one-shot restore data so a daemon restart can preserve UI continuity.
    if (g_cfg.mode == BatteryMode::Voltage) {
        RestoreState st;
        st.charging = last_charging;
        st.ema_mv = sv.ema;
        st.visible_percent = visible_percent;

        if (st.ema_mv > 0 && st.visible_percent >= 0 && st.visible_percent <= 100) {
            (void)save_restore_state(st);
        }
    }

    return 0;
}
