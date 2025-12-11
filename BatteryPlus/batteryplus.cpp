// BatteryPlus — Voltage-only battery monitor daemon for handheld Linux systems
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
//   BatteryPlus is an alternative battery reporting daemon that relies solely
//   on voltage measurements to compute battery percentage. It incorporates
//   median and exponential smoothing, droop compensation, and adaptive
//   full-voltage calibration to deliver calm, stable, and intuitive percent
//   behavior suitable for handheld devices.
//
// Core behaviors:
//
//   • Voltage-based percent only
//       - Percent derived exclusively from smoothed voltage
//       - V_EMPTY fixed (target 0%), V_FULL learned automatically
//       - Gamma curve to visually linearize discharge behavior
//
//   • Median-of-3 + EMA smoothing
//       - Filters jitter from battery load and charger noise
//
//   • Droop compensation
//       - Adaptive and per-device learning over time
//
//   • Adaptive V_FULL learning
//       - Updates V_FULL once using smoothed (EMA) voltage when status == "Full"
//       - Saves map file atomically
//
//   • Calm percent exposure (UI-friendly)
//       - Internal percent updated every INTERNAL_INTERVAL_S
//       - Visible percent written only every WRITE_INTERVAL (halved under LOW_PCT_THRESHOLD)
//       - On large resume jump (>=3%), snap to internal immediately
//       - On small delta, smoothly catch up
//
//   • Hooks system (5% buckets)
//       - Runs scripts in /etc/batteryplus/{charging.d|discharging.d}/
//       - Based on visible percent bucket changes
//
// Files:
//   /tmp/battery.percent                 - exported visible % for UI polling
//   /userdata/system/batteryplus-voltage.map - stores V_FULL, V_EMPTY, and V_DROOP
//
// Build:
//   aarch64-linux-gnu-g++ -O3 -flto -std=gnu++20 -Wall -Wextra -pedantic batteryplus-vol.cpp -o batteryplus-vol
//
//
// Signals:
//   SIGTERM / SIGINT — stop daemon
//   SIGUSR1          — reset; triggers snap if delta is over threshold (i.e. can be used when resuming from suspend)

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
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

// ========================= Config (constants) =========================
static constexpr const char* MAP_FILE = "/userdata/system/batteryplus-voltage.map";
static constexpr const char* PERCENT_FILE = "/tmp/battery.percent";
static constexpr const char* ROOT = "/etc/batteryplus"; // use {charging.d, discharging.d}

// Timers
static constexpr int INTERNAL_INTERVAL_S = 10; // how often internal calculations are done in seconds
static constexpr int CHARGE_FULL_FALLBACK_TICKS = 30 * 60 / INTERNAL_INTERVAL_S; // 30min at 10s intervals

// Percent write parameters
static constexpr int LOW_PCT_THRESHOLD = 10; // threshold where we update faster (%)
static constexpr int WRITE_INTERVAL = 60;

// EMA parameters
static constexpr int ALPHA_NUM = 2;
static constexpr int ALPHA_DEN = 10;

// Defaults for map if missing
static constexpr int DEFAULT_V_FULL = 4000; // mV (absolute ceiling, learned per device)
static constexpr int DEFAULT_V_EMPTY = 3250; // mV (fixed, never learned)
static constexpr int DEFAULT_V_DROOP = 50; // mV (offset applied while charging, learned per device)

// ========================= Globals =========================
static std::atomic<bool> g_running { true };
static std::atomic<bool> g_reset{false};

// ========================= Utilities =========================
static void handle_reset(int) { g_reset = true; }

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

static inline int parse_leading_bucket(const std::string& fname) {
    if (fname.empty() || !std::isdigit((unsigned char)fname[0])) return -1;
    int i = 0, v = 0;
    while (i < (int)fname.size() && std::isdigit((unsigned char)fname[i]) && i < 3) {
        v = v * 10 + (fname[i] - '0');
        ++i;
    }
    return (v >= 0 && v <= 100) ? v : -1;
}

static inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static int median3(int a, int b, int c) {
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
    return b;
}

static std::string read_charge_status(const fs::path& status_path) {
    auto s = slurp(status_path);
    if (!s) return "Unknown";
    return *s;
}

// ========================= Hook System =========================
// Execute all executables in {charging|discharging}.d whose filename starts with the battery% number.
// Supports plain and zero-padded, e.g., "50", "050", "50-".
// Wildcards: filenames that do NOT start with a digit run on every bucket change.

static bool is_executable(const fs::directory_entry& de) {
    if (!de.is_regular_file()) return false;
    return ::access(de.path().c_str(), X_OK) == 0;
}

static int run_hook_file(const fs::path& file) {
    pid_t pid = ::fork();
    if (pid < 0) return -1;

    if (pid == 0) {
        // child
        int nullfd = ::open("/dev/null", O_RDWR);
        if (nullfd >= 0) {
            ::dup2(nullfd, STDOUT_FILENO);
            ::dup2(nullfd, STDERR_FILENO);
            ::close(nullfd);
        }
        const char* argv[] = { file.c_str(), nullptr };
        ::execv(argv[0], (char* const*)argv);
        _exit(127);
    }

    // parent: wait with timeout
    int status = 0;
    constexpr int MAX_MS = 2000; // max time to wait for a hook before terminating it
    constexpr int STEP_MS = 50; // polling interval to check the hook process

    int waited = 0;
    while (waited < MAX_MS) {
        pid_t r = ::waitpid(pid, &status, WNOHANG);
        if (r == pid) {
            // child finished
            return status;
        } else if (r < 0) {
            // wait error
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(STEP_MS));
        waited += STEP_MS;
    }

    // Timeout: kill the hook
    ::kill(pid, SIGKILL);
    (void)::waitpid(pid, &status, 0);
    return -1;
}

static constexpr int NUM_BUCKETS = 21; // 0 -> 100 in 5% increments

struct HookCache {
    std::array<std::vector<fs::path>, NUM_BUCKETS> charging;
    std::vector<fs::path> charging_any;
    std::array<std::vector<fs::path>, NUM_BUCKETS> discharging;
    std::vector<fs::path> discharging_any;
    bool loaded = false;
};

static inline int bucket5(int percent) {
    percent = clampi(percent, 0, 100);
    return (percent / 5) * 5;
}

static inline int bucket_index(int percent) {
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
            // Ignore numbers not multiple of 5%
            continue;
        }
    }

    auto sorter = [](auto& v){ std::sort(v.begin(), v.end()); };
    for (auto& v : buckets) sorter(v);
    sorter(wildcards);
}

static void load_hook_cache(HookCache& hc) {
    fs::create_directories(fs::path(ROOT) / "charging.d");
    fs::create_directories(fs::path(ROOT) / "discharging.d");
    scan_hook_dir(fs::path(ROOT) / "charging.d",    hc.charging,    hc.charging_any);
    scan_hook_dir(fs::path(ROOT) / "discharging.d", hc.discharging, hc.discharging_any);
    hc.loaded = true;
}

static inline void run_paths(const std::vector<fs::path>& paths) {
    for (const auto& p : paths) {
        if (::access(p.c_str(), X_OK) == 0)
            run_hook_file(p);
    }
}

static void run_bucket_hooks_cached(const HookCache& hc, bool charging, int percent_value) {
    if (!hc.loaded) return;
    int bi = bucket_index(percent_value);
    const auto& buckets = charging ? hc.charging : hc.discharging;
    const auto& any     = charging ? hc.charging_any : hc.discharging_any;

    run_paths(buckets[bi]); // run all scripts for this 5% bucket
    run_paths(any); // wildcard scripts every change
}

// ========================= Battery discovery =========================
struct BatteryPaths {
    fs::path status;
    fs::path voltage_now;
};

static std::optional<BatteryPaths> find_battery() {
    auto has_required = [](const fs::path& d){
        return fs::exists(d/"status") && fs::exists(d/"voltage_now");
    };

    std::vector<std::string> patterns = {
        "BAT", "bat", "FUEL", "fuel"
    };

    fs::path base("/sys/class/power_supply");
    if (!fs::exists(base)) return std::nullopt;

    for (auto& de : fs::directory_iterator(base)) {
        std::string name = de.path().filename().string();
        bool match = false;
        for (auto& p : patterns) {
            if (name.find(p) != std::string::npos) { match = true; break; }
        }
        if (!match) continue;
        if (has_required(de.path())) {
            BatteryPaths bp;
            bp.status = de.path()/"status";
            bp.voltage_now = de.path()/"voltage_now";
            return bp;
        }
    }

    // Fallback: any power_supply that has required files
    for (auto& de : fs::directory_iterator(base)) {
        if (has_required(de.path())) {
            BatteryPaths bp;
            bp.status = de.path()/"status";
            bp.voltage_now = de.path()/"voltage_now";
            return bp;
        }
    }

    return std::nullopt;
}

// ========================= Map file =========================
struct MapVals {
    int V_FULL = DEFAULT_V_FULL;
    int V_EMPTY = DEFAULT_V_EMPTY;
    int V_DROOP = DEFAULT_V_DROOP;
};

static void save_map_atomic(const fs::path& path, const MapVals& m) {
    std::string data;
    data += "V_FULL=" + std::to_string(m.V_FULL) + "\n";
    data += "V_EMPTY=" + std::to_string(m.V_EMPTY) + "\n";
    data += "V_DROOP=" + std::to_string(m.V_DROOP) + "\n";
    fs::create_directories(path.parent_path());
    (void)write_atomic(path, data, 0644);
}

static MapVals load_map(const fs::path& path) {
    MapVals m;
    bool need_save = false;
    bool found_vfull = false;
    bool found_vempty = false;
    bool found_vdroop = false;

    std::ifstream f(path);
    if (f) {
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("V_FULL=", 0) == 0) {
                m.V_FULL = std::atoi(line.c_str() + 7);
                found_vfull = true;
            } else if (line.rfind("V_EMPTY=", 0) == 0) {
                m.V_EMPTY = std::atoi(line.c_str() + 8);
                found_vempty = true;
            } else if (line.rfind("V_DROOP=", 0) == 0) {
                m.V_DROOP = std::atoi(line.c_str() + 8);
                found_vdroop = true;
            }
        }
    } else {
        // No file yet
        return m;
    }

    // Ensure defaults if missing
    if (!found_vfull) {
        m.V_FULL = DEFAULT_V_FULL;
        need_save = true;
    }
    if (!found_vempty) {
        m.V_EMPTY = DEFAULT_V_EMPTY;
        need_save = true;
    }
    if (!found_vdroop) {
        m.V_DROOP = DEFAULT_V_DROOP;
        need_save = true;
    }

    // Sanity V_EMPTY
    if (m.V_EMPTY < 3000 || m.V_EMPTY > 3400) {
        m.V_EMPTY = DEFAULT_V_EMPTY;
        need_save = true;
    }

    // Sanity V_FULL
    if (m.V_FULL < m.V_EMPTY + 300 || m.V_FULL > 4400) {
        // Values are probably garbage so reset both main voltages
        m.V_FULL = DEFAULT_V_FULL;
        m.V_EMPTY = DEFAULT_V_EMPTY;
        need_save = true;
    }

    // Sanity V_DROOP
    if (m.V_DROOP <= 1 || m.V_DROOP > 300) {
        m.V_DROOP = DEFAULT_V_DROOP;
        need_save = true;
    }

    if (need_save) {
        save_map_atomic(path, m);
    }

    return m;
}

static void learn_vdroop(int last_charging_ema_mv, int discharge_ema_mv, MapVals& map, const char* map_file_path) {
    if (last_charging_ema_mv <= 0 || discharge_ema_mv <= 0) return;

    int sample_mv = last_charging_ema_mv - discharge_ema_mv;

    // Only learn from realistic positive droop
    if (sample_mv <= 1 || sample_mv >= 300) {
        return;
    }

    int old_droop = (map.V_DROOP > 0 ? map.V_DROOP : DEFAULT_V_DROOP);

    // 85% old, 15% new
    int blended = (17 * old_droop + 3 * sample_mv) / 20;

    int max_step_up = 10;
    int max_step_down = 5;

    int max_allowed = old_droop + max_step_up;
    int min_allowed = old_droop - max_step_down;

    if (blended > max_allowed) blended = max_allowed;
    if (blended < min_allowed) blended = min_allowed;

    blended = clampi(blended, 5, 250);

    int quantized = ((blended + 2) / 5) * 5; // round to nearest 5 mV

    if (std::abs(quantized - map.V_DROOP) < 3) {
        return;
    }

    if (quantized != map.V_DROOP) {
        map.V_DROOP = quantized;
        save_map_atomic(map_file_path, map);
    }
}

static void learn_vfull(int voltage_raw_mv, int voltage_ema_mv, MapVals& map) {
    if (voltage_raw_mv <= 0 || voltage_ema_mv <= 0) {
        return;
    }

    int candidate = voltage_ema_mv;
    int old_vfull = map.V_FULL;

    // Ignore tiny changes
    int diff = candidate - old_vfull;
    if (std::abs(diff) < 5) {
        return;
    }

    // Don't let a single calibration change it too much
    constexpr int MAX_SINGLE_STEP = 50; // mv
    if (diff > MAX_SINGLE_STEP) diff = MAX_SINGLE_STEP;
    if (diff < -MAX_SINGLE_STEP) diff = -MAX_SINGLE_STEP;

    // 75% old, 25% new
    int nudged = old_vfull + diff;
    int blended = (3 * old_vfull + nudged) / 4;

    // Quantize to only keep meaningful changes
    int quantized = ((blended + 2) / 5) * 5;

    // Only save if meaningfully changed
    if (std::abs(quantized - old_vfull) >= 5) {
        map.V_FULL = quantized;
        save_map_atomic(MAP_FILE, map);
    }
}

// ========================= Percent calc =========================
struct SmoothedV {
    int prev1 = -1;
    int prev2 = -1;
    int ema = -1;
};

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

// Dynamic droop compensation
static int compute_dynamic_droop_mv(int approx_pct, const MapVals& m)
{
    approx_pct = clampi(approx_pct, 0, 100);

    int range_mv = m.V_FULL - m.V_EMPTY;

    // Baseline learned device droop
    int base = (m.V_DROOP > 0 ? m.V_DROOP : DEFAULT_V_DROOP);

    // curb "rapid charging" appearance at low end
    constexpr double FACTOR_MIN   = 2.0; // droop multiplier at 0%
    constexpr int    LOW_BAND_MAX = 30; // max % which this stops being applied
    constexpr double SHAPE_EXP    = 2.0; // >1.0 = more weight near 0%

    double factor = 1.0;
    if (approx_pct < LOW_BAND_MAX) {
        double t = static_cast<double>(approx_pct) / LOW_BAND_MAX;
        double w = 1.0 - std::clamp(t, 0.0, 1.0);

        double shaped = std::pow(w, SHAPE_EXP);
        factor = 1.0 + (FACTOR_MIN - 1.0) * shaped;
    }

    int droop = static_cast<int>(std::lround(base * factor));

    // Clamp just in case
    int max_global = range_mv / 2; // at most half the voltage window
    droop = clampi(droop, 10, max_global);

    return droop;
}

static int voltage_to_percent(int voltage_now_mv, const MapVals& m) {
    if (voltage_now_mv <= 0) {
        // If we somehow get garbage voltage just return 1% so it's intentionally obvious
        return 1;
    }

    int v_empty = m.V_EMPTY;
    int v_full  = m.V_FULL;

    // Apply a dynamic offset from learned v_droop so when unplugging charger it's not a steep drop
    int droop_mv = (m.V_DROOP > 0 ? m.V_DROOP : DEFAULT_V_DROOP);
    droop_mv = clampi(droop_mv, 10, 150);

    int vfull_adj = v_full - droop_mv;
    if (vfull_adj <= v_empty + 50) {
        vfull_adj = v_empty + 50;
    }

    int full_range = v_full - v_empty;

    double window_frac = 0.03; // % of total range
    int window_mv = static_cast<int>(std::lround(full_range * window_frac));
    window_mv = clampi(window_mv, 10, 30);

    int v_100_start = vfull_adj - window_mv;
    if (v_100_start < v_empty + 50) {
        v_100_start = v_empty + 50;
    }

    // Top 100%
    if (voltage_now_mv >= v_100_start) {
        return 100;
    }

    // 0–99%
    int v_clamped = clampi(voltage_now_mv, v_empty, v_100_start);
    int range_adj = v_100_start - v_empty;
    double x = 0.0;
    if (range_adj > 0) {
        x = static_cast<double>(v_clamped - v_empty) / static_cast<double>(range_adj);
    }
    if (x < 0.0) x = 0.0;
    if (x > 1.0) x = 1.0;

    constexpr double gamma = 1.20;
    double shaped = std::pow(x, gamma);

    int base_percent = static_cast<int>(std::lround(shaped * 100.0));
    // apply gamma curve to only 0-99% (100% is excluded to keep an accurate top end)
    base_percent = clampi(base_percent, 0, 99);

    return base_percent;
}

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

// ========================= Signal handling =========================
static void handle_signal(int) { g_running = false; }

// ========================= Main =========================
int main() {
    // Signals
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    std::signal(SIGUSR1, handle_reset);

    // Ensure directories exist
    fs::create_directories(fs::path(ROOT));
    fs::create_directories(fs::path(ROOT) / "charging.d");
    fs::create_directories(fs::path(ROOT) / "discharging.d");

    HookCache hooks;
    load_hook_cache(hooks);

    // Find battery
    auto bp_opt = find_battery();
    if (!bp_opt) {
        std::fprintf(stderr, "batteryplus: Error: No battery detected!\n");
        return 1;
    }
    BatteryPaths bp = *bp_opt;

    // Voltage map
    MapVals map = load_map(MAP_FILE);

    if (!fs::exists(MAP_FILE)) {
        fs::create_directories(fs::path(MAP_FILE).parent_path());
        save_map_atomic(MAP_FILE, map);
    }

    // State
    SmoothedV sv;
    int internal_percent = -1; // smoothed percent from voltage
    int visible_percent = -1; // step-limited percent we expose
    int last_bucket = -1;
    int last_charging_ema_mv = -1;
    bool vfull_recorded = false;

    // Droop learning: require 3 stable ticks on each side
    int charging_streak = 0;
    int discharging_streak = 0;
    bool droop_armed = false;

    auto last_visible_write = std::chrono::steady_clock::now();

    while (g_running) {
        // Read status and voltage
        int voltage_raw_mv = read_voltage_mv(bp.voltage_now);
        std::string status_str = read_charge_status(bp.status);

        bool charging = false;
        bool status_full = false;
        bool first_visible = (visible_percent < 0);
        bool reset = g_reset.exchange(false);
        bool hooks_fired = false;

        if (!status_str.empty()) {
            if (status_str.rfind("Charging", 0) == 0) {
                charging = true;
            } else if (status_str.rfind("Full", 0) == 0) {
                charging = true;
                status_full = true;
            }
        }

        // Track charging/discharging streaks, and arm droop learning after 3 charging ticks
        if (charging) {
            charging_streak++;
            discharging_streak = 0;

            if (charging_streak >= 3) {
                droop_armed = true;
            }
        } else {
            discharging_streak++;
            charging_streak = 0;
        }

        // Median-of-3 then EMA for live voltage (for calculations only)
        if (sv.prev1 < 0)
            sv.prev1 = (voltage_raw_mv > 0 ? voltage_raw_mv : map.V_FULL);
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

        int voltage_ema_mv = sv.ema;

        // Voltage droop compensation while charging
        int voltage_for_percent_mv = voltage_ema_mv;

        if (charging) {
            // Use stable visible percent if available
            // Otherwise use a draft percent directly from the ema voltage.
            int approx_pct = (visible_percent >= 0)
                ? visible_percent
                : voltage_to_percent(voltage_ema_mv, map);

            int droop_mv = compute_dynamic_droop_mv(approx_pct, map);

            if (droop_mv > 0) {
                int adjusted = voltage_ema_mv - droop_mv;

                if (adjusted < map.V_EMPTY)
                    adjusted = map.V_EMPTY;
                if (adjusted > map.V_FULL)
                    adjusted = map.V_FULL;

                voltage_for_percent_mv = adjusted;
            }
        }

        // Calculate target percent
        int target = voltage_to_percent(voltage_for_percent_mv, map);

        internal_percent = target;

        bool timeout_full = charging && internal_percent >= 99 && charging_streak >= CHARGE_FULL_FALLBACK_TICKS;

        // Set to 100% once pmic reports
        if (charging) {
            if (status_full || timeout_full) {
                internal_percent = 100;
            } else if (internal_percent > 99) {
                internal_percent = 99;
            }
        }

        // Compute delta
        int delta_pct = 0;
        if (!first_visible && visible_percent >= 0) {
            delta_pct = std::abs(internal_percent - visible_percent);
        }

        // On reset decide if we should wipe smoothing
        bool wipe_ema = false;
        if (reset) {
            if (first_visible || delta_pct >= 3) {
                wipe_ema = true;
            }
        }

        // If needed reset ema history
        if (wipe_ema) {
            if (voltage_raw_mv > 0) {
                sv.prev1 = sv.prev2 = voltage_raw_mv;
                sv.ema = voltage_raw_mv;
            } else {
                sv.prev1 = sv.prev2 = sv.ema = -1;
            }
        }

        // Update V_FULL once when status is "Full"
        if (!vfull_recorded) {
            if ((status_full || timeout_full) && voltage_raw_mv > 0) {
                learn_vfull(voltage_raw_mv, voltage_ema_mv, map);
                vfull_recorded = true;
                }
            }

        // Decide if we need to write the file / run hooks
        bool need_visible_update = false;
        auto now = std::chrono::steady_clock::now();

        if (first_visible) {
            // Initial loop
            need_visible_update = true;

        } else if (internal_percent != visible_percent) {
            auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(now - last_visible_write).count();

            // Choose interval based on low or normal range or when charging
            int required_interval = WRITE_INTERVAL;
            if (internal_percent <= LOW_PCT_THRESHOLD || charging) {
                required_interval = WRITE_INTERVAL / 2; // lets just halve normal interval
            }

            if (reset && delta_pct >= 3) {
                // reset + meaningful change: force write now
                need_visible_update = true;
            } else if (elapsed_s >= required_interval) {
                need_visible_update = true;
            }
        }

        if (need_visible_update) {
            int new_visible = visible_percent;

            if (first_visible) {
                // Initial loop
                new_visible = internal_percent;
            } else if (reset && delta_pct >= 3) {
                // reset + meaningful change: snap visible to internal
                new_visible = internal_percent;
            } else {
                new_visible = step_limit(visible_percent, internal_percent, charging);
            }

            if (new_visible != visible_percent) {
                visible_percent = new_visible;
                fs::create_directories(fs::path(PERCENT_FILE).parent_path());
                (void)write_atomic(PERCENT_FILE, std::to_string(visible_percent) + "\n", 0644);
                last_visible_write = now;

                // fire once and on exact 5% increments
                if (visible_percent % 5 == 0) {
                    int b = visible_percent;
                    if (b != last_bucket) {
                        run_bucket_hooks_cached(hooks, charging, visible_percent);
                        last_bucket = b;
                        hooks_fired = true; // we fired off hooks this loop
                    }
                }
            }
        }

        // Run wildcard scripts once on reset only if we didn't already
        if (reset) {
            if (!hooks_fired) {
                const auto& any = charging ? hooks.charging_any : hooks.discharging_any;
                run_paths(any);
            }
        }

        // Learn droop once when armed
        if (droop_armed && !charging && discharging_streak >= 3) {
            if (last_charging_ema_mv > 0 && v_med > 0) {
                learn_vdroop(last_charging_ema_mv, v_med,map, MAP_FILE);
            }
            // Reset arming
            droop_armed = false;
        }

        // Remember last charging voltage for next loop
        if (charging) {
            last_charging_ema_mv = voltage_ema_mv;
        }

        // Sleep
        for (int i = 0; i < INTERNAL_INTERVAL_S && g_running; ++i) {
            if (g_reset.load()) break;
            std::this_thread::sleep_for(1s);
        }
    }

    return 0;
}
