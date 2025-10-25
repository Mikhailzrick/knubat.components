// batteryplus.cpp — battery monitor daemon
// - Finds battery in /sys/class/power_supply
// - Computes blended % from raw capacity and voltage (with additional smoothing)
// - Maintains /userdata/system/battery-voltage.map with V_FULL (raw mV at >=99% while charging/full),
//   V_EMPTY (fixed, target 0%), V_RAW0 (learned once per process when raw hits 0 after arming at raw>10)
// - Writes /tmp/battery.percent atomically on change
// - Executes hooks in /etc/batteryplus/{charging.d,discharging.d} on change (5% increments, example: 50testscript executes at 50%)
// - Build Arm64: aarch64-linux-gnu-g++ -O3 -flto -std=gnu++20 -Wall -Wextra -pedantic batteryplus.cpp -o batteryplus

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
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
static constexpr const char* MAP_FILE = "/userdata/system/battery-voltage.map";
static constexpr const char* PERCENT_FILE = "/tmp/battery.percent";
static constexpr const char* HOOK_ROOT = "/etc/batteryplus"; // use {charging.d, discharging.d}

static constexpr int BASE_INTERVAL_S = 60;
static constexpr int FAST_INTERVAL_S = 10;

// EMA parameters
static constexpr int ALPHA_NUM = 2;
static constexpr int ALPHA_DEN = 10;

// Blending weights
static constexpr int MAX_CHARGE_VOLT_WEIGHT = 30; // cap voltage influence when charging

// Snap-to-target if the deviation is too large
static constexpr int SNAP_DELTA = 6;  // percent

// RAW0 arming threshold
static constexpr int RAW0_ARM_THRESHOLD = 10;

// Defaults for map if missing
static constexpr int DEFAULT_V_FULL = 4000;  // mV
static constexpr int DEFAULT_V_EMPTY = 3250; // mV (fixed, never learned)
static constexpr int DEFAULT_V_RAW0 = 3325;  // mV

// ========================= Globals =========================
static std::atomic<bool> g_running { true };
static std::atomic<bool> g_wakeup{false};

// ========================= Utilities =========================
static void handle_wakeup(int) { g_wakeup = true; }

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

static bool is_executable(const fs::directory_entry& de) {
    if (!de.is_regular_file()) return false;
    return ::access(de.path().c_str(), X_OK) == 0;
}

static int run_hook_file(const fs::path& file) {
    pid_t pid = ::fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        // redirect stdout/stderr to /dev/null
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
    int status = 0;
    (void)::waitpid(pid, &status, 0);
    return status;
}

// Execute all executables in {charging|discharging}.d whose filename starts with the battery% number.
// Supports plain and zero-padded, e.g., "50", "050", "50-".
// Wildcards: filenames that do NOT start with a digit run on every bucket change.
static void run_bucket_hooks(const std::string& phase, int bucket) {
    fs::path dir = fs::path(HOOK_ROOT) / (phase + ".d");
    if (!fs::exists(dir) || !fs::is_directory(dir)) return;

    std::vector<fs::directory_entry> bucket_entries;
    std::vector<fs::directory_entry> wildcard_entries;

    for (auto& de : fs::directory_iterator(dir)) {
        if (!is_executable(de)) continue;
        std::string fname = de.path().filename().string();
        if (fname.empty()) continue;

        // Wildcard: first char is NOT a digit -> always run on bucket change
        if (!std::isdigit(static_cast<unsigned char>(fname[0]))) {
            wildcard_entries.push_back(de);
            continue;
        }

        // Bucket script: parse leading integer prefix (up to 3 digits)
        int i = 0;
        while (i < static_cast<int>(fname.size()) && std::isdigit(static_cast<unsigned char>(fname[i])) && i < 3) {
            ++i;
        }
        if (i == 0) continue;
        int val = 0;
        try {
            val = std::stoi(fname.substr(0, i));
        } catch (...) {
            continue;
        }
        // Only exact buckets in range 0-100 are valid
        if (val < 0 || val > 100) continue;
        if (val == bucket) bucket_entries.push_back(de);
    }
    std::sort(bucket_entries.begin(), bucket_entries.end(),
              [](const auto& a, const auto& b){
                  return a.path().filename().string() < b.path().filename().string();
              });
    for (auto& de : bucket_entries) run_hook_file(de.path());

    std::sort(wildcard_entries.begin(), wildcard_entries.end(),
              [](const auto& a, const auto& b){
                  return a.path().filename().string() < b.path().filename().string();
              });
    for (auto& de : wildcard_entries) run_hook_file(de.path());
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

// ========================= Battery discovery =========================
struct BatteryPaths {
    fs::path root;
    fs::path capacity;
    fs::path status;
    fs::path voltage_now;
};

static std::optional<BatteryPaths> find_battery() {
    auto has_required = [](const fs::path& d){
        return fs::exists(d/"capacity") && fs::exists(d/"status") && fs::exists(d/"voltage_now");
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
            bp.root = de.path();
            bp.capacity = de.path()/"capacity";
            bp.status = de.path()/"status";
            bp.voltage_now = de.path()/"voltage_now";
            return bp;
        }
    }

    // Fallback: any power_supply that has required files
    for (auto& de : fs::directory_iterator(base)) {
        if (has_required(de.path())) {
            BatteryPaths bp;
            bp.root = de.path();
            bp.capacity = de.path()/"capacity";
            bp.status = de.path()/"status";
            bp.voltage_now = de.path()/"voltage_now";
            return bp;
        }
    }

    return std::nullopt;
}

// ========================= Map file =========================
struct MapVals {
    int V_FULL = DEFAULT_V_FULL;  // mV
    int V_EMPTY = DEFAULT_V_EMPTY; // mV (fixed)
    int V_RAW0 = DEFAULT_V_RAW0;  // mV
};

static MapVals load_map(const fs::path& path) {
    MapVals m;
    std::ifstream f(path);
    if (!f) return m;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("V_FULL=", 0) == 0) m.V_FULL = std::atoi(line.c_str()+7);
        else if (line.rfind("V_EMPTY=", 0) == 0) m.V_EMPTY = std::atoi(line.c_str()+8);
        else if (line.rfind("V_RAW0=", 0) == 0) m.V_RAW0 = std::atoi(line.c_str()+7);
    }
    return m;
}

static void save_map_atomic(const fs::path& path, const MapVals& m) {
    std::string data;
    data += "V_FULL=" + std::to_string(m.V_FULL) + "\n";
    data += "V_EMPTY=" + std::to_string(m.V_EMPTY) + "\n";
    data += "V_RAW0=" + std::to_string(m.V_RAW0) + "\n";
    fs::create_directories(path.parent_path());
    (void)write_atomic(path, data, 0644);
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

static inline bool is_charging(const std::string& st) {
    return (st == "Charging" || st == "Full");
}

static int blend_percent(int raw_adj, int voltage_now_mv, const MapVals& m, const std::string& status) {
    raw_adj = clampi(raw_adj, 0, 100);

    int voltage_percent = raw_adj; // fallback
    int weight_volt = 50;
    int weight_raw = 50;

    if (m.V_FULL > m.V_EMPTY && voltage_now_mv > 0) {
        int range = m.V_FULL - m.V_EMPTY;
        int pos = voltage_now_mv - m.V_EMPTY;
        pos = clampi(pos, 0, range);
        voltage_percent = (pos * 100) / range;
        voltage_percent = clampi(voltage_percent, 0, 100);

        // near-empty bias: higher voltage weight near empty
        weight_volt = ((range - pos) * 100) / range; // 0 near full, 100 near empty
        weight_volt = clampi(weight_volt, 5, 95);
        weight_raw = 100 - weight_volt;
    }

    if (is_charging(status)) {
        if (weight_volt > MAX_CHARGE_VOLT_WEIGHT) weight_volt = MAX_CHARGE_VOLT_WEIGHT;
        weight_raw = 100 - weight_volt;
    }

    int blended = (weight_raw * raw_adj + weight_volt * voltage_percent) / 100;
    return clampi(blended, 0, 100);
}

static int step_limit(int last, int target, const std::string& status) {
    if (last < 0) return target; // first value
    if (is_charging(status)) {
        if (target <= last) return last;           // never decrease while charging
        if (target <= last + 1) return target;     // rise at most +1
        return last + 1;
    } else {
        if (target >= last) return last;           // never increase while discharging/unknown
        if (target >= last - 1) return target;     // drop at most -1
        return last - 1;
    }
}

static int bucket5(int percent) {
    percent = clampi(percent, 0, 100);
    return (percent / 5) * 5;
}

// ========================= Signal handling =========================
static void handle_signal(int) { g_running = false; }

// ========================= Main =========================
int main() {
    // Signals
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    std::signal(SIGUSR1, handle_wakeup);

    // Ensure directories exist
    fs::create_directories(fs::path(HOOK_ROOT));
    fs::create_directories(fs::path(HOOK_ROOT) / "charging.d");
    fs::create_directories(fs::path(HOOK_ROOT) / "discharging.d");

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
    int last_percent = -1;
    int last_bucket = -1;
    bool raw0_armed = false;
    bool raw0_written = false; // per process

    int cur_interval = BASE_INTERVAL_S;

    while (g_running) {
        // Read status, capacity, voltage
        std::string status = slurp(bp.status).value_or("Unknown");
        int raw_capacity = slurp_int(bp.capacity).value_or(-1);
        raw_capacity = clampi(raw_capacity, 0, 100);
        int voltage_raw_mv = read_voltage_mv(bp.voltage_now);

        // Median-of-3 then EMA for live voltage (for calculations only)
        if (sv.prev1 < 0) sv.prev1 = (voltage_raw_mv > 0 ? voltage_raw_mv : map.V_FULL);
        if (sv.prev2 < 0) sv.prev2 = sv.prev1;
        int v_med = median3(sv.prev2, sv.prev1, voltage_raw_mv > 0 ? voltage_raw_mv : sv.prev1);
        sv.prev2 = sv.prev1;
        sv.prev1 = (voltage_raw_mv > 0 ? voltage_raw_mv : sv.prev1);
        if (sv.ema < 0) sv.ema = v_med;
        else sv.ema = (ALPHA_NUM * v_med + (ALPHA_DEN - ALPHA_NUM) * sv.ema) / ALPHA_DEN;
        int voltage_ema_mv = sv.ema;

        // Arm RAW0 once raw% > 10
        if (!raw0_written && !raw0_armed && raw_capacity >= RAW0_ARM_THRESHOLD) {
            raw0_armed = true;
        }

        // Stretch raw via V_RAW0
        int scale_mil = 1000;
        if (map.V_RAW0 > 0 && map.V_FULL > map.V_RAW0) {
            scale_mil = 1000 * (map.V_FULL - map.V_EMPTY) / (map.V_FULL - map.V_RAW0);
            if (scale_mil <= 0) scale_mil = 1000;
        }
        int raw_adj = raw_capacity;
        raw_adj = (raw_adj * scale_mil) / 1000;
        raw_adj = clampi(raw_adj, 0, 100);

        // Blended target
        int target = blend_percent(raw_adj, voltage_ema_mv, map, status);

        // Adaptive interval + snap
        int gap = (last_percent < 0) ? 0 : std::abs(last_percent - target);

        // Determine percent
        int percent;
        // Snap if woken explicitly or gap exceeds threshold
        if (g_wakeup.exchange(false) || (last_percent >= 0 && gap > SNAP_DELTA)) {
            percent = target;
            cur_interval = BASE_INTERVAL_S;
        } else {
            if (!is_charging(status) && gap >= 2)
                cur_interval = FAST_INTERVAL_S;
            else if (gap <= 1)
                cur_interval = BASE_INTERVAL_S;

            percent = step_limit(last_percent, target, status);
        }

        // Update V_FULL when charging/full and raw >= 99 using RAW instantaneous voltage
        if (is_charging(status) && raw_capacity >= 99 && voltage_raw_mv > 0) {
            if (std::abs(map.V_FULL - voltage_raw_mv) >= 10) {
                map.V_FULL = voltage_raw_mv;
                save_map_atomic(MAP_FILE, map);
            }
        }

        // Learn V_RAW0 once (armed at raw > 10), when raw hits 0 and not yet written
        if (raw0_armed && !raw0_written && raw_capacity == 0 && !is_charging(status)) {
            int v_raw0 = v_med; // median-of-3 at this tick
            if (v_raw0 > 0) {
                map.V_RAW0 = v_raw0;
                save_map_atomic(MAP_FILE, map);
                raw0_written = true;
                raw0_armed = false;
            }
        }

        // Guardrail: if raw0 wasn't learned and we reach/below V_EMPTY while discharging,
        // set V_RAW0 = V_EMPTY to ensure a sane stretch point on devices that never hit raw0%
        if (raw0_armed
            && !raw0_written
            && !is_charging(status)
            && voltage_raw_mv > 0
            && voltage_raw_mv <= map.V_EMPTY) {
            map.V_RAW0 = map.V_EMPTY;
            save_map_atomic(MAP_FILE, map);
            raw0_written = true;
            raw0_armed  = false;
        }

        // Write percent file only on change (atomic)
        if (percent != last_percent) {
            fs::create_directories(fs::path(PERCENT_FILE).parent_path());
            (void)write_atomic(PERCENT_FILE, std::to_string(percent) + "\n", 0644);
            last_percent = percent;
        }

        // Bucket hooks
        int b = bucket5(percent);
        if (b != last_bucket) {
            if (is_charging(status)) run_bucket_hooks("charging", b);
            else run_bucket_hooks("discharging", b);
            last_bucket = b;
        }

        // Sleep
        for (int i = 0; i < cur_interval && g_running; ++i) {
            if (g_wakeup.exchange(false)) break;
            std::this_thread::sleep_for(1s);
        }
    }

    return 0;
}
