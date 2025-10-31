// Lightweight evdev-based activity monitor
// States: ACTIVE, IDLE, EXTENDED
// Timers loaded from congif with sane defaults (min 60s / max 12h)
// Writes 1/0 to /var/run/idle.state (1 = ACTIVE, 0 = IDLE/EXTENDED)
// Runs hook scripts in /etc/idlewatcher/{idle.d,extended.d,active.d}
//   IDLE -> run scripts with arg "idle"
//   EXTENDED -> run scripts with arg "extended"
//   ACTIVE -> run scripts with arg "active"
// Config has hooks_mirror= for secondary/mirrored hook scripts location
// Watches /dev/input/event*
// EV_ABS counts as activity only if delta ≥ AXIS_DZ_PCT
// Throttles pulses to reduce excessive work
// Creates hook directories on startup if missing
// Build arm: aarch64-linux-gnu-g++ -O3 -flto -std=gnu++20 -Wall -Wextra -pedantic idlewatcher.cpp -o idlewatcher
// Build x86: g++ -O3 -flto -std=gnu++20 -Wall -Wextra -pedantic idlewatcher.cpp -o idlewatcher

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <linux/input.h>
#include <string>
#include <sys/epoll.h>
#include <sys/inotify.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/timerfd.h>
#include <time.h>
#include <unistd.h>
#include <vector>

// ========== Config and constants ==========
static constexpr int    DEFAULT_IDLE_S      = 900;  // 15 minutes
static constexpr int    DEFAULT_EXTENDED_S  = 3600; // 60 minutes
static constexpr double DEFAULT_AXIS_DZ_PCT = 0.15; // 15%
static constexpr int    AXIS_DZ_MIN         = 64;   // small floor
static constexpr int    AXIS_DZ_BADSPAN     = 128;  // fallback just in case
static double AXIS_DZ_PCT = DEFAULT_AXIS_DZ_PCT;    // runtime deadzone, overriden by config

static const int DEBOUNCE_MS = 3000; // global debounce

static const char* CONFIG_FILE = "/etc/idlewatcher/idlewatcher.conf";
static const char* STATE_FILE = "/var/run/idle.state";
static const char* HOOKS_ROOT = "/etc/idlewatcher";
static const char* INPUT_DIR  = "/dev/input";
static std::string HOOKS_MIRROR;  // optional secondary hooks

static inline void die(const char* fmt, ...) __attribute__((noreturn));
static inline void die(const char* fmt, ...) {
  va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); fprintf(stderr, "\n"); va_end(ap); exit(1);
}

// ========== Types and globals ==========
enum class State{ACTIVE,IDLE,EXTENDED};

struct Dev{
  int fd{-1};
  std::string path;

  int  abs_last[ABS_MAX+1];
  bool abs_seen[ABS_MAX+1];
  int  abs_min[ABS_MAX+1];
  int  abs_max[ABS_MAX+1];
  int  abs_dz[ABS_MAX+1];

  Dev() {
    memset(abs_last, 0, sizeof(abs_last));
    memset(abs_seen, 0, sizeof(abs_seen));
    memset(abs_min,  0, sizeof(abs_min));
    memset(abs_max,  0, sizeof(abs_max));
    memset(abs_dz,   0, sizeof(abs_dz));
  }
};

struct Runtime{
  int epfd{-1};int tfd{-1};int ifd{-1};
  std::vector<Dev> devices;
  int64_t last_activity_ms{0};
  int64_t last_pulse_ms{0};
  State state{State::ACTIVE};
  int idle_s{DEFAULT_IDLE_S};
  int extended_s{DEFAULT_EXTENDED_S};
}RT;

// ========== Helpers ==========
static inline bool is_hat_abs(int code) {
  return code >= ABS_HAT0X && code <= ABS_HAT3Y;
}

static inline int64_t now_ms() {
  timespec ts{}; clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
  return (int64_t)ts.tv_sec*1000 + ts.tv_nsec/1000000;
}

static int parse_pos_int(const char* s) {
  long v = strtol(s, nullptr, 10);
  if (v < 0 || v > 12*60*60) return -1; // clamp to 12h max
  return (int)v;
}

// simple check for "event" + digits
static inline bool is_event_name(const char* s){
    if (!s) return false;
    if (strncmp(s, "event", 5) != 0) return false;
    const char* p = s + 5;
    if (!*p) return false;
    for (; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
    }
    return true;
}

// ======== Config load and layout ==========
// dirs
static void ensure_dir(const std::string& p){ if(mkdir(p.c_str(),0755)<0 && errno!=EEXIST) die("mkdir %s: %s",p.c_str(),strerror(errno)); }

static void ensure_default_config(){
  struct stat st{};
  if (stat(CONFIG_FILE, &st) == 0) return;
  ensure_dir(HOOKS_ROOT);
  FILE* f = fopen(CONFIG_FILE, "w");
  if (!f) return;
  fprintf(f,
    "[Config]\n"
    "idle=%d\n"
    "extended=%d\n"
    "ABS_Deadzone=%.3f\n"
    "hooks_mirror=\n",
    DEFAULT_IDLE_S,
    DEFAULT_EXTENDED_S,
    DEFAULT_AXIS_DZ_PCT
  );
  fclose(f);
}

static void read_config_or_defaults(int &idle_s, int &extended_s) {
  // defaults
  idle_s      = DEFAULT_IDLE_S;
  extended_s  = DEFAULT_EXTENDED_S;
  AXIS_DZ_PCT = DEFAULT_AXIS_DZ_PCT;
  HOOKS_MIRROR.clear();

  FILE* f = fopen(CONFIG_FILE, "r");
  if (!f) return;

  char line[256];
  while (fgets(line, sizeof(line), f)) {
    // strip leading spaces
    char* p = line;
    while (*p==' '||*p=='\t') ++p;
    // skip blanks & comments and section headers
    if (*p=='\0' || *p=='\n' || *p=='#' || *p=='[') continue;

    // key=value
    char* eq = strchr(p, '=');
    if (!eq) continue;
    *eq = '\0';
    char* key = p;
    char* val = eq+1;

    // trim trailing key spaces
    for (char* t=key + strlen(key); t>key && (t[-1]==' '||t[-1]=='\t'||t[-1]=='\r'||t[-1]=='\n'); --t) t[-1]='\0';
    // trim leading val spaces
    while (*val==' '||*val=='\t') ++val;
    // trim trailing val spaces/newlines
    for (char* t=val + strlen(val); t>val && (t[-1]==' '||t[-1]=='\t'||t[-1]=='\r'||t[-1]=='\n'); --t) t[-1]='\0';

    if (strcmp(key, "idle")==0) {
      int n = parse_pos_int(val);
      if (n >= 60) idle_s = n;
    } else if (strcmp(key, "extended")==0) {
      int n = parse_pos_int(val);
      if (n >= 60) extended_s = n;
    } else if (strcmp(key, "hooks_mirror")==0) {
      if (*val) HOOKS_MIRROR = val;
    } else if (strcmp(key, "ABS_Deadzone")==0) {
      // Accept either percent like "20" or ratio like "0.2"
      char* end = nullptr;
      double v = strtod(val, &end);
      if (end != val) {
        if (v > 1.0) v = v / 100.0;       // treat 20..100 as percent
        if (v < 0.0) v = 0.0;
        if (v > 0.90) v = 0.90;           // clamp sanity: ≤90% of span
        AXIS_DZ_PCT = v;
      }
    }
  }
  fclose(f);

  // enforce minimums
  if (idle_s < 60) idle_s = 60;
  if (extended_s < 60) extended_s = 60;
}

static void ensure_hooks_root_layout(const std::string& root){
    ensure_dir(root);
    ensure_dir(root + "/idle.d");
    ensure_dir(root + "/extended.d");
    ensure_dir(root + "/active.d");
}

// ========== Hooks and state file ========
static void write_state(State s){int val=(s==State::ACTIVE)?1:0;FILE*f=fopen(STATE_FILE,"w");if(f){fprintf(f,"%d\n",val);fclose(f);}}

static void run_folder(const std::string& dir,const char*arg){
  DIR*d=opendir(dir.c_str()); if(!d) return;
  std::vector<std::string>items; dirent*e;
  while((e=readdir(d))){ if(e->d_name[0]=='.') continue; items.emplace_back(e->d_name);}
  closedir(d);
  std::sort(items.begin(),items.end());
  for(auto&fn:items){
    std::string path=dir+"/"+fn; struct stat st{}; if(stat(path.c_str(),&st)!=0) continue;
    if(!S_ISREG(st.st_mode)||(st.st_mode&S_IXUSR)==0) continue;
    pid_t pid=fork(); if(pid==0){ execl(path.c_str(),path.c_str(),arg,(char*)nullptr); _exit(127); }
  }
}

static inline void run_hook_roots(const char* subdir, const char* arg){
    run_folder(std::string(HOOKS_ROOT) + "/" + subdir, arg);
    if (!HOOKS_MIRROR.empty())
        run_folder(HOOKS_MIRROR + "/" + subdir, arg);
}

// ======== State machine and timers ==========
static inline int64_t effective_idle_ms() {
  const int64_t now = now_ms();
  const int64_t delta = now - RT.last_activity_ms;
  return delta > 0 ? delta : 0;
}

// timer helpers
static void arm_timer_ms(int64_t ms){
  itimerspec its{}; // zero struct disarms
  if(ms>0){ its.it_value.tv_sec=ms/1000; its.it_value.tv_nsec=(ms%1000)*1000000; }
  if(timerfd_settime(RT.tfd,0,&its,nullptr)<0) die("timerfd_settime: %s", strerror(errno));
}

static void schedule_timer(){
  const int64_t idle_ms = (int64_t)RT.idle_s * 1000;
  const int64_t ext_ms  = (int64_t)RT.extended_s * 1000;

  if (RT.state == State::EXTENDED) { arm_timer_ms(0); return; }

  const int64_t eff_since = effective_idle_ms();

  if (RT.state == State::ACTIVE) {
    int64_t remain = idle_ms - eff_since;
    arm_timer_ms(remain > 1 ? remain : 1);
    return;
  }
  if (RT.state == State::IDLE) {
    int64_t remain = (idle_ms + ext_ms) - eff_since;
    arm_timer_ms(remain > 1 ? remain : 1);
    return;
  }
  arm_timer_ms(0);
}

static void enter(State to){
    if (to == RT.state) return;
    RT.state = to;
    write_state(to);

    if (to == State::ACTIVE)
        run_hook_roots("active.d", "active");
    else if (to == State::IDLE)
        run_hook_roots("idle.d", "idle");
    else if (to == State::EXTENDED)
        run_hook_roots("extended.d", "extended");
}

static void on_activity(){
  int64_t now=now_ms();
  if(now-RT.last_pulse_ms<DEBOUNCE_MS) return; // global debounce
  RT.last_pulse_ms=now; RT.last_activity_ms=now;
  if(RT.state!=State::ACTIVE) enter(State::ACTIVE);
  schedule_timer();
}

static void reevaluate(){
  const int64_t eff_since = effective_idle_ms();
  const int64_t idle_ms = (int64_t)RT.idle_s * 1000;
  const int64_t ext_ms  = (int64_t)RT.extended_s * 1000;

  if (RT.state == State::ACTIVE) {
    if (eff_since >= idle_ms) enter(State::IDLE);
  } else if (RT.state == State::IDLE) {
    if (eff_since < idle_ms) enter(State::ACTIVE);
    else if (eff_since >= (idle_ms + ext_ms)) enter(State::EXTENDED);
  } else {
    if (eff_since < idle_ms) enter(State::ACTIVE);
  }
  schedule_timer();
}

// ========== Device discovery and input handling =========
static Dev* dev_by_fd(int fd){
  for (auto &d : RT.devices) if (d.fd == fd) return &d;
  return nullptr;
}

static int dev_index_by_fd(int fd){
  for (size_t i=0;i<RT.devices.size();++i) if (RT.devices[i].fd == fd) return (int)i;
  return -1;
}

static int dev_index_by_path(const std::string& path){
  for (size_t i=0;i<RT.devices.size();++i) if (RT.devices[i].path == path) return (int)i;
  return -1;
}

static void del_dev_fd(int fd){
  int idx = dev_index_by_fd(fd);
  if (idx >= 0) {
    epoll_ctl(RT.epfd, EPOLL_CTL_DEL, RT.devices[idx].fd, nullptr);
    close(RT.devices[idx].fd);
    RT.devices.erase(RT.devices.begin()+idx);
  }
}

static void init_abs_info(Dev& d) {
  auto got_abs = [&](int code, input_absinfo& ai)->bool {
    memset(&ai, 0, sizeof(ai));
    return ioctl(d.fd, EVIOCGABS(code), &ai) == 0;
  };
  auto span_of = [](const input_absinfo& ai)->long {
    return (long)ai.maximum - (long)ai.minimum;
  };

  // clear (in case of reuse)
  memset(d.abs_min, 0, sizeof(d.abs_min));
  memset(d.abs_max, 0, sizeof(d.abs_max));
  memset(d.abs_dz,  0, sizeof(d.abs_dz));

  for (int code = 0; code <= ABS_MAX; ++code) {
    input_absinfo ai{};
    if (!got_abs(code, ai)) continue;

    d.abs_min[code] = ai.minimum;
    d.abs_max[code] = ai.maximum;

    if (is_hat_abs(code)) {
      d.abs_dz[code] = 0; // HATs are unfiltered
      continue;
    }

    long span = span_of(ai);
    int dz;
    if (span > 0) {
      dz = (int)std::lround(span * AXIS_DZ_PCT);
      if (dz < AXIS_DZ_MIN) dz = AXIS_DZ_MIN;
    } else {
      dz = AXIS_DZ_BADSPAN; // bad/zero span fallback
    }
    d.abs_dz[code] = dz;
  }
}

static void add_dev(const std::string& path){
  int fd = open(path.c_str(), O_RDONLY|O_NONBLOCK|O_CLOEXEC);
  if (fd < 0) return;

  epoll_event ev{}; ev.events = EPOLLIN; ev.data.fd = fd;
  if (epoll_ctl(RT.epfd, EPOLL_CTL_ADD, fd, &ev) < 0) { close(fd); return; }

  Dev d; d.fd = fd; d.path = path;
  init_abs_info(d); // compute per-device stick DZ once
  RT.devices.push_back(std::move(d));
}

static void del_dev(const std::string& path){
  int idx = dev_index_by_path(path);
  if (idx >= 0) {
    epoll_ctl(RT.epfd, EPOLL_CTL_DEL, RT.devices[idx].fd, nullptr);
    close(RT.devices[idx].fd);
    RT.devices.erase(RT.devices.begin()+idx);
  }
}

static void scan_inputs(){
  DIR*d=opendir(INPUT_DIR); if(!d) die("open %s: %s",INPUT_DIR,strerror(errno));
  dirent*e;
  while((e=readdir(d))){
    if(e->d_name[0]=='.') continue;
    if(!is_event_name(e->d_name)) continue;
    add_dev(std::string(INPUT_DIR)+"/"+e->d_name);
  }
  closedir(d);
}

static void handle_input(int fd){
  Dev* dvp = dev_by_fd(fd); if (!dvp) return;
  Dev& dv = *dvp;
  input_event buf[128]; // bigger buffer to drain faster
  bool pulsed = false; // ensure we only pulse once per batch

  for (;;) {
    ssize_t n = read(fd, buf, sizeof(buf));
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) break;  // fully drained
      del_dev_fd(fd); return;                              // real error
    }
    if (n == 0) { del_dev_fd(fd); return; }                // device gone

    int cnt = n / sizeof(input_event);
    for (int i = 0; i < cnt; ++i) {
      const input_event &e = buf[i];
      if (pulsed) continue; // we already decided; just draining
      if (e.type == EV_SYN) continue;

      switch (e.type) {
        case EV_KEY:
        case EV_REL:
          on_activity(); pulsed = true; break;

        case EV_ABS: {
          int code = e.code, val = e.value;

          if (!dv.abs_seen[code]) { dv.abs_last[code] = val; dv.abs_seen[code] = true; break; }

          int delta = std::abs(val - dv.abs_last[code]);

          if (is_hat_abs(code)) {
            if (delta != 0) { dv.abs_last[code] = val; on_activity(); pulsed = true; }
          } else {
            int dz = dv.abs_dz[code]; // already per-axis
            if (dz <= 0) dz = AXIS_DZ_MIN;
            if (delta >= dz) { dv.abs_last[code] = val; on_activity(); pulsed = true; }
          }
          break;
        }
        default: break;
      }
    }
  }
}

int main(){
  struct sigaction sa{}; sa.sa_handler=SIG_DFL; sa.sa_flags=SA_NOCLDWAIT; sigemptyset(&sa.sa_mask); sigaction(SIGCHLD,&sa,nullptr);
  ensure_hooks_root_layout(HOOKS_ROOT);
  ensure_default_config();
  read_config_or_defaults(RT.idle_s, RT.extended_s);
  if (!HOOKS_MIRROR.empty()) ensure_hooks_root_layout(HOOKS_MIRROR);
  RT.last_activity_ms = now_ms();
  write_state(State::ACTIVE);

  // ======== Event setup ==========
  RT.epfd = epoll_create1(EPOLL_CLOEXEC);
  if (RT.epfd < 0) die("epoll_create1: %s", strerror(errno));

  RT.tfd = timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC);
  if (RT.tfd < 0) die("timerfd_create: %s", strerror(errno));

  {
    epoll_event tev{};
    tev.events = EPOLLIN;
    tev.data.fd = RT.tfd;
    if (epoll_ctl(RT.epfd, EPOLL_CTL_ADD, RT.tfd, &tev) < 0)
      die("epoll add tfd: %s", strerror(errno));
  }

  RT.ifd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
  if (RT.ifd < 0) die("inotify_init1: %s", strerror(errno));

  if (inotify_add_watch(RT.ifd, INPUT_DIR, IN_CREATE | IN_DELETE) < 0)
    die("inotify_add_watch: %s", strerror(errno));

  {
    epoll_event iev{};
    iev.events = EPOLLIN;
    iev.data.fd = RT.ifd;
    if (epoll_ctl(RT.epfd, EPOLL_CTL_ADD, RT.ifd, &iev) < 0)
      die("epoll add ifd: %s", strerror(errno));
  }

  scan_inputs();
  schedule_timer();

  std::vector<char> buf(4096);
  std::array<epoll_event,32> events{};
  while (true) {
    int n = epoll_wait(RT.epfd, events.data(), (int)events.size(), -1);
    if (n < 0) { if (errno == EINTR) continue; die("epoll_wait: %s", strerror(errno)); }
    for (int i = 0; i < n; i++) {
      int fd = events[i].data.fd;
      if (fd == RT.tfd) {
        uint64_t exp; (void)read(RT.tfd, &exp, sizeof(exp));
        reevaluate();
      } else if (fd == RT.ifd) {
        ssize_t r;
        while ((r = read(RT.ifd, buf.data(), buf.size())) > 0) {
          for (char* p = buf.data(); p < buf.data() + r; ) {
            inotify_event* e = (inotify_event*)p;
            if (e->len && is_event_name(e->name)) {
              std::string full = std::string(INPUT_DIR) + "/" + e->name;
              if (e->mask & IN_CREATE) add_dev(full);
              if (e->mask & IN_DELETE) del_dev(full);
            }
            p += sizeof(inotify_event) + e->len;
          }
        }
      } else {
        handle_input(fd);
      }
    }
  }
}
