# BatteryPlus

**Great Scott! 1.21 Gigawatts!!**

BatteryPlus is an alternative battery percentage daemon for handheld Linux systems.

It is designed for devices where the built-in PMIC-reported battery percentage is inaccurate, unstable, jumpy, or unsuitable for a smooth user-facing battery display.

BatteryPlus can run in two modes:

- **Voltage mode** — calculates battery percent from battery voltage.
- **PMIC mode** — reads the kernel/PMIC-reported capacity directly, while still using BatteryPlus output and hook handling.

The main goal is simple: provide a calmer, more believable, and more useful battery percentage for handheld devices.

---

## Why BatteryPlus?

Many handheld Linux devices report battery percentage poorly. Common issues include:

- Percent drops too quickly at the top.
- Percent stays high for too long, then collapses.
- Battery jumps up or down after suspend/resume.
- PMIC calibration is inaccurate.
- Different devices behave inconsistently.
- UI battery display feels erratic or untrustworthy.

BatteryPlus solves this by using voltage-based reporting with smoothing, shaped charge/discharge curves, full-voltage learning, and step-limited visible output.

---

## Features

### Voltage-based battery percentage

In voltage mode, BatteryPlus reads `/sys/class/power_supply/*/voltage_now`.

It converts battery voltage into a percentage using:

- Separate charging and discharging curves
- Separate empty-voltage anchors
- Separate learned full-voltage anchors
- Median-of-3 filtering
- EMA smoothing
- Step-limited visible output

This avoids relying on inaccurate PMIC capacity values.

---

### Separate charge and discharge behavior

Battery voltage behaves differently while charging versus discharging.

BatteryPlus tracks these separately:

- `V_FULL_CHG` — learned full voltage while charging
- `V_FULL_DIS` — learned settled full voltage shortly after unplugging from full
- `V_EMPTY_CHG` — configured empty voltage while charging
- `V_EMPTY_DIS` — configured empty voltage while discharging

This helps account for voltage relaxation after unplugging and improves the 100% behavior while discharging.

---

### Automatic full-voltage learning

BatteryPlus can automatically learn full-voltage anchors.

While charging, it tracks the highest stable charging voltage near the top of the battery range. A full event can be detected from either:

- Kernel power-supply status reporting `Full`
- A stable charging-voltage peak that persists long enough

After a valid full event, BatteryPlus waits briefly after unplugging and records the settled discharge-side full voltage.

This creates a better discharge-side 100% anchor than using charger-inflated voltage directly.

---

### Smooth visible percentage

BatteryPlus separates internal percent from visible percent.

The internal percent updates regularly, but the visible percent is written more calmly to `/tmp/battery.percent`.

Visible percent behavior:

- Charging percent does not count down.
- Discharging percent does not count up.
- Visible percent normally changes one point at a time.
- Larger internal/visible deltas shorten the update interval.
- Long resume gaps can trigger a burst sample and snap update.

This makes the UI feel more stable without hiding real battery movement for too long.

---

### Resume-aware behavior

BatteryPlus handles suspend/resume-style gaps.

It supports `SIGUSR1` as an explicit resume hint. On resume or long loop gaps, BatteryPlus can:

- Abort in-progress full-voltage calibration
- Force an immediate visible update
- Burst-sample voltage after long gaps
- Snap visible percent when the previous value is likely stale
- Run wildcard hooks if no bucket hook already fired

---

### Hook system

BatteryPlus includes a simple hook system for device-specific behavior.

Hook directories:

- `/etc/batteryplus/charging.d/`
- `/etc/batteryplus/discharging.d/`
- `/etc/batteryplus/state.d/`

Charging and discharging hooks may use numeric prefixes for exact 5% buckets.

Examples:

- `/etc/batteryplus/discharging.d/20-low-power`
- `/etc/batteryplus/discharging.d/10-critical`
- `/etc/batteryplus/charging.d/100-full`

Numeric hook names must target 5% increments:

`0, 5, 10, 15, ... 100`

Non-numeric hook names are treated as wildcards and run on every bucket or resume event.

Example:

`/etc/batteryplus/discharging.d/update-leds`

State hooks run when the charge state changes.

Hooks receive:

- `$1` = visible percent
- `$2` = charge state string

Example hook:

    #!/bin/sh

    PERCENT="$1"
    STATE="$2"

    echo "BatteryPlus: ${PERCENT}% ${STATE}" >> /tmp/batteryplus-hook.log

---

## Files

### Runtime output

`/tmp/battery.percent`

Contains the current visible battery percentage.

Example:

`87`

This file is intended for UI polling.

---

### Config file

`/etc/batteryplus/batteryplus.conf`

Example:

    [Config]
    mode=voltage
    data_dir=/userdata/system/configs/batteryplus
    V_EMPTY_CHG=3400
    V_EMPTY_DIS=3250

---

### Data directory

The configured `data_dir` must be an absolute path to a persistent directory.

BatteryPlus stores learned and temporary state here.

Example:

`/userdata/system/configs/batteryplus/`

Files:

- `batteryplus-voltage.map`
- `batteryplus-calibrated`
- `batteryplus-restore.state`

---

### Voltage map

`<data_dir>/batteryplus-voltage.map`

Stores learned voltage anchors:

    V_FULL_CHG=4050
    V_FULL_DIS=4000

If the map is missing or invalid, BatteryPlus falls back to safe defaults.

---

### Calibration flag

`<data_dir>/batteryplus-calibrated`

This is a presence-only flag.

It is created after both charge-side and discharge-side full anchors have been learned during the calibration flow.

---

### Restore state

`<data_dir>/batteryplus-restore.state`

Temporary one-shot restore file.

BatteryPlus writes this on clean exit and consumes it on next startup if the current voltage and charge state still match closely enough.

This helps avoid visible battery jumps when the daemon restarts.

---

## Configuration

### Required

`data_dir=/absolute/persistent/path`

BatteryPlus requires `data_dir`.

If `data_dir` is missing, the daemon exits with an error.

---

### Optional

`mode=voltage`

Supported modes:

- `voltage`
- `pmic`

Default:

`voltage`

---

### Empty voltage anchors

    V_EMPTY_CHG=3400
    V_EMPTY_DIS=3250

Defaults:

- `V_EMPTY_CHG = 3400 mV`
- `V_EMPTY_DIS = 3250 mV`

These values represent the configured 0% anchors.

Charging empty voltage is intentionally higher than discharging empty voltage.

BatteryPlus validates these values at startup and falls back to defaults if they are outside the accepted range.

---

## Calibration

For best voltage-mode accuracy, BatteryPlus should be allowed to learn the device’s full-voltage anchors.

Basic calibration flow:

1. Boot the device.
2. Plug in a wall charger while the device is powered on.
3. Let it charge until BatteryPlus reaches 100%.
4. Keep the device powered on during this process.
5. After it reaches 100%, unplug the charger.
6. Wait briefly so BatteryPlus can record the settled discharge-side full voltage.

BatteryPlus will create `<data_dir>/batteryplus-calibrated` after both full anchors have been learned.

If the device is charged while powered off, then unplugged and booted later, BatteryPlus may not have a valid discharge-side full anchor. In that case, calibration may need to be repeated while the device is powered on.

---

## Signals

### Stop daemon

- `SIGTERM`
- `SIGINT`

Stops BatteryPlus and saves restore state when possible.

---

### Resume hint

`SIGUSR1`

Signals that the system has resumed or that BatteryPlus should immediately refresh its calculation path.

Example:

    kill -USR1 "$(cat /run/batteryplus.pid)"

A udev rule or resume hook can use this after suspend/resume.

---

## Build

Example cross-compile command:

    aarch64-linux-gnu-g++ -O3 -flto -std=gnu++20 -Wall -Wextra -pedantic batteryplus.cpp -o batteryplus

## Example init usage

Example minimal service script:

    #!/bin/sh

    case "$1" in
      start)
        /usr/bin/batteryplus &
        ;;
      stop)
        killall batteryplus 2>/dev/null
        ;;
    esac

    exit 0

A real init script should usually use start-stop-daemon with a pidfile and send SIGTERM on stop. This allows BatteryPlus to save restore state cleanly before exiting.

---

## Example resume hint

BatteryPlus supports `SIGUSR1` as an explicit refresh/resume hint.

A suspend/resume script or device power-management script can call:

    kill -USR1 "$(cat /run/batteryplus.pid)"

This tells BatteryPlus to refresh immediately after resume, abort any in-progress calibration timing, and update visible percent sooner than the normal write interval.

---

## PMIC mode

PMIC mode can be useful on devices where the kernel-reported capacity is already acceptable, but you still want BatteryPlus hooks and output behavior.

Example config:

    [Config]
    mode=pmic
    data_dir=/userdata/system/configs/batteryplus

In PMIC mode, BatteryPlus reads `/sys/class/power_supply/*/capacity`.

---

## Notes

BatteryPlus is designed for embedded handheld Linux environments.

It does not try to be a desktop power manager. It does not manage charging hardware, battery safety, suspend policy, or shutdown policy by itself.

It reports a stable user-facing percentage and provides hooks so the surrounding system can react to battery level and charge state changes.

Recommended integrations include:

- UI battery display
- Low-battery warnings
- LED behavior
- Performance profile changes
- Auto-shutdown policy
- Suspend/resume refresh hooks

---

## License

BatteryPlus is licensed under the GNU General Public License version 2.

See the license text in the source header or the included license file.
