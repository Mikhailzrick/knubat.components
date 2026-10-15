# BatteryPlus

BatteryPlus is a battery reporting daemon for Linux handhelds. It provides steady battery percentages, tracks charge-state changes, and runs hooks for LEDs, warnings, shutdown, and other device actions.

- **Voltage mode:** Estimates percent using filtered voltage, learned full-voltage values, and charging compensation.
- **PMIC mode:** Uses the percent reported by the battery controller.
- **Integration:** Exports percent and charge state through files in `/tmp`, with hooks for percentage changes, state changes, startup, and resume.

## Features

### Voltage-based reporting

Voltage mode uses one discharge curve, with a charging offset removed before calculating percent.

It reads `voltage_now`, or `voltage_avg` if `voltage_now` is absent. A median-of-three filter rejects brief spikes, then an exponential moving average smooths the result.

Calibration learns the charging peak and settled discharge full voltage for the device.

### Smooth percentage updates

BatteryPlus separates the calculated percent from the shown percent. Normal changes move one point at a time, never downward while charging or upward while discharging.

Update timing depends on the gap between calculated and shown percent:

- 1–2 points: 120 seconds.
- 3–5 points: 60 seconds.
- 6 or more points: 30 seconds.
- While charging, these times are doubled.
- Reported or detected full restores normal timing.

These are minimum waits, not scheduled increases. The calculated percent must also allow the change.

Voltage mode holds charging at no more than 99% until full is reported or detected. Some devices may remain at 99% for an extended period.

Startup and resume can restore or replace the shown value in one step.

### Charging compensation

Charging raises measured voltage. BatteryPlus subtracts an offset before calculating percent.

After a normal plug-in, three readings measure the voltage rise from the last discharge reading. Further readings can increase this measured offset over the next 30 seconds. It then tapers toward the difference between the charging and discharge full voltages as voltage rises.

Without a usable plug-in measurement, BatteryPlus uses a fallback based on four times that full-voltage difference, capped at 300 mV before tapering.

If the measured offset was established at or above the charging full-voltage anchor, it stays fixed because there is no remaining voltage range over which to taper.

Full detection uses uncompensated voltage.

### Full detection

Voltage mode accepts full through three paths:

- **Controller Full:** accepts the confirmed controller state without waiting for either voltage-based timer.
- **Peak dwell:** 45 minutes without a new filtered charging peak.
- **Peak drop:** raw and filtered voltage both stay at least 10 mV below the tracked peak for 60 seconds.

Peak tracking begins at `V_FULL_CHG - 50 mV`. Every new filtered peak restarts dwell.

Voltage can fall when charging finishes, making a sustained peak drop useful when the controller does not report Full promptly. Checking raw voltage alongside filtered voltage prevents a recovered dip from qualifying while the filter remains low.

Drop confirmation resets if either voltage recovers or a read fails. Voltage is normally sampled every 15 seconds, so filtering and sampling can add delay.

Discharging, detected resume, or a main-loop gap of at least 3 seconds clears tracking. Otherwise, accepted full remains active for the charging session, with the peak retained for calibration.

### Startup and resume

Voltage mode saves the shown percent, voltage, and charge direction on a clean stop. Startup uses this saved state; resume uses the state held before suspension.

BatteryPlus compares the previous and current voltage:

- Small changes preserve the shown percent.
- Moderate changes blend it with a fresh estimate.
- Large changes use the fresh estimate.

Charge-state changes require one second of confirmation. Resume is detected through system clocks, then voltage is sampled for reconciliation.

An established charging offset survives a charging-to-charging resume. A charge-direction change during resume clears it. Normal plug-in while running measures a new offset.

## Hooks

Hooks are executable scripts run when battery events occur.

### Folders

- `/etc/batteryplus/charging.d/` — charging and full.
- `/etc/batteryplus/discharging.d/` — discharging.
- `/etc/batteryplus/state.d/` — confirmed charge-state changes.

Charging and discharging hooks run on shown-percent changes, startup, resume, and charge-direction changes.

State hooks run at startup and when the confirmed state changes.

### Script arguments

- `$1` — shown percent, from `0` to `100`.
- `$2` — `Charging`, `Discharging`, or `Full`.

Every executable script in the selected folder is launched. Each script must check its own percentage threshold.

```sh
#!/bin/sh

# Act only at 0% while discharging.
[ "$1" = "0" ] || exit 0
[ "$2" = "Discharging" ] || exit 0

# Add the device's low-battery action here.
```

### Execution rules

- Charging hooks launch in reverse filename order.
- Discharging and state hooks launch in filename order.
- Hooks run without blocking BatteryPlus.
- Launch order does not guarantee finish order.
- Restart BatteryPlus after adding or removing hooks.
- Hook input and output go to `/dev/null`. Scripts must write their own logs.

Hooks can run again after resume or a state change. Long-running scripts should prevent duplicate work.

Full inferred from dwell or a confirmed peak drop does not change the controller state. A hook may receive `100 Charging`. Check `$1` when the action depends on the shown percent.

## Configuration

System config: `/etc/batteryplus/batteryplus.conf`

```ini
[Config]
mode=voltage
data_dir=/userdata/system/configs/batteryplus
V_EMPTY=3250
debug=false
```

- **`mode`** — `voltage` or `pmic`. Default: `voltage`.
- **`data_dir`** — absolute path for saved values, state, and logs.
- **`V_EMPTY`** — voltage used for 0%. Default: `3250` mV. Allowed range: `3000–3400`.
- **`debug`** — enable event logging with `true`. Default: `false`.

Voltage values are in millivolts. `V_EMPTY` only affects voltage mode.

If `data_dir` is blank, BatteryPlus tries `$XDG_CONFIG_HOME/batteryplus`, then `$HOME/.config/batteryplus`. The base path must be absolute.

Startup fails if no valid data path is available or its folder cannot be created.

### Local overrides

`<data_dir>/batteryplus.conf` can override all settings except `data_dir`.

Invalid `V_EMPTY` values retain the value from the previous config layer. Restart BatteryPlus after changing settings.

The old `V_EMPTY_CHG` and `V_EMPTY_DIS` settings are no longer used.

## Files

### Runtime output

- **`/tmp/battery.percent`** — shown percent, such as `87`.
- **`/tmp/battery.status`** — confirmed controller state.

Read these files from your UI or scripts.

For inotify, watch the percent file's folder because the file is replaced on each write. The status file is written in place and supports `close_write` watches.

The status file does not change to `Full` when dwell or a confirmed peak drop detects full. It reflects the confirmed controller state.

### Saved files

These files are stored in `data_dir`:

- **`batteryplus-voltage.map`** — learned full-voltage values.
- **`batteryplus-calibrated`** — flag that stops further calibration.
- **`batteryplus-restore.state`** — state saved on clean exit and read once at startup.
- **`batteryplus-debug.log`** — event log when debug is enabled.

### Voltage map

```ini
V_FULL_CHG=4050
V_FULL_DIS=4000
```

- `V_FULL_CHG` is the highest filtered charging voltage learned during calibration.
- `V_FULL_DIS` is the settled voltage after unplugging from full.

These example values are also the defaults.

Both values must be within `3600–4600` mV. Discharge full voltage cannot exceed charging full voltage.

Missing or invalid values fall back to defaults. The calibration flag still controls whether learning is allowed.

## Debug logging

Set `debug=true` and restart BatteryPlus.

`<data_dir>/batteryplus-debug.log` records startup and resume decisions, percentage and state changes, charging compensation, full detection, calibration, and selected runtime failures.

Entries include uptime and an event name. This is an event log, not a continuous voltage trace, and it has no built-in size limit. Startup errors go to stderr.

## Calibration

Calibration learns the full-voltage values for your device. It runs in voltage mode while `batteryplus-calibrated` is absent.

### Steps

1. Start BatteryPlus with the device powered on.
2. Plug in a charger.
3. Keep the device awake and mostly idle until the shown percent reaches 100%.
4. Unplug the charger.
5. Leave the device on and mostly idle for up to 60 seconds.

BatteryPlus must observe charging before full. Starting with the controller already reporting Full does not provide a peak to learn.

### After unplugging

Sampling starts after 8 seconds. Each burst takes three readings one second apart, with bursts repeating about every 10 seconds.

Sampling finishes when two consecutive changes between burst results are no more than 5 mV, normally around 30 seconds if voltage settles quickly. At the 60-second timeout, BatteryPlus uses the median of the later sampled values if available.

A valid voltage pair is saved and calibration is locked. Full detection remains active. Sampling does not block the main loop; replugging or resuming cancels the attempt.

### Calibrate again

Remove `<data_dir>/batteryplus-calibrated` and restart BatteryPlus.

Keep the voltage map unless you also want to return to default full-voltage values.

## PMIC mode

Use `mode=pmic` when the controller's percentage already works well.

BatteryPlus reads `capacity` and applies the same adaptive timing, charging multiplier, one-point updates, state checks, and hooks.
Voltage estimation, charging compensation, calibration, and voltage-based restore and full detection are unused. Controller-reported Full still restores normal update timing.

## Signals

- **`SIGTERM` or `SIGINT`** — stop cleanly and save valid restore state in voltage mode.
- **`SIGUSR1`** — ignored for now.

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

For normal use, use start-stop-daemon with a pidfile. Send SIGTERM on stop and wait for the process to exit before restarting or shutting down.

This gives BatteryPlus time to save restore state. A restore file is only written in voltage mode when valid readings are available.

## Developer tuning

These values are in `batteryplus.cpp`, not the config file. Rebuild after changing them.

### Update timing

- `INTERNAL_INTERVAL_S = 15` — normal calculation interval.
- `WRITE_INTERVAL_S = 120` — normal shown-percent interval.
- `WRITE_INTERVAL_DELTA_SMALL_S = 60` — interval for a 3–5 point gap.
- `WRITE_INTERVAL_DELTA_LARGE_S = 30` — interval for a gap of 6 or more.
- `CHARGE_VISIBLE_INTERVAL_MULTIPLIER = 2` — doubles adaptive intervals while charging; reported or inferred full uses normal timing.
- `STATUS_CONFIRM_S = 1` — charge-state confirmation delay.

Intervals are in seconds. The charging multiplier is unitless.

### Filtering and curve shape

`ALPHA_NUM / ALPHA_DEN` sets how much each new reading affects the filtered voltage. The default is `2 / 10`, or 20%. A smaller share gives smoother but slower results.

These settings control `voltage_to_percent()`:

- `SCURVE_STRENGTH_LOW` and `SCURVE_STRENGTH_HIGH` shape the voltage curve.
- `DISCHARGE_100_WINDOW_PERCENT` sets the full-voltage plateau.
- `MIN_RANGE_MV` keeps at least 500 mV between empty and the start of that plateau.

Calibration changes the full-voltage anchors. It does not learn a new curve shape.

### Charging offset

- `CHARGE_OFFSET_MULTIPLIER = 4.0` — base-offset multiplier applied to the full-voltage gap before the cap and taper.
- `CHARGE_COMP_REFINE_WINDOW_S = 30` — time to refine plug-in inflation.
- `CHARGE_COMP_MAX_INFLATION_MV = 300` — cap for measured plug-in inflation and the base offset before tapering.

The multiplier is inside `base_charge_offset()`. A usable measured plug-in offset takes precedence over the base estimate.

### Full detection

- `PEAK_DWELL_S = 2700` — 45 minutes without a new filtered charging peak.
- `PEAK_TRACK_RANGE_MV = 50` — how far below the charging full anchor peak tracking can begin.
- `PEAK_FULL_DROP_MV = 10` — required drop below the tracked peak in both raw and filtered voltage.
- `PEAK_FULL_DROP_CONFIRM_S = 120` — how long qualifying samples must span before accepting peak-drop full.

### Unplug sampling

- `VFULL_DIS_INITIAL_SETTLE_S = 8` — wait before the first burst.
- `VFULL_DIS_CHECK_S = 8` — wait between bursts.
- `VFULL_DIS_STABLE_MV = 5` — allowed change between burst results.
- `VFULL_DIS_STABLE_COUNT = 2` — stable changes needed.
- `VFULL_DIS_MAX_WAIT_S = 60` — sampling timeout.

Each burst takes three readings, one second apart.

### Startup and resume restore

The `RESTORE_*` constants set the allowed voltage fall and rise:

- Discharging → Discharging: fall 50 mV, rise 35 mV.
- Charging → Charging: fall 35 mV, rise 75 mV.
- Discharging → Charging: fall 35 mV, rise 300 mV.
- Charging → Discharging: fall 300 mV, rise 35 mV.

Within the limit, keep the saved percent. Between the limit and 1.5 times that limit, blend saved and fresh values. Beyond that, use the fresh value.

`RESTORE_BLEND_SCALE = 1.5` sets the blend range.

### Boot adjustment

- `BOOT_COMP_MAX_UPTIME_S = 60` — boot adjustment applies within this time.
- `BOOT_COMP_MV = 25` — voltage added to the fresh discharge startup estimate.
- `BOOT_EMPTY_FRESH_MARGIN_MV = 50` — skip saved-percent restore this close to empty at startup.

## Limits

Battery percent from voltage is an estimate. Load, temperature, and battery age can affect it.

- Smooth output does not prove exact remaining charge.
- Calibration learns full-voltage values, not battery capacity.
- A sustained load-induced voltage drop can resemble charging completion. The confirmation period rejects brief dips but cannot distinguish every cause of a prolonged drop.

## License

BatteryPlus is licensed under the GNU General Public License version 2.

See the license text in the source header or the included license file.
