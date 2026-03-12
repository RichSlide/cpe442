#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh — Raspberry Pi 5 competition benchmark launcher
#
# Disables every non-essential service, interrupt source, and background
# process that could steal CPU time or pollute PAPI counters, then runs
# the sobel filter at real-time priority with the performance governor.
#
# Usage:
#   chmod +x run_benchmark.sh
#   sudo ./run_benchmark.sh <video_file> [--no-display]
#
# Restore everything afterward:
#   sudo ./run_benchmark.sh --restore
# =============================================================================

set -uo pipefail
# NOTE: deliberately omitting -e so that a segfault in the sobel binary
# (non-zero exit) does not short-circuit the restore block at the bottom.

# ---------- colour helpers ---------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERR]${NC}   $*"; }

# ---------- must run as root -------------------------------------------------
if [[ $EUID -ne 0 ]]; then
    error "Please run as root: sudo $0 $*"
    exit 1
fi

# ---------- resolve absolute path of this script so --restore works ----------
# $0 can be a relative path that breaks after a directory change; resolve once
# at startup so the self-call at the end is always valid.
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

# ---------- binary and restore-state paths -----------------------------------
SOBEL_BIN="$(dirname "$SELF")/sobel"
STATE_DIR="/tmp/sobel_benchmark_state"
mkdir -p "$STATE_DIR"

# =============================================================================
# RESTORE MODE — undo everything this script changed
# =============================================================================
if [[ "${1:-}" == "--restore" ]]; then
    info "Restoring system to pre-benchmark state..."

    # re-enable services
    SERVICES=(
        bluetooth hciuart
        avahi-daemon
        triggerhappy
        cups cups-browsed
        ModemManager
        wpa_supplicant
        NetworkManager
        systemd-timesyncd
    )
    for svc in "${SERVICES[@]}"; do
        if systemctl list-unit-files --quiet "$svc.service" &>/dev/null; then
            systemctl start "$svc" 2>/dev/null || true
            info "Started $svc"
        fi
    done

    # restore CPU governor
    if [[ -f "$STATE_DIR/governor" ]]; then
        OLD_GOV=$(cat "$STATE_DIR/governor")
        for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo "$OLD_GOV" > "$f"
        done
        info "CPU governor restored to: $OLD_GOV"
    fi

    # restore perf_event_paranoid
    if [[ -f "$STATE_DIR/perf_paranoid" ]]; then
        cat "$STATE_DIR/perf_paranoid" > /proc/sys/kernel/perf_event_paranoid
        info "perf_event_paranoid restored"
    fi

    # restore IRQ affinity
    if [[ -d "$STATE_DIR/irq_affinity" ]]; then
        for f in "$STATE_DIR"/irq_affinity/*; do
            irq=$(basename "$f")
            [[ -f "/proc/irq/$irq/smp_affinity" ]] && \
                cat "$f" > "/proc/irq/$irq/smp_affinity" 2>/dev/null || true
        done
        info "IRQ affinities restored"
    fi

    # re-enable Wi-Fi and Bluetooth via rfkill
    rfkill unblock all 2>/dev/null || true
    info "rfkill unblocked"

    # restore NMI watchdog
    if [[ -f "$STATE_DIR/nmi_watchdog" ]]; then
        cat "$STATE_DIR/nmi_watchdog" > /proc/sys/kernel/nmi_watchdog
        info "NMI watchdog restored"
    fi

    rm -rf "$STATE_DIR"
    info "Done. System restored."
    exit 0
fi

# =============================================================================
# BENCHMARK MODE
# =============================================================================

# ---------- argument check ---------------------------------------------------
if [[ $# -lt 1 ]]; then
    error "Usage: sudo $0 <video_file> [--no-display]"
    error "       sudo $0 --restore"
    exit 1
fi

VIDEO_FILE="$1"
EXTRA_ARGS="${2:-}"   # e.g. --no-display

if [[ ! -f "$VIDEO_FILE" ]]; then
    error "Video file not found: $VIDEO_FILE"
    exit 1
fi

if [[ ! -x "$SOBEL_BIN" ]]; then
    error "sobel binary not found or not executable: $SOBEL_BIN"
    error "Run 'make' first."
    exit 1
fi

echo ""
echo "==========================================================="
echo "   Raspberry Pi 5 — Sobel Benchmark Launcher"
echo "==========================================================="
echo ""

# =============================================================================
# 1. CPU FREQUENCY GOVERNOR
# =============================================================================
info "Saving current CPU governor..."
CURRENT_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
echo "$CURRENT_GOV" > "$STATE_DIR/governor"

info "Setting all cores to 'performance' governor..."
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$f"
done

# Pin to maximum frequency (Pi 5 stock = 2400000 kHz)
MAX_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 2400000)
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do
    echo "$MAX_FREQ" > "$f" 2>/dev/null || true
done
info "All cores pinned to ${MAX_FREQ} kHz"

# =============================================================================
# 2. DISABLE NON-ESSENTIAL SERVICES
# =============================================================================
info "Stopping non-essential services..."

SERVICES=(
    bluetooth          # Bluetooth stack
    hciuart            # Bluetooth UART
    avahi-daemon       # mDNS/zeroconf — fires periodic network packets
    triggerhappy       # GPIO button daemon — unnecessary interrupts
    cups               # print spooler
    cups-browsed       # print discovery
    ModemManager       # modem management — interrupts serial IRQs
    wpa_supplicant     # Wi-Fi — even idle it polls and fires IRQs
    NetworkManager     # network management daemon
    systemd-timesyncd  # NTP sync — fires timer interrupts periodically
)

for svc in "${SERVICES[@]}"; do
    if systemctl is-active --quiet "$svc" 2>/dev/null; then
        systemctl stop "$svc" 2>/dev/null && info "  Stopped: $svc" || warn "  Could not stop: $svc"
    else
        info "  Already stopped: $svc"
    fi
done

# =============================================================================
# 3. BLOCK Wi-Fi AND BLUETOOTH AT THE RF LAYER
# =============================================================================
info "Blocking Wi-Fi and Bluetooth via rfkill..."
rfkill block wifi      2>/dev/null && info "  Wi-Fi blocked"      || warn "  rfkill wifi failed"
rfkill block bluetooth 2>/dev/null && info "  Bluetooth blocked"  || warn "  rfkill bluetooth failed"

# =============================================================================
# 4. IRQ AFFINITY — push all IRQs off the worker cores (0-3)
#    onto core 3 only, so cores 0-2 get fewer interruptions.
#    If you set NUM_THREADS=3 in the header, move IRQs to core 3 exclusively.
# =============================================================================
info "Saving and remapping IRQ affinities to core 3..."
mkdir -p "$STATE_DIR/irq_affinity"

for irq_dir in /proc/irq/*/; do
    irq=$(basename "$irq_dir")
    aff_file="$irq_dir/smp_affinity"
    [[ ! -f "$aff_file" ]] && continue
    # save original
    cat "$aff_file" > "$STATE_DIR/irq_affinity/$irq" 2>/dev/null || true
    # move to core 3 only (bitmask = 0x8)
    echo 8 > "$aff_file" 2>/dev/null || true
done
info "  IRQs redirected to core 3 (bitmask 0x8)"

# =============================================================================
# 5. DISABLE NMI WATCHDOG
#    The NMI watchdog fires periodic non-maskable interrupts to detect
#    kernel hangs. Disabling it eliminates those interruptions entirely.
#    Not present on all kernels/boards (Pi 5 often omits it) — skip if so.
# =============================================================================
if [[ -f /proc/sys/kernel/nmi_watchdog ]]; then
    info "Disabling NMI watchdog..."
    cat /proc/sys/kernel/nmi_watchdog > "$STATE_DIR/nmi_watchdog"
    echo 0 > /proc/sys/kernel/nmi_watchdog
else
    warn "NMI watchdog not present on this kernel — skipping"
fi

# =============================================================================
# 6. PERF EVENT PARANOID — allow PAPI hardware counters without root
# =============================================================================
info "Setting perf_event_paranoid = 1 (PAPI needs this)..."
cat /proc/sys/kernel/perf_event_paranoid > "$STATE_DIR/perf_paranoid"
echo 1 > /proc/sys/kernel/perf_event_paranoid

# =============================================================================
# 7. DROP FILESYSTEM CACHES
#    Forces the video decode to actually read from storage, giving a
#    realistic cold-cache measurement rather than a warmed-up one.
#    Comment this out if you want warm-cache numbers instead.
# =============================================================================
info "Dropping filesystem caches (cold-cache benchmark)..."
sync
echo 3 > /proc/sys/vm/drop_caches

# =============================================================================
# 8. PRINT FINAL SYSTEM STATE BEFORE LAUNCH
# =============================================================================
echo ""
echo "-----------------------------------------------------------"
echo "  System state at launch:"
printf "  CPU governor:      %s\n" "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
printf "  CPU freq (core 0): %s kHz\n" "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo unknown)"
printf "  NMI watchdog:      %s\n" "$([[ -f /proc/sys/kernel/nmi_watchdog ]] && cat /proc/sys/kernel/nmi_watchdog || echo 'not present')"
printf "  perf_event_paranoid: %s\n" "$(cat /proc/sys/kernel/perf_event_paranoid)"
printf "  Video file:        %s\n" "$VIDEO_FILE"
printf "  Display:           %s\n" "$([[ "$EXTRA_ARGS" == "--no-display" ]] && echo no || echo yes)"
echo "-----------------------------------------------------------"
echo ""

# =============================================================================
# 9. LAUNCH WITH REAL-TIME PRIORITY
#
#    chrt -f 99   — SCHED_FIFO at priority 99 (highest possible in Linux).
#                   The scheduler will not preempt this process for any
#                   normal or nice'd task. Only IRQs and kernel threads
#                   can interrupt it.
#
#    taskset -c 0-3 — Explicitly allow all four A76 cores. Combined with
#                     the in-code pin_thread_to_core() calls this is
#                     redundant but makes intent clear and prevents the
#                     OS from migrating the main thread away from core 0.
# =============================================================================
info "Launching sobel at SCHED_FIFO priority 99..."
echo ""

chrt -f 99 taskset -c 0-3 "$SOBEL_BIN" "$VIDEO_FILE" ${EXTRA_ARGS:+"$EXTRA_ARGS"}
EXIT_CODE=$?

# =============================================================================
# 10. RESTORE AUTOMATICALLY AFTER RUN
# =============================================================================
echo ""
if [[ $EXIT_CODE -ne 0 ]]; then
    warn "Binary exited with code $EXIT_CODE (segfault = 139, check the C++ binary not this script)"
fi
info "Restoring system..."
"$SELF" --restore

exit $EXIT_CODE
