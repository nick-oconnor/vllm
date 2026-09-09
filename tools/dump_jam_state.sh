#!/usr/bin/env bash
# dump_jam_state.sh - dump vLLM hang state to /home/vllm for post-mortem.
#
# Run from inside a jammed vLLM pod as the same UID the engine runs as
# (root in the default vllm-openai image, vllm/2000 in -nonroot). py-spy
# ptrace needs same-UID access; --nonblocking lets it dump a wedged
# target that won't respond to SIGSTOP via /proc/<pid>/{syscall,stack}.
#
# Captures:
#   - timestamp + hostname + uname + pod name
#   - nvidia-smi full + a one-liner of per-GPU util
#   - py-spy dump for every vllm/EngineCore/Worker PID
#   - /proc/<pid>/{cmdline,status,wchan} for each PID
#   - last 50 dmesg Xid/NVRM lines (if reachable; usually requires host)
#
# Output: /home/vllm/jam-<UTC>-<hostshort>/  (created if missing)
#
# Usage:
#   dump-jam-state.sh
#   dump-jam-state.sh --out-dir /var/log/vllm
#
# Exit code: 0 on success even if py-spy fails for individual PIDs
# (we still want the partial dump).

set -u

OUT_DIR="/home/vllm"
while [ $# -gt 0 ]; do
    case "$1" in
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,28p' "$0"
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

TS="$(date -u +%Y%m%dT%H%M%SZ)"
HOST_SHORT="$(hostname -s 2>/dev/null || hostname || echo unknown)"
DEST="${OUT_DIR%/}/jam-${TS}-${HOST_SHORT}"

if ! mkdir -p "$DEST" 2>/dev/null; then
    echo "FATAL: cannot create output dir: $DEST" >&2
    echo "hint: are you running as a UID that can write to ${OUT_DIR%/}/ ?" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# summary.txt — one file with all the small/cheap captures
# ----------------------------------------------------------------------------
SUMMARY="$DEST/summary.txt"
{
    echo "=== timestamp (UTC) ==="
    date -u +%Y-%m-%dT%H:%M:%SZ
    echo
    echo "=== hostname ==="
    hostname 2>/dev/null || echo "(unknown)"
    echo
    echo "=== uname ==="
    uname -a
    echo
    echo "=== whoami / id ==="
    whoami 2>/dev/null || true
    id 2>/dev/null || true
    echo
    echo "=== nvidia-smi ==="
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi 2>&1 || echo "(nvidia-smi failed)"
    else
        echo "(nvidia-smi not in PATH)"
    fi
    echo
    echo "=== nvidia-smi per-GPU util ==="
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader 2>&1 || true
    fi
    echo
    echo "=== pgrep targets (vllm|EngineCore|Worker) ==="
    pgrep -af 'vllm|EngineCore|Worker' 2>&1 || echo "(no matches)"
    echo
    echo "=== dmesg Xid / NVRM (last 50) ==="
    if dmesg >/dev/null 2>&1; then
        dmesg 2>&1 | grep -iE 'xid|nvrm' | tail -50 || echo "(no Xid/NVRM lines)"
    else
        echo "(dmesg unavailable; usually needs host, not container)"
    fi
} > "$SUMMARY" 2>&1

# ----------------------------------------------------------------------------
# py-spy dump + /proc state per PID
# ----------------------------------------------------------------------------
PIDS="$(pgrep -f 'vllm|EngineCore|Worker' | sort -un)"
if [ -z "$PIDS" ]; then
    echo "no matching PIDs" >> "$SUMMARY"
else
    for p in $PIDS; do
        {
            echo "============================================================"
            echo "= pid $p"
            echo "============================================================"
            echo "-- /proc/$p/cmdline --"
            tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null
            echo
            echo "-- /proc/$p/status --"
            cat "/proc/$p/status" 2>/dev/null || echo "(status unavailable)"
            echo
            echo "-- /proc/$p/wchan --"
            cat "/proc/$p/wchan" 2>/dev/null || echo "(wchan unavailable)"
            echo
            echo "-- /proc/$p/stack --"
            cat "/proc/$p/stack" 2>/dev/null | head -40 || echo "(stack unavailable)"
        } > "$DEST/proc-$p.txt" 2>&1

        # py-spy dump. --nonblocking: use /proc fallbacks if SIGSTOP fails.
        PYSPY="$(command -v py-spy || echo py-spy)"
        if "$PYSPY" dump --pid "$p" --nonblocking > "$DEST/py-spy-$p.txt" 2>&1; then
            echo "py-spy dump OK: pid $p -> $DEST/py-spy-$p.txt"
        else
            echo "py-spy dump FAILED: pid $p (see $DEST/py-spy-$p.txt)"
        fi
    done >> "$SUMMARY"
fi

echo
echo "============================================================"
echo "Dump complete: $DEST"
echo "============================================================"
ls -la "$DEST"
