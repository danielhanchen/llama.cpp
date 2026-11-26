#!/usr/bin/env bash
# Metal check for the unslothai/llama.cpp prebuilt on an Apple Silicon Mac.
#
#   bash mac_metal_check.sh            # pins + tiny-model KLD + bench + the few-token MUL_MAT_ID kernel A/B
#   QUICK=1 bash mac_metal_check.sh    # skip the full FLASH_ATTN_EXT sweep (about half the test time)
#   MVID=0 bash mac_metal_check.sh     # only the shipped prebuilt tree (no added kernel commit)
#
# Everything lands in ./mac_metal_check_work (delete it afterwards). At the end it prints a summary and writes
# mac_metal_check_<date>.tar.gz (result JSON + summary + logs) to attach or paste.
# Env: TAG (upstream base tag, default b12722), MVID_COMMIT (<repo url>@<sha40>), MVID_ROUNDS (3), PINSET_REF (unslothai/llama.cpp branch holding pr-set.json, default master), PARTS, WORK, JOBS,
#      MAC_METAL_CI_HOOKS=1 (GitHub paravirtual runners only; never needed on a real Mac),
#      MAC_METAL_CHECK_PY (path or URL of mac_metal_check.py, default: next to this script, else the fork branch).
set -euo pipefail

TAG="${TAG:-b12722}"
MVID="${MVID:-1}"
MVID_COMMIT="${MVID_COMMIT:-https://github.com/danielhanchen/llama.cpp@084c4350e1d65893cc993a8369ad65ad95003e3f}"
WORK="${WORK:-$PWD/mac_metal_check_work}"
PY_URL_DEFAULT="https://raw.githubusercontent.com/danielhanchen/llama.cpp/mac-metal-check/scripts/mac-metal-check/mac_metal_check.py"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEED_GB="${NEED_GB:-10}"

die() { echo "ERROR: $*" >&2; exit 2; }

# ---------------------------------------------------------------- prerequisites
[ "$(uname -s)" = "Darwin" ] || [ "${MAC_METAL_CHECK_ALLOW_NON_MAC:-0}" = "1" ] || die "this needs macOS on Apple Silicon"
if [ "$(uname -s)" = "Darwin" ]; then
  [ "$(uname -m)" = "arm64" ] || die "Apple Silicon (arm64) needed, got $(uname -m) (Rosetta shell? run 'arch -arm64 bash $0')"
  xcode-select -p >/dev/null 2>&1 || die "Xcode Command Line Tools missing: run 'xcode-select --install', then rerun"
  xcrun --show-sdk-path >/dev/null 2>&1 || die "no macOS SDK: reinstall the Command Line Tools ('xcode-select --install')"
fi
command -v git >/dev/null || die "git missing (comes with the Command Line Tools)"
command -v cmake >/dev/null || die "cmake missing: 'brew install cmake' (or 'python3 -m pip install --user cmake' and add it to PATH)"
command -v python3 >/dev/null || die "python3 missing (comes with the Command Line Tools)"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' || die "python3 >= 3.8 needed"
mkdir -p "$WORK"
FREE_GB=$(df -Pk "$WORK" | awk 'NR==2 {print int($4/1048576)}')
[ "$FREE_GB" -ge "$NEED_GB" ] || die "need ~${NEED_GB} GB free under $WORK, have ${FREE_GB} GB"
JOBS="${JOBS:-$( (sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || nproc) | head -1)}"
echo "host: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m), macOS $(sw_vers -productVersion 2>/dev/null || echo n/a), \
cmake $(cmake --version | head -1 | awk '{print $3}'), $(python3 --version), ${FREE_GB} GB free, -j $JOBS"

# ---------------------------------------------------------------- the job script
PY="${MAC_METAL_CHECK_PY:-}"
if [ -z "$PY" ]; then
  for c in "$HERE/mac_metal_check.py" "$HERE/../jobs/mac_metal_check.py"; do
    [ -f "$c" ] && PY="$c" && break
  done
fi
case "${PY:-$PY_URL_DEFAULT}" in
  http*://*)
    curl -fsSL "${PY:-$PY_URL_DEFAULT}" -o "$WORK/mac_metal_check.py" || die "download of mac_metal_check.py failed"
    PY="$WORK/mac_metal_check.py" ;;
esac
[ -f "$PY" ] || die "mac_metal_check.py not found ($PY)"

ARGS=(--tag "$TAG" --work "$WORK" --out "$WORK/mac_metal_check.json" --jobs "$JOBS")
[ "$MVID" = "1" ] && ARGS+=(--add-commit "$MVID_COMMIT")
[ -n "${PINSET_REF:-}" ] && ARGS+=(--pinset-ref "$PINSET_REF")
[ "${QUICK:-0}" = "1" ] && ARGS+=(--quick)
[ "${MAC_METAL_CI_HOOKS:-0}" = "1" ] && ARGS+=(--ci-hooks)
[ -n "${PARTS:-}" ] && ARGS+=(--parts "$PARTS")
[ -n "${MVID_ROUNDS:-}" ] && ARGS+=(--mvid-rounds "$MVID_ROUNDS")

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$WORK/run_$STAMP.log"
echo "running: python3 $PY ${ARGS[*]}"
echo "log: $LOG  (takes ~30-60 min; the Mac must stay awake: 'caffeinate -i bash $0' keeps it so)"
set +e
python3 -u "$PY" "${ARGS[@]}" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

# ---------------------------------------------------------------- bundle
TAR="$PWD/mac_metal_check_$STAMP.tar.gz"
( cd "$WORK" && find . -maxdepth 1 \( -name '*.log' -o -name '*.json' -o -name 'summary.md' \) -print0 \
    | xargs -0 tar -czf "$TAR" ) || echo "WARN: tarball failed"
echo
case $RC in
  0) echo "RESULT: PASS" ;;
  1) echo "RESULT: FAIL (a gate failed, see the table above)" ;;
  *) echo "RESULT: SETUP FAIL rc=$RC (clone / merge / build, see $LOG)" ;;
esac
echo "json:    $WORK/mac_metal_check.json"
echo "summary: $WORK/summary.md"
echo "bundle:  $TAR  (attach this)"
echo "cleanup: rm -rf '$WORK'"
exit "$RC"
