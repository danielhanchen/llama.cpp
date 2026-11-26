# mac-metal-check

Metal check of the unslothai/llama.cpp prebuilt (upstream tag + `scripts/unsloth/pr-set.json` pins) and of the
few-token MUL_MAT_ID kernel (`084c4350e1`, branch `metal-mul-mv-id-ext`).

On an Apple Silicon Mac (Xcode Command Line Tools, cmake, python3, ~10 GB free):

    curl -fsSLO https://raw.githubusercontent.com/danielhanchen/llama.cpp/mac-metal-check/scripts/mac-metal-check/mac_metal_check.sh
    caffeinate -i bash mac_metal_check.sh

Everything goes to `./mac_metal_check_work`; the result is `mac_metal_check_<date>.tar.gz` plus the summary printed
at the end. `QUICK=1` skips the full FLASH_ATTN_EXT sweep, `MVID=0` checks only the shipped tree.

`.github/workflows/mac-metal-check.yml` runs the same on macos-15 with the paravirtual-GPU force hooks on
(`MAC_METAL_CI_HOOKS=1`): correctness only, no meaningful timing.
