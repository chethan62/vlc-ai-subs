#!/usr/bin/env bash
# Branch matrix for install.sh / setup.sh.
#
# Runs the REAL installer scripts against a sandbox: a private HOME, a curated
# PATH (so nvidia-smi can be present or absent, simulating an NVIDIA vs an
# AMD/Intel box), and stubs for the heavyweight steps (whisper.cpp build, model
# downloads, uv) so no sudo, no 600 MB download and no GPU are needed. Nothing
# outside $SB is ever touched, and `sudo` is stubbed to fail.
#
# The point: the AMD/Intel engine-selection path and the "did the optional step
# actually happen?" reporting are otherwise unexercised — the test box has
# nvidia-smi and a working toolchain, so those branches never run for real.
#
# Usage: bash tests/install_branches.sh
set -u

SB=$(mktemp -d "${TMPDIR:-/tmp}/aisubs-instest.XXXXXX")
trap 'rm -rf "$SB"' EXIT
REPO=$(cd "$(dirname "$0")/.." && pwd)
CALLS="$SB/calls.log"
PASS=0; FAIL=0

say()   { printf '\n\033[36m══ %s\033[0m\n' "$*"; }
pass()  { printf '  \033[32mok\033[0m   %s\n' "$*"; PASS=$((PASS+1)); }
bad()   { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }

# ── curated PATH: the tools install.sh needs, minus nvidia-smi/sudo ──────────
TOOLS="bash sh env python3 python ffmpeg ffprobe git cmake cc gcc make glslc curl tar \
unzip sed awk grep cut head tail sort uniq wc tr find ls mkdir cp mv rm ln chmod dirname \
basename date sleep printf tee cat touch xargs which du df id stat readlink realpath getconf"
mkdir -p "$SB/bin"
for t in $TOOLS; do p=$(command -v "$t" 2>/dev/null) && ln -sf "$p" "$SB/bin/$t"; done
printf '#!/bin/sh\necho "[stub uv] $*" >> %s\nexit 0\n' "$CALLS" > "$SB/bin/uv"
printf '#!/bin/sh\necho "VLC media player 3.0.23 Vetinari"\n'      > "$SB/bin/vlc"
# sudo exists but always refuses: the install must fall back to the user dir
printf '#!/bin/sh\necho "[stub sudo] refused: $*" >> %s\nexit 1\n' "$CALLS" > "$SB/bin/sudo"
printf '#!/bin/sh\necho "NVIDIA-SMI 580.00"; echo "NVIDIA GeForce GTX 1650"\n' > "$SB/nvidia-smi"
chmod +x "$SB/bin/uv" "$SB/bin/vlc" "$SB/bin/sudo" "$SB/nvidia-smi"

fresh_repo() {
    rm -rf "$SB/repo" && cp -r "$REPO" "$SB/repo" && rm -rf "$SB/repo/.git"
    for s in install-whisper-cpp.sh install-parakeet-model.sh install-nllb-model.sh install-m2m-model.sh; do
        printf '#!/usr/bin/env bash\necho "[stub %s] $*" >> %s\nexit ${STUB_FAIL:-0}\n' "$s" "$CALLS" > "$SB/repo/$s"
        chmod +x "$SB/repo/$s"
    done
}

# install [VAR=value ...] — runs install.sh in the sandbox (HOME already prepared)
install_sh() {
    if [ $# -gt 0 ]; then
        ( cd "$SB/repo" && export "$@" && HOME="$SB/home" PATH="$SB/bin" bash "$SB/repo/install.sh" ) > "$SB/out.txt" 2>&1
    else
        ( cd "$SB/repo" && HOME="$SB/home" PATH="$SB/bin" bash "$SB/repo/install.sh" ) > "$SB/out.txt" 2>&1
    fi
    echo $?
}
prepare_home() { rm -rf "$SB/home" && mkdir -p "$SB/home"; }
run() { prepare_home; install_sh "$@"; }
have_extension() { [ -f "$SB/home/.local/share/vlc/lua/extensions/aisubs.lua" ]; }
logs_contain()   { grep -qF "$1" "$SB/out.txt"; }
stub_calls()     { grep -c 'stub install-whisper-cpp' "$CALLS" 2>/dev/null || true; }

# ── 1. AMD/Intel box: no nvidia-smi → Vulkan engine installed automatically ──
say "1. AMD/Intel box (no nvidia-smi)"
fresh_repo; : > "$CALLS"; rm -f "$SB/bin/nvidia-smi"; rc=$(run)
[ "$rc" = 0 ] && pass "install.sh exits 0" || bad "exit $rc"
[ "$(stub_calls)" = 1 ] && pass "whisper.cpp (Vulkan) installed automatically" || bad "whisper.cpp not installed ($(stub_calls) calls)"
have_extension && pass "extension installed into the user dir" || bad "extension missing"
logs_contain "Install complete." && pass "reports completion" || bad "no completion line"

# ── 2. NVIDIA box: whisper.cpp is opt-in, not automatic ──────────────────────
say "2. NVIDIA box (nvidia-smi present)"
fresh_repo; : > "$CALLS"; cp "$SB/nvidia-smi" "$SB/bin/nvidia-smi"; rc=$(run)
[ "$(stub_calls)" = 0 ] && pass "whisper.cpp NOT installed (CUDA box)" || bad "installed whisper.cpp on an NVIDIA box"
logs_contain "NVIDIA GPU detected" && pass "says why it skipped" || bad "no NVIDIA skip message"

# ── 3. NVIDIA box that wants the Vulkan engine anyway ────────────────────────
say "3. NVIDIA box + VSCL_AISUBS_WHISPERCPP=1"
fresh_repo; : > "$CALLS"; rc=$(run VSCL_AISUBS_WHISPERCPP=1)
[ "$(stub_calls)" = 1 ] && pass "installed on request" || bad "opt-in ignored"

# ── 4. Opting out wins on any box ────────────────────────────────────────────
say "4. AMD box + VSCL_AISUBS_SKIP_WHISPERCPP=1"
fresh_repo; : > "$CALLS"; rm -f "$SB/bin/nvidia-smi"; rc=$(run VSCL_AISUBS_SKIP_WHISPERCPP=1)
[ "$(stub_calls)" = 0 ] && pass "skipped on request" || bad "skip ignored"
logs_contain "Skipping whisper.cpp" && pass "says it skipped" || bad "silent skip"

# ── 5. Already installed: no rebuild ─────────────────────────────────────────
say "5. AMD box where whisper.cpp is already installed"
fresh_repo; : > "$CALLS"; rm -f "$SB/bin/nvidia-smi"
prepare_home
mkdir -p "$SB/home/.local/share/whisper-cpp"
printf '#!/bin/sh\necho fake\n' > "$SB/home/.local/share/whisper-cpp/whisper-cli"
chmod +x "$SB/home/.local/share/whisper-cpp/whisper-cli"
rc=$(install_sh)
[ "$(stub_calls)" = 0 ] && pass "did not rebuild" || bad "rebuilt an existing install"

# ── 6. The Vulkan build fails: an optional engine must not kill the install ──
say "6. AMD box where the Vulkan build FAILS"
fresh_repo; : > "$CALLS"; rm -f "$SB/bin/nvidia-smi"
printf '#!/usr/bin/env bash\necho "[stub install-whisper-cpp.sh] FAILING" >> %s\nexit 1\n' "$CALLS" > "$SB/repo/install-whisper-cpp.sh"
rc=$(run)
[ "$rc" = 0 ] && pass "install still exits 0" || bad "exit $rc — an optional engine aborted the install"
logs_contain "whisper.cpp build failed" && pass "reports the failure" || bad "failure not reported"
have_extension && pass "extension still installed" || bad "extension missing"

# ── 7. sudo denied (the README's `curl … | bash`): the USER dir must still work
# Regression: install_to used `sudo cp … && ok … && INSTALLED=1`, which aborted
# setup.sh under `set -e` when sudo failed — before the user-level fallback ran.
say "7. sudo refused (non-interactive install)"
fresh_repo; : > "$CALLS"; rm -f "$SB/bin/nvidia-smi"; rm -rf "$SB/home/.local/share/vlc"; rc=$(run)
[ "$rc" = 0 ] && pass "install exits 0" || bad "exit $rc"
have_extension && pass "extension installed into the user dir despite no sudo" || bad "nothing installed — sudo failure aborted the fallback"
logs_contain "Plugin installed" && pass "reports the extension honestly" || bad "no extension status"

# ── 8. A hard setup.sh failure must NOT be reported as success ───────────────
say "8. extension install fails hard (aisubs.lua missing)"
fresh_repo; : > "$CALLS"; rm -f "$SB/bin/nvidia-smi" "$SB/repo/aisubs.lua"; rc=$(run)
logs_contain "Plugin installed" && bad "claimed success while nothing was installed" \
                                 || pass "does not claim the extension was installed"
logs_contain "NOT installed" && pass "says the extension is missing" || bad "no warning about the missing extension"

printf '\n\033[36m══ %d passed, %d failed\033[0m\n' "$PASS" "$FAIL"
[ "$FAIL" = 0 ]
