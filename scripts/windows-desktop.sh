#!/usr/bin/env bash
# Builds jevons-desktop in release mode natively on Windows from WSL, installs it and starts it.
# The embedded runtime links a prebuilt LLVM for the host, so it cannot cross-compile.
#
#   scripts/windows-desktop.sh            build, install and start the app
#   scripts/windows-desktop.sh --no-run   build and install, leaving the app stopped
#   scripts/windows-desktop.sh --build    only build
#
# The build runs in a Windows clone of this repository, JEVONS_WINDOWS_CLONE (by default
# %USERPROFILE%\src\jevons-rs), which it first makes match this working tree, uncommitted and
# untracked files included: other changes in the clone are discarded (its target/ stays). Only
# files whose contents differ are written, so cargo rebuilds only what changed. The app goes
# into JEVONS_WINDOWS_APP (by default %USERPROFILE%\jevons), where it is started, so a running
# app never locks the build's own executable.
#
# Windows needs the MSVC Build Tools, rustup, the AMD HIP SDK and Python 3 from python.org
# (stylo generates code with it; the Microsoft Store alias does not work).
set -euo pipefail

mode=${1:-run}
case "$mode" in
run | --no-run | --build) ;;
*)
    echo "usage: $0 [--no-run | --build]" >&2
    exit 2
    ;;
esac

repo=$(git rev-parse --show-toplevel)
system=/mnt/c/Windows/System32
# cmd.exe refuses a UNC working directory, such as a WSL path.
cd /mnt/c
profile=$(wslpath "$("$system/cmd.exe" /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')")
clone=${JEVONS_WINDOWS_CLONE:-$profile/src/jevons-rs}
app=${JEVONS_WINDOWS_APP:-$profile/jevons}

# A build still running from an earlier call would fight over target/.
while "$system/tasklist.exe" /FI "IMAGENAME eq cargo.exe" 2>/dev/null | grep -q cargo.exe; do
    echo "waiting for a cargo build already running on Windows"
    sleep 10
done

# Make the clone match this working tree.
if [ ! -d "$clone/.git" ]; then
    mkdir -p "$(dirname "$clone")"
    git clone -q "$repo" "$clone"
fi
git -C "$clone" fetch -q "$repo" HEAD
if [ "$(git -C "$clone" rev-parse HEAD)" != "$(git -C "$clone" rev-parse FETCH_HEAD)" ]; then
    git -C "$clone" checkout -q -f --detach FETCH_HEAD
fi
# Every file of this working tree, written only where its contents differ. Without -t the
# files written are dated now, so cargo sees them as changed.
git -C "$repo" ls-files -z --cached --others --exclude-standard |
    rsync -rl --checksum --ignore-missing-args --from0 --files-from=- "$repo/" "$clone/"
# Files deleted here, and files the clone has that this working tree does not.
git -C "$repo" ls-files -z --deleted | (cd "$clone" && xargs -0 -r rm -f --)
comm -z -23 \
    <(git -C "$clone" ls-files -z --others --exclude-standard | sort -z) \
    <(git -C "$repo" ls-files -z --others --exclude-standard | sort -z) |
    (cd "$clone" && xargs -0 -r rm -f --)
changes=$(git -C "$clone" status --porcelain | wc -l)
echo "building $(git -C "$repo" log --oneline -1) with $changes uncommitted changes, in $(wslpath -w "$clone")"

# The same environment as the Desktop workflow: pinned HIP bindings, and python.org's Python.
hip=$(sed -n 's/^ *JEVONS_HIP_VERSION: *"\(.*\)"/\1/p' "$repo/.github/workflows/desktop.yml" | head -1)
mkdir -p "$clone/target"
script=$clone/target/jevons-desktop-build.cmd
log=$clone/target/jevons-desktop-build.log
printf '%s\r\n' \
    '@echo off' \
    'setlocal' \
    "cd /d \"$(wslpath -w "$clone")\"" \
    "set JEVONS_HIP_VERSION=$hip" \
    'for /d %%p in ("%LOCALAPPDATA%\Programs\Python\Python3*") do set "PYTHON3=%%p\python.exe"' \
    'set "SHIM=%TEMP%\jevons-bin"' \
    'set "PATH=%SHIM%;%USERPROFILE%\.cargo\bin;%PATH%"' \
    'if not exist "%SHIM%" mkdir "%SHIM%"' \
    'rustc -O .github\hipconfig.rs -o "%SHIM%\hipconfig.exe" || exit /b 1' \
    'cargo build --release --locked -p jevons-desktop' \
    >"$script"
started=$(date +%s)
if ! "$system/cmd.exe" /c "$(wslpath -w "$script")" 2>&1 | tr -d '\r' | tee "$log"; then
    echo "BUILD FAILED after $(($(date +%s) - started)) s; the log is $(wslpath -w "$log")" >&2
    exit 1
fi
echo "built in $(($(date +%s) - started)) s"
[ "$mode" = --build ] && exit 0

"$system/taskkill.exe" /IM jevons-desktop.exe /F >/dev/null 2>&1 || true
mkdir -p "$app"
# The executable stays locked for a moment after the process ends.
for _ in $(seq 1 30); do
    cp "$clone/target/release/jevons-desktop.exe" "$clone/target/release/jevons_desktop.pdb" "$app/" 2>/dev/null && break
    sleep 1
done
if ! cmp -s "$clone/target/release/jevons-desktop.exe" "$app/jevons-desktop.exe"; then
    echo "INSTALL FAILED: $(wslpath -w "$app")\\jevons-desktop.exe is still the old build" >&2
    exit 1
fi
echo "installed into $(wslpath -w "$app")"
[ "$mode" = --no-run ] && exit 0

exe=$(wslpath -w "$app/jevons-desktop.exe")
"$system/WindowsPowerShell/v1.0/powershell.exe" -NoProfile -Command \
    "Start-Process -FilePath '$exe' -WorkingDirectory '$(wslpath -w "$app")'"
echo "started $exe"
