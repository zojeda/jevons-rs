#!/usr/bin/env bash
# Runs cargo natively on Windows from WSL, in the clone scripts/windows-desktop.sh builds in,
# after making the clone match this working tree as that script does. GPU tests run there
# where WSL has no ROCm.
#
#   scripts/windows-cargo.sh NAME [VAR=VALUE ...] -- cargo args... [::: cargo args...]
#
# Each ":::" starts another cargo command; they run one after the other. The output goes to
# target\windows-cargo\NAME.log in the clone, and its end is printed here (TAIL lines, 60 by default).
# The Windows process is started detached and is never killed: a process that ends in the
# middle of a call to the GPU has left HIP failing for every process until a reboot. Run one
# at a time.
set -euo pipefail
name=$1; shift
envs=()
while [ "$1" != "--" ]; do envs+=("$1"); shift; done
shift
repo=$(git rev-parse --show-toplevel)
system=/mnt/c/Windows/System32
cd /mnt/c
profile=$(wslpath "$("$system/cmd.exe" /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')")
clone=${JEVONS_WINDOWS_CLONE:-$profile/src/jevons-rs}
while "$system/tasklist.exe" /FI "IMAGENAME eq cargo.exe" 2>/dev/null | grep -q cargo.exe; do
    echo "waiting for a cargo build already running on Windows"; sleep 10
done
git -C "$clone" fetch -q "$repo" HEAD
if [ "$(git -C "$clone" rev-parse HEAD)" != "$(git -C "$clone" rev-parse FETCH_HEAD)" ]; then
    git -C "$clone" checkout -q -f --detach FETCH_HEAD
fi
git -C "$repo" ls-files -z --cached --others --exclude-standard |
    rsync -rl --checksum --ignore-missing-args --from0 --files-from=- "$repo/" "$clone/"
git -C "$repo" ls-files -z --deleted | (cd "$clone" && xargs -0 -r rm -f --)
comm -z -23 \
    <(git -C "$clone" ls-files -z --others --exclude-standard | sort -z) \
    <(git -C "$repo" ls-files -z --others --exclude-standard | sort -z) |
    (cd "$clone" && xargs -0 -r rm -f --)
hip=$(sed -n 's/^ *JEVONS_HIP_VERSION: *"\(.*\)"/\1/p' "$repo/.github/workflows/desktop.yml" | head -1)
mkdir -p "$clone/target/windows-cargo"
script=$clone/target/windows-cargo/$name.cmd
log=$clone/target/windows-cargo/$name.log
{
    printf '%s\r\n' '@echo off' 'setlocal' "cd /d \"$(wslpath -w "$clone")\"" \
        "set JEVONS_HIP_VERSION=$hip" \
        'for /d %%p in ("%LOCALAPPDATA%\Programs\Python\Python3*") do set "PYTHON3=%%p\python.exe"' \
        'set "SHIM=%TEMP%\jevons-bin"' \
        'set "PATH=%SHIM%;%USERPROFILE%\.cargo\bin;%PATH%"' \
        'if not exist "%SHIM%" mkdir "%SHIM%"' \
        'if not exist "%SHIM%\hipconfig.exe" rustc -O .github\hipconfig.rs -o "%SHIM%\hipconfig.exe"'
    for e in "${envs[@]}"; do printf 'set "%s"\r\n' "$e"; done
    # Several cargo commands, separated by ":::", run one after the other into the one log.
    wlog=$(wslpath -w "$log")
    printf 'type nul > "%s"\r\n' "$wlog"
    cmd=()
    for arg in "$@" ":::"; do
        if [ "$arg" = ":::" ]; then
            printf 'echo === cargo %s >> "%s"\r\n' "${cmd[*]}" "$wlog"
            printf 'cargo %s >> "%s" 2>&1\r\n' "${cmd[*]}" "$wlog"
            printf 'echo --- status %%ERRORLEVEL%% >> "%s"\r\n' "$wlog"
            cmd=()
        else
            cmd+=("$arg")
        fi
    done
    printf 'echo exit done >> "%s"\r\n' "$wlog"
} >"$script"
started=$(date +%s)
rm -f "$log"
# Detached from this shell: whatever happens here, the Windows process runs to its own end.
"$system/WindowsPowerShell/v1.0/powershell.exe" -NoProfile -Command \
    "Start-Process -FilePath 'cmd.exe' -ArgumentList '/c','\"$(wslpath -w "$script")\"' -WindowStyle Hidden" >/dev/null 2>&1
until [ -f "$log" ] && grep -q "^exit " "$log"; do sleep 5; done
echo "ran in $(($(date +%s) - started)) s; log: $log"
tr -d '\r' <"$log" | grep -v "^\s*Compiling\|^\s*Downloaded\|^\s*Downloading" | tail -${TAIL:-60}
