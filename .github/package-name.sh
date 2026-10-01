#!/usr/bin/env bash
# Names a CI build: its package and the executables in it, the same way in desktop.yml and
# release.yml.
#
#   <product>-<version>-<os>-<arch>-<backend>
#   jevons-desktop-0.1.0-dev.57.g065def7-windows-x86_64-hip-rocm7.2.zip
#     jevons-desktop-0.1.0-dev.57.g065def7-windows-x86_64-hip-rocm7.2.exe
#   jevons-rs-0.1.88-linux-x86_64-hip-rocm7.2.tar.gz
#
# The fields run from the most to the least significant, so a folder of downloads groups by
# product, then version, then platform and GPU backend. The version is a release's (0.1.<run>,
# its tag without the v) or a pre-release of the workspace version: the branch (pr<number> for
# a pull request), the run number, which orders a branch's builds, and the commit, with git
# describe's g so it never reads as a number. The backend is the GPU runtime compiled in and
# the driver release it needs, since a build only runs where that driver is installed. Names
# use only [a-z0-9._-]: no spaces or shell characters, and no + (SemVer build metadata),
# which GitHub may rename in release asset names.
#
# usage: bash .github/package-name.sh PRODUCT TARGET BACKEND [RELEASE]
#
# Prints KEY=value lines for $GITHUB_ENV: PACKAGE (the name), EXE_SUFFIX (.exe on Windows),
# ARCHIVE (the package file) and REQUIRES (what the build needs to run). GITHUB_REF_NAME,
# GITHUB_EVENT_NAME, GITHUB_RUN_NUMBER and GITHUB_SHA describe the build; git stands in for
# them in a local run.
set -euo pipefail

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "usage: $0 PRODUCT TARGET BACKEND [RELEASE]" >&2
    exit 2
fi
product=$1 target=$2 backend=$3 release=${4:-}
fail() {
    echo "package-name.sh: $*" >&2
    exit 1
}

if [ -n "$release" ]; then
    [[ $release =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "release $release is not X.Y.Z"
    version=$release
else
    root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    base=$(sed -n '/^\[workspace\.package\]/,/^\[/s/^version = "\(.*\)"$/\1/p' "$root/Cargo.toml")
    [ -n "$base" ] || fail "no [workspace.package] version in Cargo.toml"
    ref=${GITHUB_REF_NAME:-$(git -C "$root" branch --show-current)}
    sha=${GITHUB_SHA:-$(git -C "$root" rev-parse HEAD)}
    case ${GITHUB_EVENT_NAME:-push} in
        # The ref of a pull request is <number>/merge.
        pull_request*) channel=pr${ref%%/*} ;;
        *) channel=$(printf '%s' "$ref" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-') ;;
    esac
    version=$base-$channel.${GITHUB_RUN_NUMBER:-0}.g${sha::7}
fi

arch=${target%%-*}
case $target in
    *-windows-msvc) os=windows exe=.exe archive=zip ;;
    *-linux-gnu) os=linux exe='' archive=tar.gz ;;
    *) fail "no OS name for $target" ;;
esac

case $backend in
    hip)
        # The HIP binding layout compiled in (hipconfig.rs) is the ROCm release the binaries
        # load at run time: 7.2.53211 is ROCm 7.2.
        hip=${JEVONS_HIP_VERSION:-}
        [[ $hip =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "JEVONS_HIP_VERSION '$hip' is not X.Y.Z"
        rocm=${hip%.*}
        tag=hip-rocm$rocm
        requires="an AMD RDNA3-class GPU and ROCm/HIP $rocm"
        if [ "$os" = windows ]; then
            requires+=": the AMD driver and the AMD HIP SDK"
        fi
        ;;
    # A CUDA build names its toolkit release from the version it pins (cuda-12.8, needing an
    # NVIDIA driver for CUDA 12.8 or newer); a WGPU build is plain wgpu (any Vulkan, Direct3D 12
    # or Metal driver).
    *) fail "unknown backend $backend" ;;
esac

package=$product-$version-$os-$arch-$tag
echo "PACKAGE=$package"
echo "EXE_SUFFIX=$exe"
echo "ARCHIVE=$package.$archive"
echo "REQUIRES=$requires"
