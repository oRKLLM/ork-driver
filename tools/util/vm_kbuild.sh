#!/bin/sh
# vm_kbuild.sh — build the board kernel on the .239 Colima VM (16 cores, ~7 min vs ~45 on the SBC),
# install the Image to the board, and SHUT THE VM DOWN again.
#
#   tools/util/vm_kbuild.sh [--keep]                 --keep leaves the VM RUNNING for the next build
#   tools/util/vm_kbuild.sh --from-branch <ref>      build a REVIEWABLE ref instead of the live board tree
#
# SOURCE OF TRUTH, and why --from-branch exists (#5). By default this builds the BOARD'S OWN working tree
# (~/kbuild/linux-rockchip). That makes the board authoritative, which is what the kernel work wants -- but
# it also means any edit ANYONE left in that tree is silently included in the next kernel ANYBODY builds,
# and gets attributed to whatever change they thought they were testing. It has cost attribution: kernel
# #59, built to validate a NULL-domain guard, also carried another agent's in-progress GEM-destroy reclaim
# (#patch69/70), so "make test passes on #59" was evidence about a composite tree, not about the change.
# `uname -v` is a number, not a manifest. --from-branch pins the inputs to a git ref instead; either way
# the run now PRINTS its provenance and patch-hunk inventory before building.
#
# Iterating? Use --keep. colima start/stop costs ~60-90s per cycle, which dwarfs an incremental
# rknpu-only rebuild (~1-2 min on 16 cores vs ~2-4 min on the SBC's 8). Shut it down with
# `tools/util/vm_kbuild.sh --stop` (or `colima stop`) when the kernel work is done.
#
# Any .py args are copied into the container and run against /root/linux-rockchip before building.
# The container `kbuild` holds the tree at the base commit with kernel changes 01-03 committed; the
# newer changes are carried by syncing the four rknpu files + .config from the board (source of truth).
set -e
KEEP=0
FROM_BRANCH=""
KREPO="${ORK_KERNEL_REPO:-$HOME/Dev/rk3588-kernel}"
while :; do
    case "$1" in
        --keep) KEEP=1; shift ;;
        --from-branch) FROM_BRANCH="$2"; shift 2 ;;
        *) break ;;
    esac
done
if [ "$1" = "--stop" ]; then
    ssh 10.3.0.239 "export PATH=\$PATH:/opt/homebrew/bin; docker stop kbuild >/dev/null 2>&1; colima stop" 2>&1 | tail -1
    echo "VM stopped"; exit 0
fi
VM=10.3.0.239
BOARD=board
P="export PATH=\$PATH:/opt/homebrew/bin"

echo "== starting colima (if not already up) =="
ssh $VM "$P; colima status >/dev/null 2>&1 || colima start" 2>&1 | tail -2
ssh $VM "$P; docker start kbuild" >/dev/null

if [ -n "$FROM_BRANCH" ]; then
    echo "== PROVENANCE: building git ref '$FROM_BRANCH' from $KREPO (NOT the live board tree) =="
    [ -d "$KREPO/.git" ] || { echo "vm_kbuild: no kernel checkout at $KREPO (set ORK_KERNEL_REPO)" >&2; exit 1; }
    git -C "$KREPO" rev-parse --verify -q "$FROM_BRANCH" >/dev/null || { echo "vm_kbuild: no such ref '$FROM_BRANCH' in $KREPO" >&2; exit 1; }
    echo "   ref: $FROM_BRANCH = $(git -C "$KREPO" rev-parse --short "$FROM_BRANCH")  ($(git -C "$KREPO" log -1 --format=%s "$FROM_BRANCH" | cut -c1-72))"
    echo "   hunks: $(git -C "$KREPO" grep -ohE '#patch[0-9A-Za-z]+' "$FROM_BRANCH" -- drivers/rknpu drivers/iommu/rockchip-iommu.c 2>/dev/null | sort -u | tr '\n' ' ')"
else
    echo "== PROVENANCE: building the LIVE BOARD TREE (~/kbuild/linux-rockchip) =="
    echo "   This includes ANY uncommitted edit anyone left there. To pin the inputs to a reviewable"
    echo "   ref instead, re-run with --from-branch <ref>. See ork-driver#5."
    echo "   hunks: $(ssh $BOARD "cd ~/kbuild/linux-rockchip && grep -ohE '#patch[0-9A-Za-z]+' drivers/rknpu/*.c drivers/iommu/rockchip-iommu.c 2>/dev/null | sort -u | tr '\n' ' '")"
fi

echo "== syncing kernel sources =="
ssh $VM "mkdir -p /tmp/vmsync_"
# Sync the WHOLE driver, not a hand-listed subset: rknpu_reset.c and rknpu_ioctl.h joined the
# patch set later and a stale list silently builds the old file (that cost a wasted boot once).
if [ -n "$FROM_BRANCH" ]; then
    FILES=$(git -C "$KREPO" ls-tree -r --name-only "$FROM_BRANCH" -- drivers/rknpu drivers/iommu/rockchip-iommu.c \
            | grep -E '\.(c|h)$' || true)
else
    FILES=$(ssh $BOARD "cd ~/kbuild/linux-rockchip && ls drivers/rknpu/*.c drivers/rknpu/*.h drivers/rknpu/include/*.h drivers/iommu/rockchip-iommu.c 2>/dev/null || true")
fi
for f in $FILES; do
    if [ -n "$FROM_BRANCH" ]; then git -C "$KREPO" show "$FROM_BRANCH:$f" > /tmp/_vmf
    else                           ssh $BOARD "cat ~/kbuild/linux-rockchip/$f" > /tmp/_vmf; fi
    scp -q /tmp/_vmf $VM:/tmp/vmsync_/$(basename $f)
    ssh $VM "$P; docker cp /tmp/vmsync_/$(basename $f) kbuild:/root/linux-rockchip/$f"
done
# .config: a snapshot branch carries it as board/rk3588-board-<N>.config (.config itself is gitignored);
# otherwise take the board's live one. Getting this wrong silently builds a DIFFERENT kernel.
if [ -n "$FROM_BRANCH" ] && git -C "$KREPO" cat-file -e "$FROM_BRANCH:board" 2>/dev/null; then
    CFG=$(git -C "$KREPO" ls-tree -r --name-only "$FROM_BRANCH" -- board | grep -E '\.config$' | head -1)
    echo "   .config: $FROM_BRANCH:$CFG"
    git -C "$KREPO" show "$FROM_BRANCH:$CFG" > /tmp/_vmf
else
    echo "   .config: board live tree"
    ssh $BOARD "cat ~/kbuild/linux-rockchip/.config" > /tmp/_vmf
fi
scp -q /tmp/_vmf $VM:/tmp/vmsync_/.config
ssh $VM "$P; docker cp /tmp/vmsync_/.config kbuild:/root/linux-rockchip/.config"

echo "== building (16 cores) =="
BUILD_OUT=$(ssh $VM "$P; docker exec kbuild sh -c 'cd /root/linux-rockchip && make ARCH=arm64 olddefconfig >/dev/null 2>&1 && nice -n 5 make ARCH=arm64 -j16 Image modules > /root/vmbuild.log 2>&1; echo build-exit=\$?; grep -E \"error:|undefined reference|Error [0-9]|ld: \" /root/vmbuild.log | head -8'" 2>&1)
echo "$BUILD_OUT"
# ABORT ON A FAILED BUILD. This used to fall through to the install step and ship the PREVIOUS image,
# so the board booted a STALE kernel while the log still said "installed". An experiment then measures
# the old code and reads as a clean negative — which is exactly what happened once (a compile error in
# a fix was scored as "the fix does not work"). Never install what did not build.
if ! echo "$BUILD_OUT" | grep -q "build-exit=0"; then
    echo "vm_kbuild: BUILD FAILED — NOT installing. The board keeps its current kernel." >&2
    exit 1
fi

echo "== installing Image to the board =="
ssh $VM "$P; docker exec kbuild cat /root/linux-rockchip/arch/arm64/boot/Image" > /tmp/_vmImage
# Ask the VM (what we just built), not the board (what it happens to have) -- with --from-branch those
# can differ, and naming the image from the wrong tree installs it over the wrong file.
V=$(ssh $VM "$P; docker exec kbuild sh -c 'cd /root/linux-rockchip && make -s ARCH=arm64 kernelrelease'" | tr -d '\r')
scp -q /tmp/_vmImage $BOARD:/tmp/Image.vm
ssh $BOARD "sudo cp /tmp/Image.vm /boot/vmlinuz-$V && ls -l /boot/vmlinuz-$V"

if [ "$KEEP" = "1" ]; then
    echo "done: installed $V, VM left RUNNING (--keep). Stop it with: tools/util/vm_kbuild.sh --stop"
else
    echo "== shutting the VM down =="
    ssh $VM "$P; docker stop kbuild >/dev/null 2>&1; colima stop" 2>&1 | tail -1
    echo "done: installed $V, VM stopped"
fi
