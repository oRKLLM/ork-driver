#!/bin/sh
# vm_kbuild.sh — build the board kernel on the .239 Colima VM (16 cores, ~7 min vs ~45 on the SBC),
# install the Image to the board, and SHUT THE VM DOWN again.
#
#   tools/util/vm_kbuild.sh [--keep]      --keep leaves the VM RUNNING for the next build
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
if [ "$1" = "--keep" ]; then KEEP=1; shift; fi
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

echo "== syncing current kernel sources from the board =="
ssh $VM "mkdir -p /tmp/vmsync_"
# Sync the WHOLE driver, not a hand-listed subset: rknpu_reset.c and rknpu_ioctl.h joined the
# patch set later and a stale list silently builds the old file (that cost a wasted boot once).
FILES=$(ssh $BOARD "cd ~/kbuild/linux-rockchip && ls drivers/rknpu/*.c drivers/rknpu/*.h drivers/rknpu/include/*.h drivers/iommu/rockchip-iommu.c 2>/dev/null || true")
for f in $FILES .config; do
    ssh $BOARD "cat ~/kbuild/linux-rockchip/$f" > /tmp/_vmf
    scp -q /tmp/_vmf $VM:/tmp/vmsync_/$(basename $f)
    ssh $VM "$P; docker cp /tmp/vmsync_/$(basename $f) kbuild:/root/linux-rockchip/$f"
done

echo "== building (16 cores) =="
ssh $VM "$P; docker exec kbuild sh -c 'cd /root/linux-rockchip && make ARCH=arm64 olddefconfig >/dev/null 2>&1 && nice -n 5 make ARCH=arm64 -j16 Image modules > /root/vmbuild.log 2>&1; echo build-exit=\$?; grep -E \"error:\" /root/vmbuild.log | head -5'"

echo "== installing Image to the board =="
ssh $VM "$P; docker exec kbuild cat /root/linux-rockchip/arch/arm64/boot/Image" > /tmp/_vmImage
V=$(ssh $BOARD 'cd ~/kbuild/linux-rockchip && make -s ARCH=arm64 kernelrelease')
scp -q /tmp/_vmImage $BOARD:/tmp/Image.vm
ssh $BOARD "sudo cp /tmp/Image.vm /boot/vmlinuz-$V && ls -l /boot/vmlinuz-$V"

if [ "$KEEP" = "1" ]; then
    echo "done: installed $V, VM left RUNNING (--keep). Stop it with: tools/util/vm_kbuild.sh --stop"
else
    echo "== shutting the VM down =="
    ssh $VM "$P; docker stop kbuild >/dev/null 2>&1; colima stop" 2>&1 | tail -1
    echo "done: installed $V, VM stopped"
fi
