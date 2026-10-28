#!/usr/bin/env bash
# run a python script inside the Noetic chroot with D:/share at /mnt and this folder at /tmp
R=/home/taeyun_kim/.codex/morai-mpc-runsh-wsl-20261010-a1/noetic-rootfs
W=$(cd "$(dirname "$0")" && pwd)
exec unshare -Urm --fork bash -c "mount --bind /mnt/d/share $R/mnt && mount --bind $W $R/tmp && chroot $R /bin/bash -c \"source /opt/ros/noetic/setup.bash; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; cd /tmp; python3 $*\""
