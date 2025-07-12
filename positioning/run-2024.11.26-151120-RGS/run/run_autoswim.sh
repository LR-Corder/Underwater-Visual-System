#!/bin/bash
gnome-terminal -- bash -c "

echo nvidia|sudo -S  chmod +777 /dev/ttyTHS1
cd  /home/nvidia/ARUCO_AUTOSWIM
python3  /home/nvidia/ARUCO_AUTOSWIM/Aruco_autoswim_toserial.py
exec bash"
