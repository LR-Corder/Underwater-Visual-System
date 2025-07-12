#!/bin/bash
gnome-terminal -- bash -c "
cd /home/nvidia/单目/aruco-3.1.12/build/utils_markermap
pcl_viewer out.pcd
"
