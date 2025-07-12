#!/bin/bash
gnome-terminal -- bash -c "
echo "ok"
cd /home/nvidia/单目/aruco-3.1.12/build/utils_markermap/
./aruco_test_markermap live:5 /home/nvidia/marker_mapper1.0.15-2/build/utils/markerset.yml /home/nvidia/marker_mapper1.0.15-2/build/utils/markerset-cam.yml -s 0.267 -ref_id 4 -e 0 -pcd "/home/nvidia/单目/aruco-3.1.12/build/utils_markermap/out.pcd" -poses.txt "/home/nvidia/单目/aruco-3.1.12/build/utils_markermap/out.txt"
exec bash"
