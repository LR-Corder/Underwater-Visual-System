#!/bin/bash
gnome-terminal -- bash -c "
cd /home/nvidia/Downloads/aruco-3.1.12/build/utils_markermap
./aruco_test_markermap live:-1 /home/nvidia/Downloads/marker_mapper1.0.15/build/utils/markerset.yml /home/nvidia/Downloads/marker_mapper1.0.15/build/utils/markerset-cam.yml -s 0.335 -ref_id 40 -e 0 -pcd "/home/nvidia/Downloads/aruco-3.1.12/build/utils_markermap/out$(date "+%H%M%S").pcd" -poses.txt "/home/nvidia/Downloads/aruco-3.1.12/build/utils_markermap/out$(date "+%H%M%S").txt" 
exec bash"
