#!/bin/bash
cd /home/nvidia/Downloads/aruco-3.1.12/build
cmake ..
make
cd /home/nvidia/Downloads/aruco-3.1.12/build/utils_markermap
./aruco_test_markermap WIN_20241204_15_23_03_Pro.mp4 /home/nvidia/Downloads/marker_mapper1.0.15/build/utils/markerset.yml /home/nvidia/Downloads/marker_mapper1.0.15/build/utils/markerset-cam.yml -s 0.335   -pcd "/home/nvidia/Downloads/aruco-3.1.12/build/utils_markermap/out$(date "+%H%M%S").pcd" -poses "/home/nvidia/Downloads/aruco-3.1.12/build/utils_markermap/out$(date "+%H%M%S").txt" 
