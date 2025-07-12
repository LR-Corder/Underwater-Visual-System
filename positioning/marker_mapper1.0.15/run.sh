#!/bin/bash
cd /home/nvidia/Downloads/marker_mapper1.0.15/build
cmake ..
make
cd utils
./mapper_from_video live markerset-cam.yml 0.335 -ref $1 -d ARUCO -noshow
echo ./mapper_from_video live markerset-cam.yml 0.335 -ref $1 -d ARUCO -noshow
