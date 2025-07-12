#!/bin/bash
gnome-terminal -- bash -c "
echo nvidia|sudo -S  chmod +777 /dev/ttyTHS1
source /home/nvidia/miniconda3/etc/profile.d/conda.sh
conda activate myenv
cd  /home/nvidia/new_rec2/followmanta_sesssion
python landing_air.py
"



