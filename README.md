## BJTU Visual Perception
1. **docking**  
   - Mainly for vertical docking; current air tests show good accuracy, control is still being optimized. Pool experiments will start after the control program is ready.  
   - `calibrate2.py` — calibration for small and center ArUco markers, outputs calibration matrix.  
   - `camera_configs_danmu.py` — monocular calibration parameters.  
   - `landing_air.py` — main docking program.

2. **obstacle_avoidance**  
   - Outputs obstacle information; currently supports dynamic obstacles, static obstacles (e.g., walls) need improvement. BJTU plans to adopt visual SLAM later.  
   - `obstacle_grid_avoid.py` — main wrapper for BJTU avoidance code.  
   - `obstacle_depth.py` — BJTU avoidance program.  
   - `camera_configs.py` — calibration parameters.

3. **following**  
   - Visual following; basic turning and straight-line following work, swarm following will be optimized by BJTU.  
   - `yolo_distance4.py` — main following program.  
   - `camera_configs.py` — calibration parameters.  
   - `getdepth1.py` — stereo depth estimation.  
   - `best-1024ls.pt` — current model in use.

4. **positioning**  
   - Uses wall-mounted ArUco markers for pool mapping and localization; tests completed but drift exists.  
   - `marker_mapper1.0.15` and `aruco-3.1.12/run.sh` — main launch scripts for mapping and localization.