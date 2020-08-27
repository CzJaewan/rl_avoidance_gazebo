# rl_avoidance_gazebo
rl_collision_avoidance test in gazebo simulator 

## SETTING
### Model
```
  git clone https://github.com/CzJaewan/servingbot.git
```
### Gazebo & Amcl & Globalplanner
```
  git clone https://github.com/CzJaewan/rl_avoidance_gazebo.git
```
## RUN
- Servingbot Gazebo world
```
  roslaunch servingbot_gazebo servingbot_rl_world.launch
``` 
- Amcl & Map & Globalplanner
```
  roslaunch gazebo_rl_test servingbot_rl.launch 
```
- PPO local planner
```
  cd ~/catkin_ws/src/rl_avoidance_gazebo/GAZEBO_TEST
  python test_run.py
```
- Policy Data link
```
  https://drive.google.com/drive/folders/1vQ1XKbU1Lid40Vm92H0UDRcvquYNSrYi?usp=sharing
```
