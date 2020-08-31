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
## RUN used Waitpoint 
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

## RUN used Look a head
- Servingbot Gazebo world
```
  roslaunch servingbot_gazebo servingbot_rl_world.launch
``` 
- Amcl & Map & Globalplanner & Look a head generator
```
  roslaunch gazebo_rl_test servingbot_rl_lookahead.launch 
```
- PPO local planner
```
  cd ~/catkin_ws/src/rl_avoidance_gazebo/GAZEBO_TEST
  change the from test_world import StageWorld' -> 'from test_world_LAH import StageWorld' in test_run.py 
  python test_run.py
  
```

## Policy Data
- Policy Data link
```
  https://drive.google.com/drive/folders/1vQ1XKbU1Lid40Vm92H0UDRcvquYNSrYi?usp=sharing

  - origin : rl-collision-avoidance reward
  - rc : rl-collision-avoidance reward, collision reward(50)
  - rc2 : rl-collision-avoidance reward, collision reward(30)
  - rg : rl-collision-avoidance reward, time_distance reward(abs)
  - rt : rl-collision-avoidance reward, time reward(-0.1)
  - rt2 : rl-collision-avoidance reward, time reward(-0.5)
  - rt05 : rl-collision-avoidance reward, time reward(-0.05)
  - rw : rl-collision-avoidance reward, omega reward remove
```  
  
