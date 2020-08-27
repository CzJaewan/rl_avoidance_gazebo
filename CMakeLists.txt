cmake_minimum_required(VERSION 3.5)
project(gazebo_rl_test)

## Compile as C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11 -O3)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  costmap_2d
  dynamic_reconfigure
  geometry_msgs
  nav_msgs
  ackermann_msgs
  message_generation
  move_base
  roscpp
  rospy
  std_msgs
  tf
  visualization_msgs
)

catkin_package(
   INCLUDE_DIRS include
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories( ${catkin_INCLUDE_DIRS} include )
include_directories(/usr/include)
link_directories(/usr/lib)

# Own Global planner for tracking desired trajectory
# add_library(global_planner_libs src/global_planner/global_planner.cpp)
# target_link_libraries(global_planner_libs ipopt ${catkin_LIBRARIES} )

