# _PLC_vrum_vrum
A 2025/2026/1 félév leggyorsabb robot versenyautó kódjának fejlesztésee



hasznalat:

colcon build

source insall/setup.bash

source/opt/ros/humble/setup.bash

ros2 launch robotverseny_bringup  roboworks.launch.py

itt az Rvizben: 

Add-> Marker Array-> topic: /goal_viz

(goal pontok megjelenitese a robot elott)

uj terminalba

source insall/setup.bash

source/opt/ros/humble/setup.bash

ros2 launch teszt2_wall_follower teszt2_wall_follower.launch.py
