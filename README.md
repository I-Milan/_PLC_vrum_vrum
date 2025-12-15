# _PLC_vrum_vrum
A 2025/2026/1 félév leggyorsabb robot versenyautó kódjának fejlesztése

Fejlesztették: Vass Levente (FNO5TU), Istiván Milán (U9YTQZ)

Használat:

colcon build

source insall/setup.bash

source/opt/ros/humble/setup.bash

ros2 launch robotverseny_bringup  roboworks.launch.py

itt az Rvizben: 

Add-> Marker Array-> topic: /goal_viz

(goal pontok megjelenítese a robot előtt)

A Marker jobban látszik, hogyha a Steering jelölése ki van kapcsolva

Add-> Path-> topic: polyline_left

Add-> Path-> topic: polyline_right

(Polyilne_builder_node által publish-olt fal vonalak)

Ezt követően javasoljuk a laser kikapcsolását, mert úgy jobban látszanak a fal-vonalak

Új terminálba:

source insall/setup.bash

source/opt/ros/humble/setup.bash

ros2 launch teszt2_wall_follower teszt2_wall_follower.launch.py
