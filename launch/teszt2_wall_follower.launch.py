from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Alapértelmezett paraméterfájl: a csomag saját params.yaml-je
    default_params = PathJoinSubstitution([
        FindPackageShare('teszt2_wall_follower'),
        'config',
        'params.yaml'
    ])

    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Paraméterfájl (YAML) a node-hoz.'
        ),

        Node(
            package="teszt2_wall_follower",
            executable="polyline_builder_node",
            name="polyline_builder",
            parameters=[
            LaunchConfiguration('params_file'),
            {"frame_id": "base_link"},
            {"use_sim_time": True},       
            ],
            
        ),

        Node(
            package="teszt2_wall_follower",
            executable="teszt2_wall_follower_node",
            name="teszt2_wall_follower",
            parameters=[LaunchConfiguration('params_file'),
            {"use_sim_time": True},
            ],
        ),
    ])
