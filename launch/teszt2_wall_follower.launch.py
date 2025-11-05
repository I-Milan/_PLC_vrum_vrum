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
            package='teszt2_wall_follower',
            executable='teszt2_wall_follower_node',
            name='teszt2_wall_follower',
            output='screen',
            parameters=[params_file]  # <-- nincs trükközés, simán átadjuk a fájlt
        ),
    ])
