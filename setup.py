from setuptools import setup

package_name = 'teszt2_wall_follower'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/teszt2_wall_follower.launch.py']),
        ('share/' + package_name + '/config', ['config/params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='LiDAR alapú fal-követő vezérlés bal/jobb oldalra (ROS 2 Humble).',
    license='MIT',
    entry_points={
        'console_scripts': [
            'teszt2_wall_follower_node = teszt2_wall_follower.teszt2_wall_follower_node:main',
        ],
    },
)
