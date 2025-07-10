
from setuptools import find_packages, setup
from glob import glob

package_name = "aggregative_optimization"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, glob("launch/aggregative_optimization_launch.py")),
        ("share/" + package_name, glob("resource/rviz_config.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "generic_agent = aggregative_optimization.the_agent:main",
            "plotter = aggregative_optimization.the_plotter:main",
            "rviz2_visualizer = aggregative_optimization.the_rviz2visualizer:main",
        ],
    },
)
