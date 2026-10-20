from glob import glob

from setuptools import find_packages, setup

package_name = "pusht_gz"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/worlds", glob("worlds/*.xacro")),
        ("share/" + package_name + "/models", glob("models/*.xacro")),
        ("share/" + package_name + "/maps", glob("maps/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Davide De Benedittis",
    maintainer_email="davide@cyber-fusion.io",
    description="Gazebo Harmonic Push-T world: T, goal marker, N Limos, bridge, reset helpers.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "gt_pose_relay = pusht_gz.gt_pose_relay:main",
            "set_pose = pusht_gz.gz_reset:main",
            "make_map = pusht_gz.landmarks:main",
        ],
    },
)
