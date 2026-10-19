from glob import glob

from setuptools import find_packages, setup

package_name = "pusht_policy"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Davide De Benedittis",
    maintainer_email="davide@cyber-fusion.io",
    description="Runs a swarp pusht_lidar policy on robots that sense a LaserScan and poses.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "policy_node = pusht_policy.policy_node:main",
            "episode_manager = pusht_policy.episode_manager:main",
            "aggregate = pusht_policy.aggregate:main",
            "sysid_gz = pusht_policy.sysid.gz_runner:main",
            "video_recorder = pusht_policy.video_recorder:main",
        ],
    },
)
