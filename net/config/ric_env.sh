#!/bin/sh
export ROS_MASTER_URI=http://bobin-HP-Pro-2080-Microtower-PC:11311
export ROS_HOSTNAME=chen-tian-pc
. /opt/ros/indigo/setup.sh
exec "$@"
