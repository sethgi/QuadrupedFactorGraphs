from fk import Robot
from dataset import Dataset
import argparse
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import rospy
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import rosbag

def mat_to_tf(mat, link_name, time):
    msg = TransformStamped()
    msg.header.frame_id = "robot/base_link"
    msg.header.stamp = time
    msg.child_frame_id = f"robot/{link_name}"

    quat = R.from_matrix(mat[:3,:3]).as_quat()
    xyz = mat[:3,3]

    msg.transform.translation.x = xyz[0]
    msg.transform.translation.y = xyz[1]
    msg.transform.translation.z = xyz[2]

    msg.transform.rotation.x = quat[0]
    msg.transform.rotation.y = quat[1]
    msg.transform.rotation.z = quat[2]
    msg.transform.rotation.w = quat[3]

    return msg


parser = argparse.ArgumentParser("data loader")
parser.add_argument("dataset_path")
args = parser.parse_args()

dataset = Dataset(args.dataset_path)
robot = Robot()

bag = rosbag.Bag("./tf_bag.bag", 'w')

for entry in tqdm(dataset):
    t = entry["time"]
    ros_t = rospy.Time.from_sec(t)
    joint_states = entry["joint_states"]

    for joint_idx, joint_name in enumerate(["shoulder", "hip", "knee", "foot"]):
        front_right, front_left, back_right, back_left = robot.forward(joint_states, joint_idx)

        tf_msg = TFMessage()
        tf_msg.transforms.append(mat_to_tf(front_right, f"front_right/{joint_name}", ros_t))    
        tf_msg.transforms.append(mat_to_tf(front_left, f"front_left/{joint_name}", ros_t))    
        tf_msg.transforms.append(mat_to_tf(back_right, f"back_right/{joint_name}", ros_t))    
        tf_msg.transforms.append(mat_to_tf(back_left, f"back_left/{joint_name}", ros_t))
        bag.write("/tf", tf_msg, t=ros_t)

bag.close()