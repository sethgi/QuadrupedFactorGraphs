from fk import Robot
from dataset import Dataset
import argparse
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


parser = argparse.ArgumentParser("data loader")
parser.add_argument("dataset_path")
args = parser.parse_args()

dataset = Dataset(args.dataset_path)
robot = Robot()


csv = "t,front_right,front_left,back_right,back_left\n"

for entry in tqdm(dataset):
    t = entry["time"]
    joint_states = entry["joint_states"]

    front_right, front_left, back_right, back_left = robot.forward(joint_states)

    fr_z = front_right[2, 3]
    fl_z = front_left[2, 3]
    br_z = back_right[2, 3]
    bl_z = back_left[2, 3]

    csv += f"{t},{fr_z},{fl_z},{br_z},{bl_z}\n"

with open("foot_heights.csv", 'w+') as out_f:
    out_f.write(csv)