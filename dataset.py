import pandas as pd
import numpy as np

import argparse

import os

from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d

R_body_gt = np.array([
    [1,0,0],
    [0,0,1],
    [0,-1,0]
])

R_gt_body = R_body_gt.T


class Dataset:
    def __init__(self, data_path, load_ground_contacts=True, load_groundtruth = False):
        columns = ["time", "real_time", str(2)]

        leg_names = ["rf", "lf", "hr", "hl"]
        joint_names = ["shoulder", "hip", "knee"]

        good_column_names = ["time", "real_time"]
        
        for leg in leg_names:

            for joint in joint_names:
                columns.append(f"{leg}_{joint}")
                good_column_names.append(f"{leg}_{joint}")
        
        if load_ground_contacts:
            for l in leg_names:
                good_column_names.append(f"c_{l}")

        while len(columns) < 49:
            columns.append(str(len(columns)))

        columns += ["imu_acc_x", "imu_acc_y", "imu_acc_z", "imu_gyro_x", "imu_gyro_y", "imu_gyro_z"]
        good_column_names += columns

        while len(columns) < 81:
            columns.append(str(len(columns)))

        dataframe = pd.read_csv(f"{data_path}/sensors.csv", names=columns)

        if load_ground_contacts:
            ground_contacts = pd.read_csv(f"{data_path}/contacts.csv", delimiter=',', names=["c_rf", "c_lf", "c_hr", "c_hl"], header=None)
            self.contacts = ground_contacts.to_numpy()
            dataframe = pd.concat([dataframe, ground_contacts], axis=1)

        self.dataframe = dataframe[good_column_names]

        self.has_ground_contacts = load_ground_contacts

        self.sensor_file = f"{data_path}/sensors.csv"
        self.vicon_file = f"{data_path}/vicon.csv"

        if load_groundtruth:
            df = pd.read_csv(f"{data_path}/vicon.csv", names=["wx", "wy", "wz", "x", "y", "z"])
            self.gt_data = df.to_numpy()
            self.gt_time = np.arange(self.gt_data.shape[0]) / 250.

            rots = self.gt_data[:, :3]
            rots = Rotation.from_euler('yxz', rots, degrees=True)

            body_corr = Rotation.from_matrix(R_gt_body)

            rots = rots * body_corr

            self.gt_slerp = Slerp(self.gt_time, rots)
            self.gt_xyz = interp1d(self.gt_time, self.gt_data[:, 3:] / 1000., axis=0)
            
            self._groundtruth_idx = 0
            
        else:
            self._groundtruth_idx = None

    def get_gt_pose(self, time):
        try:
            gt_rot = self.gt_slerp(time)
            gt_xyz = self.gt_xyz(time)

            gt_rot = gt_rot.as_matrix()

            T_gt = np.eye(4)
            T_gt[:3,:3] = gt_rot
            T_gt[:3, 3] = gt_xyz

            return T_gt
        except:
            return None
        
    def __iter__(self):
        self.df_iter = self.dataframe.itertuples()
        return self
        
    def __next__(self):
        data = next(self.df_iter)

        time = data.time
        real_time = data.real_time

        leg_names = ["rf", "lf", "hr", "hl"]
        joint_names = ["shoulder", "hip", "knee"]

        vals = []
        for leg in leg_names:
            for joint in joint_names:
                vals.append(getattr(data, f"{leg}_{joint}"))

        imu_cols = ["imu_acc_x", "imu_acc_y", "imu_acc_z", "imu_gyro_x", "imu_gyro_y", "imu_gyro_z"]
        imu_data = [getattr(data, col) for col in imu_cols]

        result = {
            "time": time,
            "real_time": real_time,
            "joint_states": np.array(vals).reshape(4, 3),
            "imu": imu_data
        }

        if self.has_ground_contacts:
            ground_contacts = []
            for l in leg_names:
                ground_contacts.append(getattr(data, f"c_{l}"))
            result["ground_contacts"] = ground_contacts

        if self._groundtruth_idx is not None:
            T_gt = self.get_gt_pose(time)
            if T_gt is not None:
                result["ground_truth_pose"] = T_gt
        return result

    def __len__(self):
        return len(self.dataframe)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("data loader")
    parser.add_argument("dataset_path")
    args = parser.parse_args()

    dataset = Dataset(args.dataset_path)