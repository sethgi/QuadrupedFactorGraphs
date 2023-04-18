"""
This file has been adapted from:

GTSAM Copyright 2010-2019, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved

See LICENSE for the license information

A script validating and demonstrating the ImuFactor inference.

Author: Frank Dellaert, Varun Agrawal
"""

import argparse


import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import B, V, X, I

import gtsam
from scipy.spatial.transform import Rotation as R

from dataset import Dataset

BIAS_KEY = B(0)
GRAVITY = -9.81

T_body_imu = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])

T_imu_body = np.linalg.inv(T_body_imu)

P_imu_body = gtsam.Pose3(T_imu_body)
P_body_imu = gtsam.Pose3(T_body_imu)


class ImuFactory():
    """Class to run example of the Imu Factor."""
    def __init__(self, use_imu: bool = True, use_gps: bool = True, noise_scale: float = 0.1):
        self.firstPriorNoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        self.velNoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.3)

        self.extrinsicsNoise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-5)

        if use_imu:
            self.priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e3, 1e3, 1e3, 0.1, 0.1, 0.1]))

        else:
            self.priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))

        params = gtsam.PreintegrationParams.MakeSharedU(GRAVITY)

        # Some arbitrary noise sigmas
        gyro_sigma = 0.5
        accel_sigma = 0.5
        I_3x3 = np.eye(3)
        params.setGyroscopeCovariance(gyro_sigma**2 * I_3x3)
        params.setAccelerometerCovariance(accel_sigma**2 * I_3x3)
        params.setIntegrationCovariance(1e-7**2 * I_3x3)

        self.params = params

        self.use_imu = use_imu
        self.use_gps = use_gps

        self.noise_scale = noise_scale

    def add_factors(self, dataset, graph, initial, duration, imu_factor_step=100):
        """
        Main runner.

        Args:
            T: Total trajectory time.
            compute_covariances: Flag indicating whether to compute marginal covariances.
            verbose: Flag indicating if printing should be verbose.
        """
        accBias = np.array([0, 0, 0])
        gyroBias = np.array([0, 0, 0])
        initBias = gtsam.imuBias.ConstantBias(accBias, gyroBias)

        file = dataset.sensor_file
        gt_file = dataset.vicon_file

        dataFile = open(file)
        data_list = dataFile.readlines() # 1000 hz

        if self.use_imu:
            data_list = data_list[:duration*1000+1]
        else:
            data_list = data_list[:duration*250+1]

        gtDataFile = open(gt_file)
        gt_data_list = gtDataFile.readlines() # 250 hz

        symbols, times = [], []

        # initialize data structure for pre-integrated IMU measurements
        pim = gtsam.PreintegratedImuMeasurements(self.params, initBias)

        i = 0
        gt_origin = np.array(gt_data_list[0].split(',')).astype(np.float64)
        gt_tf_mat = np.identity(4)
        r = R.from_euler('yxz', [[gt_origin[1], gt_origin[2], gt_origin[3]]], degrees=True)
        gt_tf_mat[:3,:3] = r.as_matrix()
        gt_tf_mat[:3,3] = np.array([gt_origin[4], gt_origin[5], gt_origin[6]])/1000.

        gt_values = gtsam.Values()
        gt_values.insert(I(0), gtsam.Pose3(gt_tf_mat))
        gt_values.insert(X(0), gtsam.Pose3(gt_tf_mat) * P_imu_body)

        pose_0 = gtsam.Pose3(gt_tf_mat)
        initial.insert(I(i), pose_0)
        initial.insert(X(i), pose_0 * P_imu_body)
        graph.push_back(gtsam.BetweenFactorPose3(I(0), X(0), P_imu_body, self.extrinsicsNoise))
        symbols.append(X(i))
        times.append(0.)
        zero_velocity = np.array([0, 0, 0])

        if self.use_imu:
            initial.insert(V(i), zero_velocity)
            initial.insert(BIAS_KEY, initBias)
        
        # add prior on beginning
        graph.push_back(gtsam.PriorFactorPose3(I(i), pose_0, self.firstPriorNoise))
        if self.use_imu:
            graph.push_back(gtsam.PriorFactorVector(V(i), zero_velocity, self.velNoise))

        old_ts = -0.001
        # imu_factor_step = 100 # add imu factor every 100 ms
        for k, line in enumerate(data_list):

            gt_data = np.array(gt_data_list[round(k/4.)].split(',')).astype(np.float64)
            data = line.split(',')
            ts = float(data[0])
            delta_t = ts - old_ts

            if self.use_imu:
                # GX5 IMU data
                GX5measuredOmega = np.array([float(data[43]), float(data[44]), float(data[45])])
                GX5measuredAcc = np.array([float(data[46]), float(data[47]), float(data[48])])
                # KVH IMU data
                KVHmeasuredOmega = np.array([float(data[49]), float(data[50]), float(data[51])])
                KVHmeasuredAcc = np.array([float(data[52]), float(data[53]), float(data[54])])
                # only use GX5 for now
                pim.integrateMeasurement(KVHmeasuredAcc, KVHmeasuredOmega, delta_t)

            # Plot IMU many times
            # if k % imu_factor_step == 0:
            #     self.plotImu(ts, measuredOmega, measuredAcc)
            
            if k==0:
                gt_tf_mat = np.identity(4)
                gt_tf_mat[:3,:3] = R.from_euler('yxz', [[gt_data[1], gt_data[2], gt_data[3]]], degrees=True).as_matrix()
                gt_tf_mat[:3,3] = np.array([gt_data[4], gt_data[5], gt_data[6]])/1000.

            if k>0 and k % imu_factor_step == 0:
                gt_tf_mat = np.identity(4)
                gt_tf_mat[:3,:3] = R.from_euler('yxz', [[gt_data[1], gt_data[2], gt_data[3]]], degrees=True).as_matrix()
                gt_tf_mat[:3,3] = np.array([gt_data[4], gt_data[5], gt_data[6]])/1000.

                fake_gps_tf_mat = np.identity(4)
                fake_gps_tf_mat[:3,:3] = R.from_euler('yxz', [gt_data[1:4]], degrees=True).as_matrix()
                fake_gps_tf_mat[:3,3] = np.array([gt_data[4:7]])/1000. + np.random.uniform(-1.0, 1.0, size=(3,))*self.noise_scale

                pose = gtsam.Pose3(fake_gps_tf_mat)
                graph.push_back(gtsam.PriorFactorPose3(I(i + 1), pose, self.priorNoise))
                if self.use_imu:
                    graph.push_back(gtsam.PriorFactorVector(V(i + 1), zero_velocity, self.velNoise))
                    # create IMU factor every second
                    factor = gtsam.ImuFactor(I(i), V(i), I(i + 1), V(i + 1),
                                            BIAS_KEY, pim)
                    graph.push_back(factor)
                    pim.resetIntegration()
                initial.insert(I(i + 1), pose)
                if self.use_imu:
                    initial.insert(V(i + 1), zero_velocity)
                    time = data[0]
                else:
                    time = k / 250.0

                graph.push_back(gtsam.BetweenFactorPose3(I(i+1), X(i+1), P_imu_body, self.extrinsicsNoise))
                initial.insert(X(i+1), pose*P_imu_body)

                symbols.append(X(i+1))
                gt_values.insert(I(i+1), gtsam.Pose3(gt_tf_mat))
                gt_values.insert(X(i+1), gtsam.Pose3(gt_tf_mat)*P_imu_body)
                times.append(float(time))
                i += 1
                
            old_ts = ts
        return symbols, times, gt_values

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str)
    
    parser.add_argument("--duration",
                        "-D",
                        default=300,
                        type=int,
                        help="sequnece lenght (s)")
    parser.add_argument("--use_imu",
                        "-IMU",
                        default=False,
                        action='store_true')
    parser.add_argument("--noise_scale",
                        default=0.25,
                        type=float,
                        help="noise added to generate fake GPS data (m)")

    args = parser.parse_args()

    factor_factory = ImuFactory(args.use_imu, True, args.noise_scale)

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    data = Dataset(args.data)
    factor_factory.add_factors(data, graph, initial, args.duration)