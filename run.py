import argparse
import gtsam
from build_gpus_imu_factors import ImuFactory
from dataset import Dataset
# from build_fk_factors import FkFactory
from build_preintegrated_fk_factors import PreintegratedFkFactory
from fk import Robot
from scipy.spatial.transform import Rotation
import numpy as np
import os

def dump_to_kitti(results: gtsam.Values, symbols, times, out_file):
    result = []

    for s, t in zip(symbols, times):
        pose = results.atPose3(s)

        xyz = pose.translation()
        rot = pose.rotation().matrix()
        res_tf_mat = np.identity(4)
        res_tf_mat[:3,:3] = rot
        res_tf_mat[:3,3] = xyz
        result.append(" ".join(["{:.12f}".format(x) for x in res_tf_mat[:3, :4].flatten()]))

    with open(out_file, 'w+') as f:
        f.write("\n".join(result))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("data")

    parser.add_argument("result_dir")

    parser.add_argument("--optimizer",
                        default=2,
                        type=int, 
                        help="0: DogLeg, 1: Gauss Newton, 2: Levenberg(default)")

    parser.add_argument("--duration",
                        "-D",
                        default=300,
                        type=int,
                        help="sequnece length (s)")
    
    parser.add_argument("--noise_scale",
                        default=0.25,
                        type=float,
                        help="noise added to generate fake GPS data (m)")

    parser.add_argument("--no_imu", default=False, action="store_true")
    parser.add_argument("--no_fk", default=False, action="store_true")

    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    print("Loading Dataset")
    dataset = Dataset(args.data)

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    print("Creating IMU and \"GPS\" Factors")
    imu_factor_factory = ImuFactory(not args.no_imu, True, args.noise_scale)
    symbols, times, gt_values = imu_factor_factory.add_factors(dataset, graph, initial, args.duration, 50)

    file_name = ["DogLeg", 'GaussNewton', "Levenberg"]
    print("Optimizing")
    print("Optimizer: " + file_name[args.optimizer])
    optimizer_list = [gtsam.DoglegOptimizer(graph, initial), gtsam.GaussNewtonOptimizer(graph, initial), gtsam.LevenbergMarquardtOptimizer(graph, initial)]
    opt = optimizer_list[args.optimizer]
    result = opt.optimize()
    dump_to_kitti(result, symbols, times, args.result_dir + "/imu.kitti")
    dump_to_kitti(gt_values, symbols, times, args.result_dir + "/gt.kitti")

    if not args.no_fk: 
        initial = result
        print("Adding FK Factors")
        robot = Robot()
        fk_factor_factory = PreintegratedFkFactory(robot, dataset)
        fk_symbols = fk_factor_factory.build_factors(graph, initial, (symbols,times))

        print("Optimizing")
        print("Optimizer: " + file_name[args.optimizer])
        opt = optimizer_list[args.optimizer]
        opt = gtsam.LevenbergMarquardtOptimizer(graph, initial)
        results = opt.optimize()
        dump_to_kitti(results, symbols, times, args.result_dir + "/fk.kitti")

        dump_to_kitti(results, fk_symbols["f"], times, args.result_dir + "/front_legs.kitti")
        dump_to_kitti(results, fk_symbols["r"], times, args.result_dir + "/rear_legs.kitti")


        original_file_name = ["imu.kitti", "gt.kitti", "fk.kitti", "front_legs.kitti", "rear_legs.kitti"]

        for name in original_file_name:
            old_name = args.result_dir + "/" + name
            new_name = args.result_dir + "/" + file_name[args.optimizer] + "_" + name
            os.rename(old_name, new_name)
