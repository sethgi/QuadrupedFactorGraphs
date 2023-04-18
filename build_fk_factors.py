import gtsam
from fk import Robot
import numpy as np

from dataset import Dataset
from tqdm import tqdm
from gtsam import symbol

class FkFactory:
    def __init__(self, robot: Robot, data: Dataset):
        self.dataset = data

        self.robot = robot

        self.joint_symbol_mapping = {
            "fr":"o",
            "fl":"l",
            "br":"p",
            "bl":"j"
        }

        self.time_tolerance = 1e-6

    def build_factors(self, graph, initial, symbol_data):
        
        symbols, times = symbol_data
        symbols = np.array(symbols)
        times = np.array(times)

        prev_contact_state = [False, False, False, False]
        prev_contact_symbols = [None, None, None, None]

        contact_start_symbol = [None, None, None, None]

        fk_symbols = {k: [] for k in self.joint_symbol_mapping.keys()}
        for i, entry in enumerate(tqdm(self.dataset)):
            t = entry["time"]

            dist_t = np.min(np.fabs(t - times))
            if dist_t > self.time_tolerance:
                if t > times.max():
                    break
                continue

            body_symbol = symbols[np.argmin(np.fabs(t-times))]

            joint_states = entry["joint_states"]
            fr_contact, fl_contact, br_contact, bl_contact = entry["ground_contacts"]

            T_fr, T_fl, T_br, T_bl = self.robot.forward(joint_states)
            J_fr, J_fl, J_br, J_bl = self.robot.jacobian(joint_states)

            contact_sigma = gtsam.noiseModel.Diagonal.Sigmas(self.robot.contact_sigma)

            if fr_contact:
                sigma = J_fr @ self.robot.joint_sigma @ J_fr.transpose() + self.robot.fk_prior_covariance
                sigma = gtsam.noiseModel.Gaussian.Covariance(sigma)
                fr_symbol = symbol(self.joint_symbol_mapping["fr"], i)
                fk_symbols["fr"].append(fr_symbol)

                graph.add(gtsam.BetweenFactorPose3(body_symbol, fr_symbol, gtsam.Pose3(T_fr), sigma))
                
                initial.insert(fr_symbol, initial.atPose3(body_symbol)*gtsam.Pose3(T_fr))
                
                # if prev_contact_state[0]:
                #     prev_fr_symbol = prev_contact_symbols[0]
                #     graph.add(gtsam.BetweenFactorPose3(prev_fr_symbol, fr_symbol, gtsam.Pose3(), contact_sigma))
                if contact_start_symbol[0] is None:
                    contact_start_symbol[0] = fr_symbol
                else:
                    prev_fr_symbol = contact_start_symbol[0]
                    graph.add(gtsam.BetweenFactorPose3(prev_fr_symbol, fr_symbol, gtsam.Pose3(), contact_sigma))

                prev_contact_symbols[0] = fr_symbol
            else:
                contact_start_symbol[0] = None


            if fl_contact:
                sigma = J_fl @ self.robot.joint_sigma @ J_fl.transpose() + self.robot.fk_prior_covariance
                sigma = gtsam.noiseModel.Gaussian.Covariance(sigma)
                fl_symbol = symbol(self.joint_symbol_mapping["fl"], i)
                fk_symbols["fl"].append(fl_symbol)

                initial.insert(fl_symbol, initial.atPose3(body_symbol)*gtsam.Pose3(T_fl))

                graph.add(gtsam.BetweenFactorPose3(body_symbol, fl_symbol, gtsam.Pose3(T_fl), sigma))
                # if prev_contact_state[1]:
                #     prev_fl_symbol = prev_contact_symbols[1]
                #     graph.add(gtsam.BetweenFactorPose3(prev_fl_symbol, fl_symbol, gtsam.Pose3(), contact_sigma))

                if contact_start_symbol[1] is None:
                    contact_start_symbol[1] = fl_symbol
                else:
                    prev_fl_symbol = contact_start_symbol[1]
                    graph.add(gtsam.BetweenFactorPose3(prev_fl_symbol, fl_symbol, gtsam.Pose3(), contact_sigma))

                prev_contact_symbols[1] = fl_symbol
            else:
                contact_start_symbol[1] = None

            if br_contact:
                sigma = J_br @ self.robot.joint_sigma @ J_br.transpose() + self.robot.fk_prior_covariance
                sigma = gtsam.noiseModel.Gaussian.Covariance(sigma)
                br_symbol = symbol(self.joint_symbol_mapping["br"], i)
                initial.insert(br_symbol, initial.atPose3(body_symbol)*gtsam.Pose3(T_br))
                fk_symbols["fl"].append(br_symbol)

                graph.add(gtsam.BetweenFactorPose3(body_symbol, br_symbol, gtsam.Pose3(T_br), sigma))
                # if prev_contact_state[2]:
                #     prev_br_symbol = prev_contact_symbols[2]
                #     graph.add(gtsam.BetweenFactorPose3(prev_br_symbol, br_symbol, gtsam.Pose3(), contact_sigma))

                if contact_start_symbol[2] is None:
                    contact_start_symbol[2] = br_symbol
                else:
                    prev_br_symbol = contact_start_symbol[2]
                    graph.add(gtsam.BetweenFactorPose3(prev_br_symbol, br_symbol, gtsam.Pose3(), contact_sigma))

                prev_contact_symbols[2] = br_symbol
            else:
                contact_start_symbol[2] = None

            if bl_contact:
                sigma = J_bl @ self.robot.joint_sigma @ J_bl.transpose() + self.robot.fk_prior_covariance
                sigma = gtsam.noiseModel.Gaussian.Covariance(sigma)
                bl_symbol = symbol(self.joint_symbol_mapping["bl"], i)
                initial.insert(bl_symbol, initial.atPose3(body_symbol)*gtsam.Pose3(T_bl))

                graph.add(gtsam.BetweenFactorPose3(body_symbol, bl_symbol, gtsam.Pose3(T_bl), sigma))
                # if prev_contact_state[3]:
                #     prev_bl_symbol = prev_contact_symbols[3]
                #     graph.add(gtsam.BetweenFactorPose3(prev_bl_symbol, bl_symbol, gtsam.Pose3(), contact_sigma))
                prev_contact_symbols[3] = bl_symbol

                if contact_start_symbol[3] is None:
                    contact_start_symbol[3] = bl_symbol
                else:
                    prev_bl_symbol = contact_start_symbol[3]
                    graph.add(gtsam.BetweenFactorPose3(prev_bl_symbol, bl_symbol, gtsam.Pose3(), contact_sigma))

            else:
                contact_start_symbol[3] = None
                
            prev_contact_state = [fr_contact, fl_contact, br_contact, bl_contact]