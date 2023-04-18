import gtsam
from fk import Robot
import numpy as np

from dataset import Dataset
from tqdm import tqdm
from gtsam import symbol

class PreintegratedFkFactory:
    def __init__(self, robot: Robot, data: Dataset):
        self.dataset = data

        self.robot = robot

        self.time_tolerance = 1e-6

    def build_factors(self, graph, initial, symbol_data):
        
        symbols, times = symbol_data
        symbols = np.array(symbols)
        times = np.array(times)

        contact_delta = [np.eye(4)]*2
        contact_sigma = [np.zeros((6,6))]*2

        prev_time = 0

        fk_count = 0
        
        fk_symbols = {"f":[], "r":[]}
        
        prev_contacts = np.array([1,1,1,1])
        
        # front and back are pairs that switch back/forth
        contact_switch_map = {0:1, 1:0, 2:3, 3:2}

        for i, entry in enumerate(tqdm(self.dataset)):
            t = entry["time"]

            if t > times.max():
                break

            joint_states = entry["joint_states"]
            contact_states = entry["ground_contacts"]

            transforms = self.robot.forward(joint_states)
            jacobians = self.robot.jacobian(joint_states)


            # Integrate contacts
            for j in range(4):
                if not contact_states[j]:
                    continue

                if prev_contacts[j] == 0: # if a contact switch

                    switch_from_idx = contact_switch_map[j]
                    H_old_to_new = np.linalg.inv(transforms[switch_from_idx]) @ transforms[j]
                    contact_delta[j//2] = contact_delta[j//2] @ H_old_to_new

                    H_new_to_old = gtsam.Pose3(H_old_to_new).inverse()
                    H_new_to_old_adj = H_new_to_old.AdjointMap()
                    
                    J_old_to_new = self.robot.cross_jacobian(switch_from_idx, j, joint_states)

                    joint_sigma = np.vstack((
                        np.hstack((self.robot.joint_sigma, np.zeros((3,3)))),
                        np.hstack((np.zeros((3,3)), self.robot.joint_sigma))
                    ))

                    contact_sigma[j//2] = H_new_to_old_adj @ contact_sigma[j//2] @ H_new_to_old_adj.T \
                                    + J_old_to_new @ joint_sigma @ J_old_to_new.T
                else:
                    contact_sigma[j//2] += np.diag(self.robot.contact_sigma) * (t - prev_time)

                
            prev_contacts = contact_states


            dist_t = np.min(np.fabs(t - times))
            
            # If it's time to make the factors
            if dist_t <= self.time_tolerance:
                body_symbol = symbols[np.argmin(np.fabs(t-times))]

                # Add FK factors first
                for j in range(4):
                    if contact_states[j]:
                        sigma = jacobians[j] @ self.robot.joint_sigma @ jacobians[j].T + self.robot.fk_prior_covariance
                        sigma = gtsam.noiseModel.Gaussian.Covariance(sigma)

                        # One symbol for "front", one for "rear"
                        symbol_prefix = "f" if j // 2 == 0 else "r"
                        fk_symbol = symbol(symbol_prefix, len(fk_symbols[symbol_prefix]))

                        fk_symbols[symbol_prefix].append(fk_symbol)

                        graph.add(gtsam.BetweenFactorPose3(body_symbol, fk_symbol, gtsam.Pose3(transforms[j]), sigma))
                        initial.insert(fk_symbol, initial.atPose3(body_symbol)*gtsam.Pose3(transforms[j]))

                if len(fk_symbols["f"]) > 1:
                    # Front
                    fk_count = len(fk_symbols["f"]) - 1
                    from_symbol = symbol("f", fk_count - 1)
                    to_symbol = symbol("f", fk_count)

                    a = initial.atPose3(from_symbol).matrix()
                    b = initial.atPose3(to_symbol).matrix()
                    T_est = np.linalg.inv(a) @ b

                    transform = gtsam.Pose3(contact_delta[0])
                    sigma = gtsam.noiseModel.Gaussian.Covariance(contact_sigma[0])
                    graph.add(gtsam.BetweenFactorPose3(from_symbol, to_symbol, transform, sigma))

                if len(fk_symbols["r"]) > 1:
                    # Rear
                    fk_count = len(fk_symbols["r"]) - 1
                    from_symbol = symbol("r", fk_count - 1)
                    to_symbol = symbol("r", fk_count)

                    a = initial.atPose3(from_symbol).matrix()
                    b = initial.atPose3(to_symbol).matrix()
                    T_est = np.linalg.inv(a) @ b

                    transform = gtsam.Pose3(contact_delta[1])
                    sigma = gtsam.noiseModel.Gaussian.Covariance(contact_sigma[1])
                    graph.add(gtsam.BetweenFactorPose3(from_symbol, to_symbol, transform, sigma))

                    
                contact_delta = [np.eye(4)]*2
                contact_sigma = [np.zeros((6,6))]*2            
            

            prev_time = t

        return fk_symbols