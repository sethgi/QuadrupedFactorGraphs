from fk import Robot
from dataset import Dataset
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal


if __name__ == "__main__":
    parser = argparse.ArgumentParser("data loader")
    parser.add_argument("dataset_path")
    args = parser.parse_args()

    dataset = Dataset(args.dataset_path, False, True)
    robot = Robot()

    contacts = []
    
    fr, fl, br, bl = [], [], [], []

    i = 0
    for entry in tqdm(dataset):
        i += 1
        t = entry["time"]

        # if i > 30_000:
        #     break

        joint_states = entry["joint_states"]

        T_b_fr, T_b_fl, T_b_br, T_b_bl = robot.forward(joint_states)

        T_wb = dataset.get_gt_pose(t)
        if T_wb is None:
            break
        
        contact = [0,0,0,0]

        T_fr = T_wb @ T_b_fr
        T_fl = T_wb @ T_b_fl

        z_fr = T_fr[2, 3]
        z_fl = T_fl[2, 3]

        # This is a hack around bad IMU calibration.
        if len(fr) > 0:
            z_fr -= fr[0]
        if len(fl) > 0:
            z_fl -= fl[0]

        if z_fr < z_fl:
            contact[0] = 1
        else:
            contact[1] = 1

        T_br = T_wb @ T_b_br
        T_bl = T_wb @ T_b_bl

        z_br = T_br[2, 3]
        z_bl = T_bl[2, 3]
        
        if len(br) > 0:
            z_br -= br[0]
        if len(bl) > 0:
            z_bl -= bl[0]

        if z_br < z_bl:
            contact[2] = 1
        else:
            contact[3] = 1

        contacts.append(contact)

        fr.append(z_fr)
        fl.append(z_fl)
        br.append(z_br)
        bl.append(z_bl)

        # if i % 1000 == 0:
        #     breakpoint()
    fr = np.hstack(fr)
    fl = np.hstack(fl)
    br = np.hstack(br)
    bl = np.hstack(bl)

    contacts = np.vstack(contacts)

    # plt.plot(range(len(fr)), fr, label="fr")
    # plt.plot(range(len(fl)), fl, label="fl")
    
    # plt.plot(range(len(fr)), fr * contacts[:, 0], label="fr_contact")
    # plt.plot(range(len(fr)), fl * contacts[:, 1], label="fl_contact")

    # plt.legend()
    # plt.show()
    
    np.savetxt("contacts.csv", contacts, delimiter=",", fmt="%d")