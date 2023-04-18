import numpy as np
from scipy.linalg import expm
from modern_robotics import JacobianSpace
from scipy.linalg import logm

class Leg:
    def __init__(self, is_right: bool, is_front: bool, params: dict):

        if is_right:
            y_sign = -1
        else:
            y_sign = 1

        if is_front:
            x_sign = 1
        else:
            x_sign = -1

        d_fh = params["d_fh"]
        d_lr = params["d_lr"]
        l0 = params["l0"]
        l1 = params["l1"]
        l2 = params["l2"] + params["l3"]

        if is_right:
            self.omega_0 = np.array([1,0,0])
        else:
            self.omega_0 = np.array([-1, 0, 0])
            
        self.omega_1 = np.array([0,1,0])
        self.omega_2 = np.array([0,1,0])

        self.w_0 = np.array([[0, -self.omega_0[2], self.omega_0[1]],
                             [self.omega_0[2], 0, -self.omega_0[0]],
                             [-self.omega_0[1], self.omega_0[0], 0]])

        self.w_1 = np.array([[0, -self.omega_1[2], self.omega_1[1]],
                             [self.omega_1[2], 0, -self.omega_1[0]],
                             [-self.omega_1[1], self.omega_1[0], 0]])

        self.w_2 = np.array([[0, -self.omega_2[2], self.omega_2[1]],
                             [self.omega_2[2], 0, -self.omega_2[0]],
                             [-self.omega_2[1], self.omega_2[0], 0]])

        self.q_0 = np.array([x_sign * d_fh/2, y_sign * d_lr/2, 0])
        self.q_1 = np.array([x_sign * d_fh/2, y_sign * d_lr/2, -l0])
        self.q_2 = np.array([x_sign * d_fh/2, y_sign * d_lr/2, -(l0 + l1)])
        self.q_3 = np.array([x_sign * d_fh/2, y_sign * d_lr/2, -(l0 + l1 + l2)])

        # hack around intellisense nonsense
        def do_cross(a,b) -> np.array:
            return np.cross(a,b)

        self.v_0 = -do_cross(self.omega_0,self.q_0)
        self.v_1 = -do_cross(self.omega_1,self.q_1)
        self.v_2 = -do_cross(self.omega_2,self.q_2)

        self.s0 = np.vstack((self.omega_0.reshape(-1, 1), self.v_0.reshape(-1, 1)))
        self.s1 = np.vstack((self.omega_1.reshape(-1, 1), self.v_1.reshape(-1, 1)))
        self.s2 = np.vstack((self.omega_2.reshape(-1, 1), self.v_2.reshape(-1, 1)))

        self.S0 = np.hstack((self.w_0, self.v_0.reshape(-1, 1)))
        self.S0 = np.vstack((self.S0, np.zeros(4)))

        self.S1 = np.hstack((self.w_1, self.v_1.reshape(-1, 1)))
        self.S1 = np.vstack((self.S1, np.zeros(4)))

        self.S2 = np.hstack((self.w_2, self.v_2.reshape(-1, 1)))
        self.S2 = np.vstack((self.S2, np.zeros(4)))


        # Shoulder
        M_r = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        M = np.hstack((M_r, self.q_0.reshape(3, 1)))

        self.M0 = np.vstack((M, [0,0,0,1]))

        M = np.hstack((M_r, self.q_1.reshape(3, 1)))
        self.M1 = np.vstack((M, [0,0,0,1]))
    
        # Knee
        M = np.hstack((M_r, self.q_2.reshape(3, 1)))
        self.M2 = np.vstack((M, [0,0,0,1]))

        # End Effector
        M = np.hstack((np.eye(3), self.q_3.reshape(3, 1)))
        self.M3 = np.vstack((M, [0,0,0,1]))


    def forward(self, theta, joint_idx=3):
        theta_0, theta_1, theta_2 = theta[0], theta[1], theta[2]
        S0_exp = expm(self.S0 * theta_0)
        S1_exp = expm(self.S1 * theta_1)
        S2_exp = expm(self.S2 * theta_2)

        if joint_idx == 0:
            return self.M0 # Shoulder
        elif joint_idx == 1:
            return S0_exp @ self.M1 # Hip
        elif joint_idx == 2:
            return S0_exp @ S1_exp @ self.M2 # Knee
        else:
            return S0_exp @ S1_exp @ S2_exp @ self.M3 # Foot     

    def jacobian(self, theta):
        theta_0, theta_1, theta_2 = theta[0], theta[1], theta[2]
        slist = np.hstack((self.s0,self.s1,self.s2))
        thetalist = np.array([theta_0, theta_1, theta_2])

        return JacobianSpace(slist,thetalist)
        
class Robot:
    def __init__(self):
        params = {}
        
        params["d_fh"] = 0.747
        params["d_lr"] = 0.414
        params["l0"] = 0.08
        params["l1"] = 0.35
        params["l2"] = 0.346
        params["l3"] = 0.02

        self.params = params

        self.front_right = Leg(True, True, params)
        self.front_left = Leg(False, True, params)
        self.back_right = Leg(True, False, params)
        self.back_left = Leg(False, False, params)

        self.fr_jacobian = []
        self.fl_jacobian = []
        self.br_jacobian = []
        self.bl_jacobian = []

        self.joint_sigma = np.diag([1, 1, 1])*5e-2
        self.fk_prior_covariance = np.diag([1]*6)*5e-2

        self.contact_sigma = np.array([1e10, 1e10, 1e10, 1e-7, 1e-7, 1e-7])

    # joint_state 4x3, order is front_right, front_left, back_right, back_left
    def forward(self, joint_states: np.array, joint_idx = 3):
        T_front_right = self.front_right.forward(joint_states[0], joint_idx)        
        T_front_left = self.front_left.forward(joint_states[1], joint_idx)        
        T_back_right = self.back_right.forward(joint_states[2], joint_idx)     
        T_back_left = self.back_left.forward(joint_states[3], joint_idx)

        return T_front_right, T_front_left, T_back_right, T_back_left
    
    def jacobian(self, joint_states: np.array):
        self.fr_jacobian = self.front_right.jacobian(joint_states[0])
        self.fl_jacobian = self.front_left.jacobian(joint_states[1])
        self.br_jacobian = self.back_right.jacobian(joint_states[2])
        self.bl_jacobian = self.back_left.jacobian(joint_states[3])

        return self.fr_jacobian, self.fl_jacobian, self.br_jacobian, self.bl_jacobian

    def cross_jacobian(self, old_idx, new_idx, joint_states):
        
        legs = [self.front_right, self.front_left, self.back_right, self.back_left]
        
        old_leg = legs[old_idx]
        new_leg = legs[new_idx]

        omega_0 = old_leg.omega_2
        omega_1 = old_leg.omega_1
        omega_2 = old_leg.omega_0
        omega_3 = new_leg.omega_0
        omega_4 = new_leg.omega_1
        omega_5 = new_leg.omega_2

        l0 = self.params["l0"]
        l1 = self.params["l1"]
        l2 = self.params["l2"] + self.params["l3"]

        q_0 = np.array([0, 0, l2])
        q_1 = np.array([0, 0, l1 + l2])
        q_2 = np.array([0, 0, l0 + l1 + l2])
        d_fh = self.params["d_fh"]
        d_lr = self.params["d_lr"]

        old_front = old_idx < 2
        new_front = new_idx < 2
        
        if old_front == new_front:
            dx = 0
        elif old_front:
            dx = -d_fh
        else:
            dx = d_fh

        old_right = old_idx % 2 == 0
        new_right = old_idx % 2 == 0

        if old_right == new_right:
            dy = 0
        elif old_right:
            dy = -d_lr
        else:
            dy = d_lr
        
        q_3 = np.array([dx, dy, l0 + l1 + l2])
        q_4 = np.array([dx, dy, l1 + l2])
        q_5 = np.array([dx, dy, l2])

        # hack around intellisense nonsense
        def do_cross(a,b) -> np.array:
            return np.cross(a,b)


        omegas = [omega_0, omega_1, omega_2, omega_3, omega_4, omega_5]
        qs = [q_0, q_1, q_2, q_3, q_4, q_5]
        s_list = []
        for i in range(6):
            omega = omegas[i]
            q = qs[i]

            v = -do_cross(omega, q)

            s = np.vstack((omega.reshape(-1, 1), v.reshape(-1, 1)))
            s_list.append(s)

        s_list = np.hstack(s_list)
        old_joints = joint_states[old_idx]
        new_joints = joint_states[new_idx]
        theta_list = np.array([old_joints[2], old_joints[1], old_joints[0],
                               new_joints[0], new_joints[1], new_joints[2]])

        return JacobianSpace(s_list, theta_list)

    def FK_model(self, pose_wb, pose_wc, jacobian):
        Hbc = np.linalg.inv(pose_wb) @ pose_wc
        residual_error = logm(np.linalg.inv(pose_wc) @ pose_wb @ Hbc)        
        Sigma_F = jacobian @ self.joint_sigma @ np.transpose(jacobian)