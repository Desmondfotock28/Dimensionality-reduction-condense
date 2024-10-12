import numpy as np
from nominal_mpc_formulation_condense import MPCformulation
from casadi import *


class NominalMPC(MPCformulation):
    def __init__(self, model, opt_params):
        super().__init__(model, opt_params)  # Initialize superclass
        self.x0 = opt_params["x_0"]
        self.xSS = opt_params["x_SS"]
        self.u0 = (1/2) * (self.model.action_space.high[:, None] - self.model.action_space.low[:, None])
        self.N = opt_params["horizon"]  # Ensure N is properly initialized
        self.nx =  model.observation_space.shape[0]
        self.nu =  model.action_space.shape[0]

    def run_open_loop_mpc(self):
        # Initial control inputs and state
        self.u_st_0 = np.tile(self.u0, (self.N, 1))
        args_p = np.array([self.x0] + [self.xSS])  # Ensure xSS is defined
        args_p = vertcat(*args_p)
        args_x0 = self.u_st_0.T.reshape(-1)
        
        # Solve the optimization problem
        sol = self.pisolver(x0=args_x0, p=args_p, lbg=self.lbg_vcsd, ubg=self.ubg_vcsd)
        usol = sol['x']
        # Extract the control inputs from the solution
        u = np.array(sol['x']).reshape((self.N, self.nu))
        # Extract predicted state
        #x_pred = np.array(sol['x'][:self.nx * (self.N + 1)]).reshape((self.N + 1, self.nx))

        return u, usol
