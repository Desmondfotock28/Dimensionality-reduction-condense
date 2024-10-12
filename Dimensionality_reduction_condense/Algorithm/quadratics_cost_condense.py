import casadi as csd
import numpy as np


class Quadratic_stage_cost_model:
    def __init__(self, model, opt_params) -> None:
        self.model = model
        self.obs_dim = model.observation_space.shape[0]
        self.action_dim = model.action_space.shape[0]
        self.opt_params = opt_params 
        self.x0 = opt_params["x_0"]
        self.xSS = opt_params["x_SS"]
        self.N  = opt_params["horizon"]
        self.Q = opt_params["Q"]
        self.R = opt_params["R"]
        self.S = opt_params["S"]
        self.stage_cost = self.quadratic_stage_cost

        # Dynamics function from the model
        self.model_dyn = self.model_dynamics_init()
        
        self.cost_parameterization()

    def model_dynamics_init(self):
        model_dyn = self.model.get_model()
        return model_dyn

    # Defind condense vector 

    def g_vec(self, x0 , U , N ):
        
        g=[x0]
        x_k = x0
        for i in range(1,N):
            x_k = self.model_dyn(x_k, U[:,i-1])
            g.append(x_k)
        x_N = self.model_dyn(x_k, U[:,N-1] )
        g.append(x_N)
        return g
    
    def compute_block_matrix(self, Q, R, N):
        
        # Construct the block diagonal matrix Q
        Q_blocks = [Q] * N + [Q]

        Q_d = np.block([[Q_blocks[i] if i == j else np.zeros_like(Q) for j in range(N+1)] for i in range(N+1)])
        
        # Construct the block diagonal matrix R
        R_d = np.block([[R if i == j else np.zeros_like(R) for j in range(N)] for i in range(N)])

        return  Q_d ,  R_d
    
    def quadratic_stage_cost(self, state, action_vec):
        s =state 
        U = action_vec
        Q_d ,  R_d = self.compute_block_matrix(self.Q, self.R, self.N) 
        G = self.g_vec(s , U , self.N )
        G = csd.vertcat(*G)
        U = csd.vertcat(csd.reshape(U, -1,1))
        cost = G.T @ Q_d @ G + U.T @ R_d@ U 
        return cost
    

    def cost_parameterization(self):
        
        """
        Create a parameterization of the MPC cost function.
        Initialize the parameterization with quadratic stage cost and
        euclidean_dist terminal cost. For each cost type in cost_defn,
        corresponding attributes are created/updated .

        Returns
        sref : Reference state
        Q : State weight matrix.
        R : Action weight matrix.
        S: Terminal state weight matrix.
        
        """

        # Create Casadi functions for the stage and terminal costs
        state = csd.SX.sym("state", self.obs_dim)
        act =   csd.SX.sym("act", self.action_dim, self.N)

        stage_cost_expr = self.quadratic_stage_cost(state, act)
    
        self.stage_cost_fn = csd.Function("stage_cost_fn", [state, act], [stage_cost_expr])
     
