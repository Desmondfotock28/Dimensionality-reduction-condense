
import numpy as np
import casadi as csd
from quadratics_cost_condense import  Quadratic_stage_cost_model


class MPCformulation:
    """
    Class to setup MPC optimization problems in Casadi
    ...

    Attributes
    ----------
    env : Environment

    obs_dim, action_dim : int
        Dimensions of the state and action spaces

    N : int
        Prediction horizon

    model_defn : {"statespace_fromenv", "casadi_fn_obj_fromenv"}
        Source of the model

    U : symbolic (casadi)
        Input variables of optimization

    Opt_Vars : [U]

    Pf : symbolic (casadi)
        Optimization parameters [x0,xSS]

    model_dyn : casadi function
        Dynamics equation, A*x + B*u, if state space from env is used

    stage_cost : method
        Predefined method, such as 'quadratic_cost'
    """
    def __init__(self, model, opt_params):  
        self.N = opt_params["horizon"]
        self.x0 = opt_params["x_0"]
        self.opt_params = opt_params
        ## Symbolic variables for optimization problem
        self.model = model
        self.obs_dim = model.observation_space.shape[0]
        self.action_dim = model.action_space.shape[0]
        self.U = csd.SX.sym("U", self.action_dim, self.N)
        
        self.Opt_Vars = csd.vertcat(
            csd.reshape(self.U, -1, 1)  
        )
        # Symbolic parameters for optimization problem
        self.nPf = 2*self.obs_dim 
        self.Pf = csd.SX.sym(
            "Pf", self.nPf
        )  # [Initial state x0, steady_state] ("fixed" params)

        # Create an instance of QuadraticStageCostModel
        self.cost_model = Quadratic_stage_cost_model(model, opt_params)
        
         # Optimization formulation setup
        self.stage_cost_fn = self.cost_model.stage_cost_fn

        # Dynamics function from the model
        self.model_dyn = self.model_dynamics_init()
        self.Pi_opt_formulation()
    
    def model_dynamics_init(self):
        model_dyn = self.model.get_model()
        return model_dyn

    
    def objective_cost(self):
        
        J = self.stage_cost_fn(self.Pf[:self.obs_dim], self.U)
        return J
    

    def inequality_constraints(self):

        U = csd.vertcat(csd.reshape(self.U, -1, 1))
        G = self.cost_model.g_vec(self.Pf[:self.obs_dim], self.U ,self.N)
        G = csd.vertcat(*G)
        # Constraint list 
        hu = []   # Box constraints on active inputs
        hx = []   # Box constraints on states
        # State bounds
        lbx = [self.model.observation_space.low[:, None]]*(self.N + 1) 
        ubx = [self.model.observation_space.high[:, None]]*(self.N + 1)
        lbx = csd.vertcat(*lbx)
        ubx = csd.vertcat (*ubx)
        lbu = [self.model.action_space.low[:, None]]*self.N
        ubu = [self.model.action_space.high[:, None]]*self.N
        ubu = csd.vertcat(*ubu)
        lbu = csd.vertcat (*lbu)

        hx.append(lbx - G)
        hx.append(G - ubx)
        hu.append(lbu - U)
        hu.append(U - ubu)
    
        return  hu, hx
    
    
    def Pi_opt_formulation(self):
        """
        Formulate optimization cost and associated constraints
        - Cost uses stage and terminal cost functions, along with penalties

        """
        # Objective cost
        J = self.objective_cost()

        # Constraints for casadi and limits
        hu, hx = self.inequality_constraints()
        Hu = csd.vertcat(*hu)
        Hx = csd.vertcat(*hx)
    
        G_vcsd = csd.vertcat(*hx,*hu)
        lbg = [-np.inf] * (Hx.shape[0] + Hu.shape[0])
        ubg = [0] * (Hx.shape[0] + Hu.shape[0])
        self.lbg_vcsd = csd.vertcat(*lbg)
        self.ubg_vcsd = csd.vertcat(*ubg)
    
     # NLP Problem for value function and policy approximation
        opts_setting = {
            "ipopt.max_iter": 500,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_tol": 1e-6,
            "ipopt.acceptable_obj_change_tol": 1e-6,
        }
   
        
    # NLP Problem for value function and policy approximation
        vnlp_prob = {
            "f": J,
            "x": self.Opt_Vars,
            "p": csd.vertcat(self.Pf),
            "g": G_vcsd,
        }

    # Create the NLP solver instance 
        self.pisolver = csd.nlpsol("vsolver", "ipopt", vnlp_prob,opts_setting)