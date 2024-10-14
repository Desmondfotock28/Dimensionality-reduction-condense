import numpy as np
import casadi as csd
from quadratics_cost_condense import  Quadratic_stage_cost_model


class ParamMPCformulation:
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

    gamma : float
        Discount factor

    etau : float
        IPOPT tuning parameter (mu_target, mu_init)


    model_defn : {"statespace_fromenv", "casadi_fn_obj_fromenv"}
        Source of the model

    V : symbolic (casadi)
        Active Input variables of optimization

    mu : symbolc (casadi)
        penalty varaibles for inactive subspace input

    Opt_Vars : [V,mu]

    Pf : symbolic (casadi)
        Optimization parameters [x0, xss, u0, w , T2]

    P : symbolic (casadi)
        Learnable Optimization parameters [T1]

    model_dyn : casadi function
        Dynamics equation, A*x + B*u, if state space from env is used

    n_P : int
        Number of parameters to be learned

    stage_cost : method
        Predefined method, such as 'quadratic_cost'
    """
    def __init__(self, model, opt_params):
        self.N = opt_params["horizon"]
        self.nv = opt_params["n_v"]
        self.x0 = opt_params ["x_0"]
        self.xSS = opt_params ["x_SS"]
        self.etau = 1e-6
        
        self.model = model

        self.obs_dim = model.observation_space.shape[0]

        self.action_dim = model.action_space.shape[0]

        ## Symbolic variables for optimization problem
        self.V = csd.SX.sym("V", self.nv)

        # penalty  variables for inactive subspace w 
        self.mu = csd.SX.sym("mu", 1)

       
       # Symbolic parameters for optimization problem
        self.nPf = 2*self.obs_dim + self.action_dim + (self.N * self.action_dim - self.nv) + (self.N*self.action_dim)*(self.N *self.action_dim - self.nv) 
        self.Pf = csd.SX.sym(
            "Pf", self.nPf
        )  # [Initial state x0, steady state xSS,initial_input , inactive vector, inactive projection matrix] ("fixed" params)

        #learnable Parameter
        self.nP=(self.N*self.action_dim)*self.nv 
        # [active subspace T1]
    
        self.P_learn =  csd.SX.sym("P_learn", self.nP)
      

        #getting the parameter T1, T2, w
        self.T1 = self.P_learn
        self.T1 = csd.reshape(self.T1,self.N*self.action_dim,self.nv )
        self.T2 = self.Pf[2*self.obs_dim + self.action_dim + (self.N * self.action_dim - self.nv): ]
        self.T2 = csd.reshape(self.T2,self.N*self.action_dim,(self.N *self.action_dim - self.nv))
        self.w  = self.Pf[2*self.obs_dim + self.action_dim :(self.N * self.action_dim - self.nv)+ 2*self.obs_dim + self.action_dim]
        self.w  = csd.reshape(self.w, (self.N * self.action_dim - self.nv ), 1)
  
        #compute U 
        self.U = self.T1@self.V + self.mu*self.T2 @ self.w
        self.U = csd.reshape(self.U,self.action_dim,self.N)
        
        self.Opt_Vars = csd.vertcat(
            csd.reshape(self.V, -1, 1),
            csd.reshape(self.mu, -1, 1)
        )

          # Create an instance of QuadraticStageCostModel
        self.cost_model = Quadratic_stage_cost_model(model, opt_params)
        
         # Optimization formulation setup
        self.stage_cost_fn = self.cost_model.stage_cost_fn

        
        # Dynamics function from the model
        self.model_dyn = self.model_dynamics_init()
        #self.model_dyn = self.dynamics
        # Optimization formulation 
        self.Pi_opt_formulation()
        self.Q_opt_formulation()

     

    def model_dynamics_init(self):
        model_dyn = self.model.get_model()
        return model_dyn
    
    #defined objective function
    
    def objective_cost(self):

        J = self.stage_cost_fn(self.Pf[:self.obs_dim], self.U)

        return J

       
    #  cost computation 
    def compute_cost(self, x, U):
          
        J = self.stage_cost_fn(x, U)
        return J
    
    def equality_constraints(self):
        g = []  # Equality constraints initialization
        g.append(self.U[:, 0] - self.Pf[2*self.obs_dim :2*self.obs_dim + self.action_dim ])
        return  g
    
    def inequality_constraints(self):
      
        G = self.cost_model.g_vec(self.Pf[:self.obs_dim], self.U ,self.N)
        X =[]
        for i in range(len(G)):
            X.append(G[i][1:2])

        X = csd.vertcat(*X)
        U = csd.vertcat(csd.reshape(self.U, -1, 1))
        
        #G = csd.vertcat(*G)
        # Constraint list
        hmu = []  # constraints for mu 
        hu = []   # Box constraints on active inputs
        hx = []   # Box constraints on states

         # State bounds
        lbx = [self.model.observation_space.low[:, None][1:2]]*(self.N + 1) 
        ubx = [self.model.observation_space.high[:, None][1:2]]*(self.N + 1)
        lbx = csd.vertcat(*lbx)
        ubx = csd.vertcat (*ubx)
        lbu = [self.model.action_space.low[:, None]]*self.N
        ubu = [self.model.action_space.high[:, None]]*self.N
        ubu = csd.vertcat(*ubu)
        lbu = csd.vertcat (*lbu)

        lb_mu = 0.8*np.ones(1)
        ub_mu = 1.0*np.ones(1)
      

        hx.append(lbx - X)
        hx.append(X - ubx)
        hu.append(lbu - U)
        hu.append(U - ubu)
        #  mu constraints 
        hmu.append(lb_mu-self.mu)
        hmu.append(self.mu-ub_mu)
       
        return  hx, hu, hmu 
    
    def Pi_opt_formulation(self):
        """
        Formulate optimization cost and associated constraints
        - Cost uses stage and terminal cost functions, along with penalties

        """
        # Objective cost
        J = self.objective_cost()
    
        hx, hu, hmu = self.inequality_constraints()
        Hu = csd.vertcat(*hu)
        Hx = csd.vertcat(*hx)
        Hmu = csd.vertcat(*hmu)
        G_vcsd = csd.vertcat( *hx, *hu,  *hmu)
        lbg =  [-np.inf] * (Hu.shape[0] + Hx.shape[0] + Hmu.shape[0])
        ubg =  [0] * (Hu.shape[0] + Hx.shape[0] + Hmu.shape[0])
        self.lbg_vcsd = csd.vertcat(*lbg)
        self.ubg_vcsd = csd.vertcat(*ubg)

        # NLP Problem for value function and policy approximation
        opts_setting = {
            "ipopt.max_iter": 500,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.mu_target": self.etau,
            "ipopt.mu_init": self.etau,
            "ipopt.acceptable_tol": 1e-6,
            "ipopt.acceptable_obj_change_tol": 1e-6,
        }

        vnlp_prob = {
            "f": J,
            "x": self.Opt_Vars,
            "p": csd.vertcat(self.Pf, self.P_learn),
            "g": G_vcsd,
        }

        self.pisolver = csd.nlpsol("vsolver", "ipopt", vnlp_prob, opts_setting)
        (
            self.Rfun,
            self.dPi,      
            self.dLagV,
            self.dRdz,    
            self.dRdp,    
        ) = self.build_sensitivity_Pi(J, Hu, Hx, Hmu)
    
    def Q_opt_formulation(self):
        # Objective cost
        J = self.objective_cost()
        g = self.equality_constraints()
        hu, hx, hmu = self.inequality_constraints()
        G = csd.vertcat(*g)
        Hu = csd.vertcat(*hu)
        Hx = csd.vertcat(*hx)
        Hmu = csd.vertcat(*hmu)
        G_qcsd = csd.vertcat(*g, *hx, *hu,  *hmu)
        lbg = [0] * G.shape[0] + [-np.inf] * (Hu.shape[0] + Hx.shape[0] + Hmu.shape[0])
        ubg = [0] * G.shape[0] + [0] * (Hu.shape[0] + Hx.shape[0] + Hmu.shape[0])
        self.lbg_qcsd = csd.vertcat(*lbg)
        self.ubg_qcsd = csd.vertcat(*ubg)

        opts_setting = {
            "ipopt.max_iter": 500,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.mu_target": self.etau,
            "ipopt.mu_init": self.etau,
            "ipopt.acceptable_tol": 1e-6,
            "ipopt.acceptable_obj_change_tol": 1e-6,
        }
        qnlp_prob = {
            "f": J,
            "x": self.Opt_Vars,
            "p": csd.vertcat(self.Pf, self.P_learn),
            "g": G_qcsd,
        }
        self.qsolver = csd.nlpsol("qsolver", "ipopt", qnlp_prob, opts_setting)

        _, _, self.dLagQ, _, _ = self.build_sensitivity(J, G, Hu, Hx, Hmu)
    
    def build_block_matrix(self,b, n):
        """
        Build the block matrix b ⊗ I_n with dimensions (p*n) x n.

        Parameters
        ----------
        b : CasADi SX
        Vector of dimension p.
        n : int
        Dimension of the identity matrix I_n.

        Returns
        -------
       block_matrix : CasADi MX
        The resulting block matrix of size (p*n) x n.
        """
        p = b.shape[0]  # Number of elements in vector b

        # Create the identity matrix I_n
        I_n = csd.SX.eye(n)

       # Initialize an empty list to store the blocks
        blocks = []

       # Iterate over the elements in vector b
        for i in range(p):
        # Create the block b_i * I_n and stack it vertically
            block = b[i] * I_n
        # Append the block to the list
            blocks.append(block)
    
    # Concatenate the blocks vertically
        block_matrix = csd.vertcat(*blocks)

        return block_matrix

    def build_sensitivity(self, G, J, Hu, Hx, Hmu):
        """
        Computes the sensitivity functions for given cost and constraints in
        J, G, Hu, Hx, Hmu

        Parameters
        ----------
        J : Cost function

        G : Equality constraints {Could be None}

        Hu : Input Constraints (u = T1@V + mu*T2@w)

        Hx : State Constraints

        Hmu : penalty constraints


        Returns
        -------
        Rfun:  TYPE 
            DESCRIPTION.  Casadi function for KKT matrix
        dPi : TYPE
            DESCRIPTION.  Casadi function for nable_pi
        dLagfunc : TYPE
            DESCRIPTION.  Casadi function for nabla_lagrange
        f_dRdz : TYPE
            DESCRIPTION.  Casadi function for derivative of KKT matrix w.r.t. primal-dual variables
        f_dRdp : TYPE
            DESCRIPTION.  Casadi function for derivative of KKT matrix w.r.t. parameters

        """
        # Sensitivity
        # Vector of Lagrange multipliers for dynamics equality constraints
        lamb = csd.SX.sym("lambda", G.shape[0])
        # Vector of Lagrange multipliers for input inequality constraints
        mu_u = csd.SX.sym("muu", Hu.shape[0])
        # Vector of Lagrange multipliers for state inequality constraints
        mu_x = csd.SX.sym("mux", Hx.shape[0])
         # Vector of Lagrange multipliers for penalty inequality constraints
        mu_mu = csd.SX.sym("mus", Hmu.shape[0])
        # Vector of Lagrange multipliers
        mult = csd.vertcat(lamb, mu_u, mu_x, mu_mu)

        # Build Lagrangian
        Lag = (
            J
            + csd.transpose(lamb) @ G
            + csd.transpose(mu_u) @ Hu
            + csd.transpose(mu_x) @ Hx
            + csd.transpose(mu_mu) @ Hmu
        )
        Lagfunc = csd.Function("Lag", [self.Opt_Vars, mult, self.Pf, self.P_learn], [Lag])

        # Generate sensitivity of the Lagrangian
        dLagfunc = Lagfunc.factory(
            "dLagfunc",
            ["i0", "i1", "i2", "i3"],
            ["jac:o0:i0", "jac:o0:i2", "jac:o0:i3"],  
            #  "jac:o0:i0" compute the partial derivative with respect to optimisation parameters (V,mu)
            #  "jac:o0:i2" compute the partial derivative with respect to the fixed parameters (x0, xSS, u0 ,w,T2)
            #  "jac:o0:i2" compute the partial derivative with respect to the learnable  parameters (T1)
           
        )
        dLdw, dLdPf, dLdP_learn = dLagfunc(self.Opt_Vars, mult, self.Pf, self.P_learn)

        # Build KKT matrix
        R_kkt = csd.vertcat(
            csd.transpose(dLdw),
            G,
            mu_u * Hu + self.etau,
            mu_x * Hx + self.etau,
            mu_mu * Hmu + self.etau,
        )
      

        # z contains all variables of the lagrangian
        z = csd.vertcat(self.Opt_Vars, lamb, mu_u, mu_x, mu_mu)
   
        # Generate sensitivity of the KKT matrix
        Rfun = csd.Function("Rfun", [z, self.Pf, self.P_learn], [R_kkt])
        dR_sensfunc = Rfun.factory("dR", ["i0", "i1", "i2"], ["jac:o0:i0", "jac:o0:i2"])
        [dRdz, dRdP_learn] = dR_sensfunc(z, self.Pf, self.P_learn)
        
        skip_code = True  # Set to True to skip the code, False to execute it

        if not skip_code: 

        # Generate sensitivity of the optimal solution
           dRdz_inv = csd.inv(dRdz)
           dzdP_learn = -dRdz_inv[(self.N+1)*self.obs_dim:(self.N+1)*self.obs_dim + self.nv, :] @ dRdP_learn 
           dmudP_learn = -dRdz_inv[(self.N+1)*self.obs_dim + self.nv :(self.N+1)*self.obs_dim + self.nv + 1, :]@ dRdP_learn 
    
        # computing dudP
        #  Given  u = T1@V + mu*T2@w 
        #  dudP = d(T1@V)/dP + d(mu*T2@w)/dP
        #  d(T1@V)/dP = (dT1/dP)@(V⊗In) + T1@dV/dP
        #  d(mu*T2@w)/dP = ((dT2/dP)@(mu⊗In) + T2@dmu/dP)@(w⊗In) + muT2@dw/dP
           T1 = csd.reshape(self.T1,-1,1)
           T2 =csd.reshape(self.T2,-1,1)
           w  = csd.reshape(self.w,-1,1)
      
           func = csd.Function("fun", [self.P_learn, self.Pf], [T1 ,T2, w])
           dfunc = func.factory("dfunc", ["i0","i1"], ["jac:o0:i0","jac:o1:i0","jac:o2:i0"])
           [dT1dP, dT2dP , dwdP] =  dfunc(self.P_learn,self.Pf)

           dT1dP = csd.reshape(dT1dP, self.N*self.action_dim,self.nP*self.nv)

           block_matrix_v = self.build_block_matrix(self.V, self.nP)
           block_matrix_mu = self.build_block_matrix(self.mu, self.nP)
           block_matrix_w_mu = self.build_block_matrix(self.mu*self.w, self.nP)
           dT2dP = csd.reshape(dT2dP, self.N*self.action_dim,self.nP*(self.N *self.action_dim - self.nv))
        #implementing this :  #  d(T1@V)/dP = (dT1/dP)@(V⊗In) + T1@dV/dP
           dzdP_learn1 = dT1dP@block_matrix_v + self.T1 @ dzdP_learn 
        
        #implementing this :d(mu*T2@w)/dP = ((dT2/dP)@(mu⊗In) + T2@dmu/dP)@(w⊗In) + muT2@dw/dP
           dzdP_learn2 = ( dT2dP@block_matrix_w_mu )+ self.T2@(dwdP@block_matrix_mu + self.w@ dmudP_learn)
           dzdP_learn = dzdP_learn1 +  dzdP_learn2
           dzdP_learn = dzdP_learn[: self.action_dim, :]
   
        # Sensitivity function for the policy
           dPi = csd.Function("dPi", [z, self.Pf, self.P_learn], [dzdP_learn])
        
        dPi  = None
        # Sensitivity function of the KKT matrix with respect to the solution
        f_dRdz = csd.Function("dRdz", [z, self.Pf, self.P_learn], [dRdz])

        
        # Sensitivity function of the KKT matrix with respect to the parameters
        f_dRdp = csd.Function("dRdP", [z, self.Pf, self.P_learn], [dRdP_learn])

        return Rfun, dPi, dLagfunc, f_dRdz, f_dRdp
    
    def build_sensitivity_Pi(self, J, Hu, Hx, Hmu):
        """
        Computes the sensitivity functions for given cost and constraints in
        J, G, Hu, Hx, Hmu

        Parameters
        ----------
        J : Cost function

        G : Equality constraints {Could be None}

        Hu : Input Constraints (u = T1@V + mu*T2@w)

        Hx : State Constraints

        Hmu : penalty constraints


        Returns
        -------
        Rfun:  TYPE 
            DESCRIPTION.  Casadi function for KKT matrix
        dPi : TYPE
            DESCRIPTION.  Casadi function for nable_pi
        dLagfunc : TYPE
            DESCRIPTION.  Casadi function for nabla_lagrange
        f_dRdz : TYPE
            DESCRIPTION.  Casadi function for derivative of KKT matrix w.r.t. primal-dual variables
        f_dRdp : TYPE
            DESCRIPTION.  Casadi function for derivative of KKT matrix w.r.t. parameters

        """
        # Sensitivity
        # Vector of Lagrange multipliers for input inequality constraints
        mu_u = csd.SX.sym("muu", Hu.shape[0])
        # Vector of Lagrange multipliers for state inequality constraints
        mu_x = csd.SX.sym("mux", Hx.shape[0])
         # Vector of Lagrange multipliers for penalty inequality constraints
        mu_mu = csd.SX.sym("mus", Hmu.shape[0])
        # Vector of Lagrange multipliers
        mult = csd.vertcat( mu_u, mu_x, mu_mu)

        # Build Lagrangian
        Lag = (
            J
            + csd.transpose(mu_u) @ Hu
            + csd.transpose(mu_x) @ Hx
            + csd.transpose(mu_mu) @ Hmu
        )
        Lagfunc = csd.Function("Lag", [self.Opt_Vars, mult, self.Pf, self.P_learn], [Lag])

        # Generate sensitivity of the Lagrangian
        dLagfunc = Lagfunc.factory(
            "dLagfunc",
            ["i0", "i1", "i2", "i3"],
            ["jac:o0:i0", "jac:o0:i2", "jac:o0:i3"],  
            #  "jac:o0:i0" compute the partial derivative with respect to optimisation parameters (V,mu)
            #  "jac:o0:i2" compute the partial derivative with respect to the fixed parameters (x0, xSS, u0 ,w,T2)
            #  "jac:o0:i2" compute the partial derivative with respect to the learnable  parameters (T1)
           
        )
        dLdw, dLdPf, dLdP_learn = dLagfunc(self.Opt_Vars, mult, self.Pf, self.P_learn)

        # Build KKT matrix
        R_kkt = csd.vertcat(
            csd.transpose(dLdw),
            mu_u * Hu + self.etau,
            mu_x * Hx + self.etau,
            mu_mu * Hmu + self.etau,
        )
      

        # z contains all variables of the lagrangian
        z = csd.vertcat(self.Opt_Vars, mu_u, mu_x, mu_mu)
   
        # Generate sensitivity of the KKT matrix
        Rfun = csd.Function("Rfun", [z, self.Pf, self.P_learn], [R_kkt])
        dR_sensfunc = Rfun.factory("dR", ["i0", "i1", "i2"], ["jac:o0:i0", "jac:o0:i2"])
        [dRdz, dRdP_learn] = dR_sensfunc(z, self.Pf, self.P_learn)
        
        skip_code = True  # Set to True to skip the code, False to execute it

        if not skip_code: 

        # Generate sensitivity of the optimal solution
           dRdz_inv = csd.inv(dRdz)
           dzdP_learn = -dRdz_inv[(self.N+1)*self.obs_dim:(self.N+1)*self.obs_dim + self.nv, :] @ dRdP_learn 
           dmudP_learn = -dRdz_inv[(self.N+1)*self.obs_dim + self.nv :(self.N+1)*self.obs_dim + self.nv + 1, :]@ dRdP_learn 
    
        # computing dudP
        #  Given  u = T1@V + mu*T2@w 
        #  dudP = d(T1@V)/dP + d(mu*T2@w)/dP
        #  d(T1@V)/dP = (dT1/dP)@(V⊗In) + T1@dV/dP
        #  d(mu*T2@w)/dP = ((dT2/dP)@(mu⊗In) + T2@dmu/dP)@(w⊗In) + muT2@dw/dP
           T1 = csd.reshape(self.T1,-1,1)
           T2 =csd.reshape(self.T2,-1,1)
           w  = csd.reshape(self.w,-1,1)
      
           func = csd.Function("fun", [self.P_learn, self.Pf], [T1 ,T2, w])
           dfunc = func.factory("dfunc", ["i0","i1"], ["jac:o0:i0","jac:o1:i0","jac:o2:i0"])
           [dT1dP, dT2dP , dwdP] =  dfunc(self.P_learn,self.Pf)

           dT1dP = csd.reshape(dT1dP, self.N*self.action_dim,self.nP*self.nv)

           block_matrix_v = self.build_block_matrix(self.V, self.nP)
           block_matrix_mu = self.build_block_matrix(self.mu, self.nP)
           block_matrix_w_mu = self.build_block_matrix(self.mu*self.w, self.nP)
           dT2dP = csd.reshape(dT2dP, self.N*self.action_dim,self.nP*(self.N *self.action_dim - self.nv))
        #implementing this :  #  d(T1@V)/dP = (dT1/dP)@(V⊗In) + T1@dV/dP
           dzdP_learn1 = dT1dP@block_matrix_v + self.T1 @ dzdP_learn 
        
        #implementing this :d(mu*T2@w)/dP = ((dT2/dP)@(mu⊗In) + T2@dmu/dP)@(w⊗In) + muT2@dw/dP
           dzdP_learn2 = ( dT2dP@block_matrix_w_mu )+ self.T2@(dwdP@block_matrix_mu + self.w@ dmudP_learn)
           dzdP_learn = dzdP_learn1 +  dzdP_learn2
           dzdP_learn = dzdP_learn[: self.action_dim, :]
   
        # Sensitivity function for the policy
           dPi = csd.Function("dPi", [z, self.Pf, self.P_learn], [dzdP_learn])
        
        dPi  = None
        # Sensitivity function of the KKT matrix with respect to the solution
        f_dRdz = csd.Function("dRdz", [z, self.Pf, self.P_learn], [dRdz])

        
        # Sensitivity function of the KKT matrix with respect to the parameters
        f_dRdp = csd.Function("dRdP", [z, self.Pf, self.P_learn], [dRdP_learn])

        return Rfun, dPi, dLagfunc, f_dRdz, f_dRdp

