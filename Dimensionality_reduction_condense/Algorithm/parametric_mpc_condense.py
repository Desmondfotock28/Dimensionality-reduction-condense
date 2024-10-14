import numpy as np
import cvxpy as cvx
import casadi as csd
from parametric_mpc_formulation_condense import ParamMPCformulation
from nominal_mpc_condense import NominalMPC
from scipy.linalg import null_space
import autograd.numpy as anp
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
from pymanopt.manifolds import Stiefel


class MPCfunapprox(ParamMPCformulation):
    def __init__(self,model, agent_params,opt_params):
        # Parameters
        # Actor initilizaiton
        super().__init__(model, opt_params)
        self.obs_dim = self.model.observation_space.shape[0]
        self.action_dim = self.model.action_space.shape[0]
        self.x0 = opt_params["x_0"]
        self.xSS = opt_params["x_SS"]
        self.N  = opt_params["horizon"]
        self.opt_params = opt_params 
        self.agent_params = agent_params
         # Store opt_params as a class attribute
        # Parameter values
        self.T1_0 = agent_params["T1"]
        self.T2_0 = agent_params["T2"]
        self.w_0  = agent_params ["w"]
        self.u0 = (1/2) * (self.model.action_space.high[:, None] - self.model.action_space.low[:, None])

        #initialise learnable parameter
        self.Pf  = csd.vertcat( 
            csd.reshape(self.x0, -1, 1), 
            csd.reshape(self.xSS, -1, 1), 
            csd.reshape(self.u0, -1, 1),
            csd.reshape(self.w_0, -1, 1),
            csd.reshape(self.T2_0, -1, 1) 
        ) 

        self.P_learn  = csd.vertcat( 
            csd.reshape(self.T1_0, -1, 1),   
        ) 

    def reset(self, obs):
        obs = self.model.get_obs(obs)
        self.X0 =self.model.get_initial_guess(self.u0, self.Pf, self.P_learn, self.N , self.nv)
        
    def P(self, obs):
        obs = self.model.get_obs(obs)
        self.opt_params['x_0']=obs
        nominal_mpc = NominalMPC(self.model, opt_params=self.opt_params)
        u, usol = nominal_mpc.run_open_loop_mpc()
        return u, usol


    def act_forward(self, obs, soln=None, Pf_val=None, P_learn=None, mode="eval"):
        
        # Forward policy function evaluation for argmin action 
        # Solver params
        
        obs = self.model.get_obs(obs)
        Pf_val = Pf_val if Pf_val is not None else self.Pf
        Pf_val[:self.obs_dim]=obs
        P_learn = P_learn if P_learn is not None else self.P_learn
        X0 = soln["x"].full() if soln is not None else self.X0
        
        # PS: X0 is reinitialized from apriori soln, usually in update routine
        
        soln = self.pisolver(
            x0=X0,
            p=np.concatenate([Pf_val, P_learn])[:, 0],
            lbg=self.lbg_vcsd,
            ubg=self.ubg_vcsd,
        )
        fl = self.pisolver.stats()
       # if not fl["success"]:
            #print("OCP Solver Unsuccessful")
           
        
        opt_var = soln["x"].full()
        Vsol = np.array(opt_var[:self.nv]).reshape((self.nv,1))
        musol = np.array(opt_var[self.nv:]).reshape((1,1)) 
        self.T1 = self.P_learn
        
        self.T1 = csd.reshape(self.T1,self.N*self.action_dim,self.nv )
        self.T2 = self.Pf[2*self.obs_dim + self.action_dim + (self.N * self.action_dim - self.nv): ]
        self.T2 = csd.reshape(self.T2,self.N*self.action_dim,(self.N *self.action_dim - self.nv))
        self.w  = self.Pf[2*self.obs_dim + self.action_dim :(self.N * self.action_dim - self.nv)+ 2*self.obs_dim + self.action_dim]
        self.w  = csd.reshape(self.w, (self.N * self.action_dim - self.nv ), 1)

        #Reconstruct optimal control action
        action = self.T1@Vsol + musol*self.T2 @self.w
        act =  np.array(action).reshape((self.N , self.action_dim))
        act0 = act[0, :]
        #act = act0.clip(self.model.action_space.low, self.model.action_space.high)
       
        self.X0 = self.update_X0(opt_var)
        info = {
            "optimal": fl["success"],
            "soln": soln.copy(),
            "pf": np.array(Pf_val).copy(),
            "p": np.array(P_learn).copy(),
        }
        return act0, action, info    
    
    def update_X0(self, opt_var):
        # "warm starting" for udapting the initial guess given to the solver
        # based on the last solution
        soln = opt_var.copy()
        Vsol = np.array(soln[:self.nv])
        musol = np.array(soln[self.nv:]).reshape((1,1))
         
        if Vsol.shape ==1:
            v_st_0 = Vsol
        else: 
             v_st_0 = np.vstack([Vsol[1:], Vsol[-1]])
        
        X0 = np.concatenate(( v_st_0, musol ), axis=0)
    
        return X0
    
    def dPidP(self, soln, Pf_val, P_learn, optimal=True):
        # Sensitivity of policy output with respect to learnable param
        # i.e. gradient of action wrt to param_val
        x = soln["x"].full()
        lam_g = soln["lam_g"].full()
        z = np.concatenate((x, lam_g), axis=0)
        Pf_val = Pf_val.copy()
        P_val = P_learn.copy()

        if optimal:
            jacob_act = self.dPi(z, Pf_val, P_val).full()
        else:
            jacob_act = np.zeros((self.N*self.action_dim, self.nP))
        return jacob_act
    
    def V_value(self, obs, soln=None, Pf_val=None, P_learn=None, mode="train"):
        # Forward policy function evaluation for argmin action
        # Solver params
        obs = self.model.get_obs(obs)
        Pf_val = Pf_val if Pf_val is not None else self.Pf
        Pf_val[:self.obs_dim]= obs
        P_learn = P_learn if P_learn is not None else self.P_learn

        X0 = soln["x"].full() if soln is not None else self.model.get_initial_guess(self.u0, self.Pf, self.P_learn, self.N , self.nv)
        # PS: X0 is reinitialized from apriori soln, usually in update routine
        
        soln = self.pisolver(
            x0=X0,
            p=np.concatenate([Pf_val, P_learn])[:, 0],
            lbg=self.lbg_vcsd,
            ubg=self.ubg_vcsd,
        )
        fl = self.pisolver.stats()
        if not fl["success"]:
            print("OCP Solver Unsuccessful")
            print(obs)
            
        

        v_mpc = soln["f"].full()[0, 0]
        info = {
            "optimal": fl["success"],
            "soln": soln.copy(),
            "pf": np.array(Pf_val).copy(),
            "p": np.array(P_learn).copy(),
        }
        return v_mpc, info
    

    def dVdP(self, soln, Pf_val, P_learn, optimal=True):
        # Gradient of value fn v wrt lernable param
        # state/obs, action, act_wt need to be from vsoln (garbage in garbage out)
        x = soln["x"].full()
        lam_g = soln["lam_g"].full()
        pf_val = Pf_val.copy()
        p_val = P_learn.copy()
        if optimal:
            _, _, dLdP = self.dLagV(x, lam_g, pf_val[:, 0], p_val[:,0])
            dLdP = dLdP.full()
        else:
            dLdP = np.zeros((1, self.nP))
        return dLdP
    

    def Q_value(self, obs, action, soln=None, Pf_val=None, P_learn=None):
        # Action-value function evaluation
        obs = self.model.get_obs(obs)
        Pf_val = Pf_val.copy() if Pf_val is not None else self.Pf
        Pf_val[:self.obs_dim]= obs
       
        Pf_val[2*self.obs_dim:2*self.obs_dim + self.action_dim] = action[:, None]
       
        P_learn = P_learn.copy() if P_learn is not None else self.P_learn
        X0 = (
            soln["x"].full()
            if soln is not None
            else self.model.get_initial_guess(action, Pf_val, self.P_learn, self.N , self.nv)
        )
       
        #print(action[:, None])
        #X0[: self.action_dim, :] = action[:, None]  
        #X0 = self.model.get_initial_guess(obs, action, Pf_val, self.P_learn, self.N , self.nv)
        qsoln = self.qsolver(
            x0=X0,
            p=np.concatenate([Pf_val, P_learn])[:, 0],
            lbg=self.lbg_qcsd,
            ubg=self.ubg_qcsd,
        )
        fl = self.qsolver.stats()
        if not fl["success"]:
            print("OCP Solver Unsuccessful")
            print(obs)
            print(action)
         

        q_mpc = qsoln["f"].full()[0, 0]
        info = {
            "optimal": fl["success"],
            "soln": qsoln.copy(),
            "pf": np.array(Pf_val).copy(),
            "p": np.array(P_learn).copy(),
        }
        return q_mpc, info
    
    def dQdP(self, soln, Pf_val, P_learn, optimal=True):
        # Gradient of action-value fn Q wrt learnable param
        # state/obs, action, act_wt need to be from qsoln (garbage in garbage out)
        x = soln["x"].full()
        lam_g = soln["lam_g"].full()
        pf_val = Pf_val.copy()
        p_val = P_learn.copy()
        if optimal:
            _, _, dLdP = self.dLagQ(x, lam_g, pf_val[:, 0], p_val[:,0])
            dLdP = dLdP.full()
        else:
            dLdP = np.zeros((1, self.nP))
        return dLdP
    

    def param_update(self, dJ, param_val=None, constrained_updates=True):
        # Param update scheme
        lr = self.learning_params["lr"]
        param_val = param_val if param_val is not None else self.P_learn
        if constrained_updates:
       
            #self.P_learn = self.Stiefel_param_update(dJ , param_val, lr)
            self.P_learn = self.constraint_param_update(dJ, param_val)
            self.P_learn = self.gramm_schmidt()
            self.compute_null_space()
        else:
            dP = -self.lr[0] * dJ    #need to check shape 
            self.P_learn += dP.clip(
                -self.tr[0], self.tr[0]
            )
            self.P_learn = self.gramm_schmidt()
            self.compute_null_space()
    
    def constraint_param_opt(self, lr, tr):
        # SDP for param update to ensure stable MPC formulation
        # l1, l2 = 0.000, 0.000
        self.dP_th = cvx.Variable((self.nP, 1))
        self.dJ_th = cvx.Parameter((self.nP, 1))
        self.P_th = cvx.Parameter((self.nP, 1))
        P_th_next = self.P_th + self.dP_th

        J_th = 0.5 * cvx.sum_squares(self.dP_th) + lr * self.dJ_th.T @ self.dP_th
        # J_up += l1 * cvx.norm(P_cost_next, 1) + l2 * cvx.norm(P_cost_next, 2)
        constraint = [self.dP_th <= tr, self.dP_th >= -tr]
       
        self.update_step = cvx.Problem(cvx.Minimize(J_th), constraint)
        #
        #self.update_step = cvx.Problem(cvx.Minimize(J_th)))
    

    def Stiefel_param_update(self, dJ, p_val, lr ):
        
        """
        SDP on the Stiefel Manifold param update to ensure stable MPC formulation

        :param p_val: Initial orthogonal matrix (shape n x k).
        :param dJ: Expected value of the gradient of the parametrized state-action value function.
        :param lr: Learning rate.
        :return: Optimized matrix theta, or None if optimization fails.
        """    
        try:
            P_up = np.array(p_val).copy()
            Jac   = np.array(dJ).copy()
            P_up = P_up.reshape(self.N*self.action_dim , self.nv , order='F')
            Jac = Jac.reshape(self.N*self.action_dim , self.nv , order='F')
            n, k = P_up.shape  # Dimensions of the matrix
            print("start RL update scheme")
            # Define the Stiefel manifold
            manifold = Stiefel(n, k)
            @pymanopt.function.autograd(manifold)
            def cost(point):
                diff = point - P_up
                constraint_violation = 1000000*(np.sum(np.maximum(0, diff - 0.2)) + np.sum(np.maximum(0, -0.2 - diff)))
            # Define the cost function
                if k == 1:
                    cost_n = 0.5 * anp.linalg.norm(diff) ** 2 +  lr * anp.dot(dJ.T, diff) 
                else:
                    
                    cost_n = 0.5 * anp.linalg.norm(diff, 'fro') ** 2 + lr * anp.trace(Jac.T @ diff) 
                
                return cost_n  + constraint_violation
            

           # Set up the optimization problem on the Stiefel manifold
            problem = pymanopt.Problem(manifold=manifold, cost=cost)

            # Use the Steepest Descent optimizer with verbosity turned off
            optimizer = pymanopt.optimizers.SteepestDescent(verbosity=0)

            # Solve the optimization problem
            result = optimizer.run(problem)

            result.point = csd.vertcat( csd.reshape(result.point, -1, 1))
            # Return the optimized matrix
            result.point = np.array(result.point)
            return result.point

        except Exception as e:
            # If any error occurs, print the error message and return None
            print("SDP solver for cost param update failed:", str(e))
            return p_val
    

    def constraint_param_update(self, dJ, p_val):
        # input  param update
        P_up = np.array(p_val).copy()
        self.dJ_th.value = np.array(dJ).copy()
        self.P_th.value = np.array(P_up).copy()
        try:
            self.update_step.solve()
        except:
            print("SDP solver for cost param update failed")

        if self.update_step.status == "optimal":
            P_up += self.dP_th.value.copy()
        else:
            print(f"Problem status: {self.update_step.status}")
        return P_up
    
    def _parse_agent_params(self, gamma, eps, learning_params, **kwargs):
        self.gamma = gamma
        self.eps = eps
        self.learning_params = learning_params
        
        # Handle additional parameters like T1, T2, w
        for key, value in kwargs.items():
            setattr(self, key, value)
    

    def compute_null_space(self,param_val=None ):
        # Param update scheme for T2 uaing nullspace of T1
        param_val = param_val if param_val is not None else self.P_learn
        matrix = np.array( param_val).reshape(self.N*self.action_dim , self.nv , order='F')
         # Handle the case where the matrix is n x 1
        if matrix.shape[1] < self.N*self.action_dim:
        # Find all vectors orthogonal to the given vector
        # We treat the vector as a column vector and find the null space of its transpose
            T2 = null_space(matrix.T)   
        else:
        # Use scipy's null_space function to compute the null space of the matrix
            T2 = null_space(matrix)
         # update the fixed parameter list 
        self.Pf[2*self.obs_dim + self.action_dim + (self.N * self.action_dim - self.nv):] = csd.vertcat( csd.reshape(T2, -1, 1))
    

    def gramm_schmidt(self,param_val=None):
    # Check the number of columns in the matrix
        param_val = param_val if param_val is not None else self.P_learn
        param_val = np.array(param_val).reshape(self.N*self.action_dim , self.nv , order='F')
        num_columns =  param_val.shape[1]
    
    # If the matrix has only one column, return the matrix itself
        if num_columns == 1:
            Q_up =csd.vertcat( csd.reshape(param_val, -1, 1))
            norm= csd.norm_2(Q_up)
            Q_up = Q_up/norm
           
    # If the matrix has more than one column, perform QR decomposition
        else:
            Q_up, _ = np.linalg.qr(param_val)
            Q_up  = csd.vertcat( csd.reshape(Q_up, -1, 1))
        return Q_up