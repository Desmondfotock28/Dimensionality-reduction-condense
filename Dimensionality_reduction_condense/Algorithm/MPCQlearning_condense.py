import casadi as csd 
import numpy as np
from scipy.linalg import expm
from scipy.linalg import null_space


class MPCQlearning:
    def __init__(self, mpc, learning_params,exploration_strategy):
        # hyperparameters
        self.mpc = mpc
        self._parse_agent_params(**learning_params)
        self.exploration_strategy = exploration_strategy  # New exploration strategy
        self.policy_theta =[]
        self.K = np.array([[119.959032  ,  27.09347287,  27.10575672,  25.59835605]])
        #self.penalty_factor = 500.0
    def train(self, mode = "train"):
        """
        Updates Qfn parameters

        Parameters
        ----------

        Returns
        -------
        dict : {l_theta: parameters of Qfn,
                TD_error: observed TD_error}

        """
        state, obs = self.mpc.model.reset()
        nx = obs.shape[0]
        self.mpc.reset(obs)
        del_J = 0.0
        td_avg = 0
        self.rollout_return = 0
        self.average_td = 0
        u_tilda_k ,  usol  = self.mpc.P(obs)
        nu = usol.shape[1]
        # Exploration with decayed probability (epsilon-greedy style)
        #if np.random.rand() < self.exploration_strategy.value:
             #pertube parameter space to encourage exploration
             #print("pertubating parameter space......")
             #self.mpc.P_learn = self.perturb_parameter_space(self.mpc.P_learn)
             
        T2 = self._extract_T2_matrix(usol)
        self._update_w_tilda(T2, usol)
       

        for it in range(self.mpc.train_it):

            act0, action, add_info = self.mpc.act_forward(obs, mode=mode)
              #compute nominal cost
            J_n = add_info["soln"]['f']
            print("nominal_cost:",J_n)
        
                        #store the optimal policy
            self.policy_theta.append(np.array(action))
         
            #calculate and record the stage cost L_θ (s_k,a_k ), 
            next_state, next_obs, reward, done_step = self.mpc.model.step(act0, it)

            q, info = self.mpc.Q_value(state, act0, soln=add_info["soln"])

            # calculate and record the value function of next state V_θ (s_k )
            v_next, info_next = self.mpc.V_value(
                            next_state, soln=add_info["soln"], mode="update"
                        )
            
            #compute u_fb
            u_fb =csd.mtimes(self.K,next_obs)

            if next_obs[0]<= np.pi/3 and next_obs[0]>=-np.pi/3:
                act_n =  csd.reshape(u_fb, 1, -1)
                print("using feedback law")
            else:
                act_n = action[-1, :]
             #update utilda
            u_tilda_k = np.vstack([action[1:], act_n])
        
             #update wtilda

            self._update_w_tilda(T2, u_tilda_k)
           
            #calculate the sensitivity ∇_θ Q_θ (s_k,a_k )
            grad_q = self.mpc.dQdP(info["soln"], info["pf"], info["p"], info["optimal"])
            
            self.rollout_return += reward
            # TD error
            td_target = reward + self.mpc.gamma * v_next - q

            
             # estimate of dJ
            del_J -= td_target * grad_q.T
            td_avg +=  td_target
            state = next_state.copy()
            obs = next_obs.copy()
            
         
         # RL update step
        
        self.mpc.param_update(del_J, constrained_updates=self.constrained_updates)
        self.average_td = td_avg / self.mpc.train_it
        # Step the exploration strategy to decay epsilon
        self.exploration_strategy.step()
        print(self.exploration_strategy.value)
        print(f"Averaged TD error: {td_avg / self.mpc.train_it}")


    def _extract_T2_matrix(self, usol):
        T2 = self.mpc.Pf[2 * self.mpc.obs_dim + self.mpc.action_dim +
                         (self.mpc.N * self.mpc.action_dim - self.mpc.nv):]
        return csd.reshape(T2, self.mpc.N * self.mpc.action_dim,
                           (self.mpc.N * self.mpc.action_dim - self.mpc.nv))

    def _update_w_tilda(self, T2, u_tilda_k):
        self.mpc.Pf[2 * self.mpc.obs_dim + self.mpc.action_dim:
                    (self.mpc.N * self.mpc.action_dim - self.mpc.nv) +
                    2 * self.mpc.obs_dim + self.mpc.action_dim] = T2.T @ u_tilda_k   
    
    def _parse_agent_params(self, lr, tr, train_params, constrained_updates=False):
        self.lr = lr
        self.tr = tr
        self.iterations = train_params["iterations"]
        self.batch_size = train_params["batch_size"]
        self.constrained_updates = constrained_updates

    def _compute_cost(self, action, state):

        action = csd.reshape(action, 1, 100) 
        cost = self.mpc.cost_model.quadratic_stage_cost( state, action)
        return cost
    
    def penalize_cost_increase(self, current_cost, previous_cost):
        """
        Penalize if the cost increases between iterations.
        Penalty is proportional to the increase in cost.
        """
        cost_difference = current_cost - previous_cost
        penalty = self.penalty_factor * cost_difference
        return penalty
    
    def perturb_parameter_space(self, p_val, epsilon=1e-3):
        """
        Perturbs an orthogonal matrix P_learn while preserving orthogonality.
        
        Parameters:
        - P_learn (np.ndarray): An orthogonal matrix of dimension (N, n_v).
        - epsilon (float): The magnitude of perturbation. Small epsilon results in a small perturbation.
        
        Returns:
        - np.ndarray: A perturbed orthogonal matrix with the same dimensions as P_learn.
            """
       
        P_up = np.array(p_val).copy()
        P_up = P_up.reshape(self.mpc.N*self.mpc.action_dim , self.mpc.nv , order='F')
        N, n_v =P_up.shape
        # Generate a random skew-symmetric matrix Q
        Q = np.random.randn(N, N)
        Q = Q - Q.T  # Make Q skew-symmetric
        
        # Scale Q by epsilon to control the perturbation magnitude
        Q *= epsilon
        
        # Exponentiate the skew-symmetric matrix to get an orthogonal perturbation
        perturbation_matrix = expm(Q)
        
        # Apply the orthogonal perturbation to P_learn
        P_perturbed = np.dot(perturbation_matrix, P_up)
        T2 = null_space(P_perturbed.T) 
        self.mpc.Pf[2*self.mpc.obs_dim + self.mpc.action_dim + (self.mpc.N * self.mpc.action_dim - self.mpc.nv):] = csd.vertcat( csd.reshape(T2, -1, 1))

        P_perturbed = csd.vertcat( csd.reshape(P_perturbed, -1, 1))
        
        return P_perturbed

"""


            #compute cost associated to ultilda_fb
            J_fb = self._compute_cost(u_tilda_k, obs)
            print("feedback_cost:", J_fb)

             #compute nominal cost
            J_n = add_info["soln"]['f']
            print("nominal_cost:",J_n)
            if J_fb <= J_n:
                act0 = u_tilda_k[:nu]
              
            else:
                pass
        
                # Skip cost violation check if this is the first iteration
            #if previous_nominal_cost is None:
                #previous_nominal_cost = J_n
                #done = False
            #else:
                # Check for cost violation by comparing nominal costs
                 #if  J_n > (previous_nominal_cost + 2.0):
                     #print("Cost violation detected, ending episode.")
                     #reward += self.penalty_factor * (J_n - previous_nominal_cost)
                     #done = True
                 #else:
                     #done = False
            #previous_nominal_cost = J_n
            # Q value of the state-action pair
               # End the episode if done
            #if done: 
                #break

                
                 #update u_tilda_k 
            u_tilda_k = np.vstack([action[1:], csd.reshape(u_fb, 1, -1)])
              #calculate state feedback 
            u_fb =csd.mtimes(self.K,next_obs)   #controller.control_action(x0)
             # Initialize previous nominal cost as None
        #previous_nominal_cost = None
         
"""