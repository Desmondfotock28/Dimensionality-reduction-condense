import casadi as csd 
import numpy as np


class MPCQlearning:
    def __init__(self, mpc, learning_params ):
        # hyperparameters
        self.mpc = mpc
        self._parse_agent_params(**learning_params)

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
        T2 = self.mpc.Pf[2*self.mpc.obs_dim + self.mpc.action_dim + (self.mpc.N * self.mpc.action_dim - self.mpc.nv): ]
        T2 = csd.reshape(T2,self.mpc.N*self.mpc.action_dim,(self.mpc.N *self.mpc.action_dim - self.mpc.nv))
        self.mpc.Pf[2*self.mpc.obs_dim + self.mpc.action_dim :(self.mpc.N * self.mpc.action_dim - self.mpc.nv)+
                         2*self.mpc.obs_dim + self.mpc.action_dim] = csd.mtimes(T2.T, usol)
        epsilon =0.01
        
        for it in range(self.mpc.train_it):
            
            
            if np.random.random() < epsilon:
                act0 = np.array(np.random.uniform(-80, 80)).reshape(1,1)
                soln = None
            else: 
                act0, action, add_info = self.mpc.act_forward(obs,  mode=mode)
                soln = add_info["soln"]
                    #update u_tilda_k using feedback law 
                u_tilda_k = np.vstack([action[1:], action[-1, :]])

                    #update w_k 

                self.mpc.Pf[2*self.mpc.obs_dim + self.mpc.action_dim :(self.mpc.N * self.mpc.action_dim - self.mpc.nv)+
                                2*self.mpc.obs_dim + self.mpc.action_dim] = T2.T@u_tilda_k
                
                    
            #calculate and record the stage cost L_θ (s_k,a_k ), 
            next_state, next_obs, reward, _ = self.mpc.model.step(act0)

            # Q value of the state-action pair
            q, info = self.mpc.Q_value(state, act0, soln=soln)

            # calculate and record the value function of next state V_θ (s_k )
            v_next, info_next = self.mpc.V_value(
                            next_state, soln=soln, mode="update"
                        )
            
           
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
        print(f"Averaged TD error: {td_avg / self.mpc.train_it}")
       

    def _parse_agent_params(self, lr, tr, train_params, constrained_updates=False):
        self.lr = lr
        self.tr = tr
        self.iterations = train_params["iterations"]
        self.batch_size = train_params["batch_size"]
        self.constrained_updates = constrained_updates

