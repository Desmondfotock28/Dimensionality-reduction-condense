from replay_buffer import ReplayBuffer

class Qlearning:
    def __init__(self, mpc, learning_params, seed):
        # hyperparameters
        self.mpc = mpc
        self._parse_agent_params(**learning_params)

    def train(self, replay_buffer: ReplayBuffer):
        """
        Updates Qfn parameters by using (random) data from replay_buffer

        Parameters
        ----------
        replay_buffer : ReplayBuffer object
            Class instance containing all past data

        Returns
        -------
        dict : {l_theta: parameters of Qfn,
                TD_error: observed TD_error}

        """
        td_avg = 0
        batch_size = min(self.batch_size, replay_buffer.size)
        train_it = min(self.iterations, int(3.0 * replay_buffer.size / batch_size))
        for it in range(train_it):
            (
                states,
                _,
                actions,
                rewards,
                next_states,
                _,
                infos,
            ) = replay_buffer.sample(batch_size)

            del_J = 0.0
            for j, state in enumerate(states):
                action = actions[j]
                next_state = next_states[j]
                reward = rewards[j]
                
                # Q value of the state-action pair
                q, info = self.mpc.Q_value(state, action, soln=infos[j]["soln"])
                
                # V value of the next state
                v_next, info_next = self.mpc.V_value(
                    next_state, soln=infos[j]["soln"], mode="update"
                )

                # TD error
                td_target = rewards[j][0] + self.mpc.gamma * v_next - q
                
                # sensitivity of Q 
                grad_q = self.mpc.dQdP(info["soln"], info["pf"], info["p"])
                
                # estimate of dJ
                del_J -= td_target * grad_q.T
                td_avg += td_target
            del_J = del_J/batch_size
                
            # RL update step
            self.mpc.param_update(del_J, constrained_updates=self.constrained_updates)
        print(f"Averaged TD error: {td_avg / train_it}")

    def _parse_agent_params(self, lr, tr, train_params, constrained_updates=False):
        self.lr = lr
        self.tr = tr
        self.iterations = train_params["iterations"]
        self.batch_size = train_params["batch_size"]
        self.constrained_updates = constrained_updates