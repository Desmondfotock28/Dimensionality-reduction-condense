import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import casadi as csd
from CartPole_model import CartPole
from nominal_mpc_condense import NominalMPC
from parametric_mpc_formulation_condense import ParamMPCformulation
from parametric_mpc_condense import  MPCfunapprox
from replay_buffer import BasicBuffer, ReplayBuffer
from quadratics_cost_condense import Quadratic_stage_cost_model
from Qlearning import Qlearning
from IPython.display import HTML



class MPCfunapprox_ex(MPCfunapprox, ParamMPCformulation):
    def __init__(self, model, agent_params, opt_params, seed=1):
        # Step 1: Initialize the parent class (MPCfunapprox)
        super().__init__(model, agent_params,  opt_params)

        # Step 2: Parse agent-specific parameters and other initialization
        self._parse_agent_params(**agent_params)
        
        # Step 3: Initialize the learning module
        self.learning_module = Qlearning(self, self.learning_params, seed)
        
        # Step 4: Initialize RL update formulation
        self.constraint_param_opt(self.learning_module.lr, self.learning_module.tr)
    
    def train(self, replay_buffer):
        # Call the Qlearning train method
       self.learning_module.train(replay_buffer)


env = CartPole('rgb_array')

env.reset()
nx = env.observation_space.shape[0]
nu = env.action_space.shape[0]

N= 100

Q = 1
Q = Q * np.diag([1, 1, 0.1, 0.1])

R = 1
R = R * np.diag([0.001])

x0 =  np.array([np.pi, 1,0, 0])  


T1_0 = np.load('T1_G10.npy')
T2_0 = np.load('T2_G10.npy')

nv = T1_0.shape[1]


x0 =  np.array([np.pi, 1,0, 0])  

xSS=  np.array([0, 0 , 0, 0])

param = {"horizon":N,"x_0":x0,"Q":Q,"R":R,"S":Q , "x_SS":xSS,"n_v":nv}

cost_model = Quadratic_stage_cost_model(env, param)

# Create an instance of NominalMPC
nominal_mpc = NominalMPC(model=env, opt_params=param)

u, usol = nominal_mpc.run_open_loop_mpc()

w_0  =  csd.mtimes(T2_0.T, usol )


def rollout_sample(env, agent, mode="train"):
    state, obs = env.reset()
    agent.reset(obs)
    rollout_return = 0
    rollout_buffer = BasicBuffer()
    u_tilda_k ,  usol  = agent.P(obs)
    agent.Pf[2*nx + nu :(N *nu- nv)+ 2*nx + nu] = csd.mtimes(agent.T2.T, usol)
 

    for _ in range(n_steps):

        act0, action, add_info = agent.act_forward(obs,  mode=mode)

        next_state, next_obs, reward, _ = env.step(act0)

        if mode == "train":
            rollout_buffer.push(
                state, obs, act0 , reward, next_state, next_obs, add_info
            )
        #update u_tilda_k using feedback law 
        u_tilda_k = np.vstack([action[1:], action[-1, :]])

        #update w_k using utilda 
        agent.Pf[2*nx + nu :(N *nu- nv)+ 2*nx + nu] = agent.T2.T@u_tilda_k
        rollout_return += reward
        state = next_state.copy()
        obs = next_obs.copy()

    return rollout_return, rollout_buffer


def plot_stats(stats):
    rows = len(stats)
    cols = 1

    fig, ax = plt.subplots(rows, cols, figsize=(12, 6))

    for i, key in enumerate(stats):
        vals = stats[key]
        vals = [np.mean(vals[i-10:i+10]) for i in range(10, len(vals)-10)]
        if len(stats) > 1:
            ax[i].plot(range(len(vals)), vals)
            ax[i].set_title(key, size=18)
        else:
            ax.plot(range(len(vals)), vals)
            ax.set_title(key, size=18)
    plt.tight_layout()
    plt.show()


n_steps = 200
seed = 1
agent_params= {
        
        "gamma": 0.95,
        "T1":T1_0,
        "T2":T2_0,
        "w": w_0,
        "eps": 0.25,
        "learning_params": {
            "lr": 1e-3,
            "tr": 0.1,
            "train_params": {
                "iterations": 5,
                "batch_size": 32
            },
            "constrained_updates": True
        }
    }
n_iterations = 5
n_trains = 1
n_evals = 0
n_steps = 200
max_len_buffer = 25

# Experiment init
replay_buffer = ReplayBuffer(max_len_buffer, seed)          

# Agent init
agent = MPCfunapprox_ex(env, agent_params,param)
# Test run
_, obs = env.reset()
agent.reset(obs)
act0, act, info = agent.act_forward(obs)


# main loop
for it in range(n_iterations):
    print(f"Iteration: {it}")
    t_returns, e_returns = [],[]
    
    # training rollouts
    for _ in range(n_trains):
        rollout_return, rollout_buffer = rollout_sample(env, agent, mode="train")
        replay_buffer.push(rollout_buffer.buffer)
        t_returns.append(rollout_return)
   
    # agent training
    agent.train(replay_buffer)

    # training rollouts
    for _ in range(n_evals):
        rollout_return, rollout_buffer = rollout_sample(env, agent, mode="eval")
        e_returns.append(rollout_return)

    print(f"Training rollout return: {np.mean(t_returns)}")
    # print(f"Evaluation rollout return: {np.mean(e_returns)}")
stats = {'TD Loss': t_returns, 'Returns':  e_returns}
# final evaluation performance

#f_returns = []
#for _ in range(10):
    #rollout_return, rollout_buffer = rollout_sample(env, agent, mode="final")
    #f_returns.append(rollout_return)
#print(f"Final rollout return: {np.mean(f_returns)}")

T1 =  agent.P_learn
T1 = np.array(T1).reshape(N*nu , nv , order='F')
T2 = agent.Pf[2*nx + nu + (N *nu - nv):]
T2 = np.array(T2).reshape(N*nu , (N *nu - nv), order='F')

np.save('T1_nv_new.npy', T1)
np.save('T2_nv_new.npy', T2)


print(agent.P_learn)

plot_stats(stats)