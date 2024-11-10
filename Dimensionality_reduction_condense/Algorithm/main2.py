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
from scipy.linalg import null_space
from exploration import EpsilonGreedyExploration
from schedulers import ExponentialScheduler
from scipy.linalg import expm
from PrincipalComponentAnalysis import PrincipalComponentAnalysis



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


T1_0=np.load('W_reduce0.npy')
T2_0 = null_space(T1_0.T)

K = np.array([[119.959032  ,  27.09347287,  27.10575672,  25.59835605]])
policy_theta =[]
#T1_0=np.load('T1_newGram.npy')
#T2_0 = null_space(T1_0.T)

nv = T1_0.shape[1]


x0 =  np.array([np.pi, 1,0, 0])  

xSS=  np.array([0, 0 , 0, 0])

param = {"horizon":N,"x_0":x0,"Q":Q,"R":R,"S":Q , "x_SS":xSS,"n_v":nv}

cost_model = Quadratic_stage_cost_model(env, param)

# Create an instance of NominalMPC
nominal_mpc = NominalMPC(model=env, opt_params=param)
pcA = PrincipalComponentAnalysis()
u, usol = nominal_mpc.run_open_loop_mpc()

w_0  =  csd.mtimes(T2_0.T, usol )

# Initialize the EpsilonGreedyExploration strategy
exploration_strategy = EpsilonGreedyExploration(
    epsilon=0.1,  # Initial epsilon
    strength=80,  # Perturbation strength
    hook="timestep",  # Step every timestep
    seed=42  # For reproducibility
)

init_value = 1  # Start with full exploration (epsilon = 1)
decay_factor = 0.95  # Reduce exploration by 5% per step

exploration_scheduler = ExponentialScheduler(init_value=init_value, factor=decay_factor)

def _compute_cost(action, state):

    action = csd.reshape(action, 1, 100) 
    cost = cost_model.quadratic_stage_cost( state, action)
    return cost

def perturb_parameter_space( p_val, epsilon=1e-3):
        """
        Perturbs an orthogonal matrix P_learn while preserving orthogonality.
        
        Parameters:
        - P_learn (np.ndarray): An orthogonal matrix of dimension (N, n_v).
        - epsilon (float): The magnitude of perturbation. Small epsilon results in a small perturbation.
        
        Returns:
        - np.ndarray: A perturbed orthogonal matrix with the same dimensions as P_learn.
            """
       
        P_up = np.array(p_val).copy()
        P_up = P_up.reshape(N*nu , nv , order='F')
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
        agent.Pf[2*nx + nu + (N *nu - nv):] = csd.vertcat( csd.reshape(T2, -1, 1))

        P_perturbed = csd.vertcat( csd.reshape(P_perturbed, -1, 1))
        
        return P_perturbed

def rollout_sample(env, agent, mode="train"):
    state, obs = env.reset()
    agent.reset(obs)
    print(obs)
    rollout_return = 0
    rollout_buffer = BasicBuffer()
    u_tilda_k ,  usol  = agent.P(obs)
    agent.Pf[2*nx + nu :(N *nu- nv)+ 2*nx + nu] = csd.mtimes(agent.T2.T, usol)
    
    for it in range(n_steps):

        act0, action, add_info = agent.act_forward(obs,  mode=mode)

           #compute nominal cost
        J_n = add_info["soln"]['f']

        print("nominal_cost:",J_n)
        
        policy_theta.append(action)

        next_state, next_obs, reward, _ = env.step(act0 , it)

        if mode == "train":
            rollout_buffer.push(
                state, obs, act0 , reward, next_state, next_obs, add_info
            )
        
        #compute u_fb
        u_fb =csd.mtimes(K,next_obs)

        if next_obs[0]<= np.pi/3 and next_obs[0]>=-np.pi/3:
            act_n =  csd.reshape(u_fb, 1, -1)
            print("using feedback law")
        else:
            act_n = action[-1, :]
             #update utilda
        u_tilda_k = np.vstack([action[1:], act_n])
        

        #update u_tilda_k using feedback law 
        #u_tilda_k = np.vstack([action[1:], action[-1, :]])

        #update w_k using utilda 
        agent.Pf[2*nx + nu :(N *nu- nv)+ 2*nx + nu] = agent.T2.T@u_tilda_k
        rollout_return += reward
        state = next_state.copy()
        obs = next_obs.copy()


    return rollout_return, rollout_buffer


def plot_stats1(stats):
    rows = len(stats)
    cols = 1

    fig, ax = plt.subplots(rows, cols, figsize=(12, 6))

    for i, key in enumerate(stats):
        vals = stats[key]
        # Calculate a moving average to smooth the plot
        smoothed_vals = [np.mean(vals[j-10:j+10]) for j in range(10, len(vals)-10)]
        if len(stats) > 1:
            ax[i].plot(range(len(smoothed_vals)), smoothed_vals)
            ax[i].set_title(key, size=18)
        else:
            ax.plot(range(len(smoothed_vals)), smoothed_vals)
            ax.set_title(key, size=18)
    
    plt.tight_layout()

    plt.show()


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


n_steps = 300
seed = 1
agent_params= {
        
        "gamma": 0.95,
        "T1":T1_0,
        "T2":T2_0,
        "w": w_0,
        "eps": 0.25,
        "learning_params": {
            "lr": 1e-4,
            "tr": 0.2,
            "train_params": {
                "iterations": 200,
                "batch_size": 32
            },
            "constrained_updates": True
        }
    }
n_iterations = 200
n_trains = 10
n_evals = 1
n_steps = 300
max_len_buffer = 500

# Experiment init
replay_buffer = ReplayBuffer(max_len_buffer, seed)          

# Agent init
agent = MPCfunapprox_ex(env, agent_params,param)
# Test run
_, obs = env.reset()
agent.reset(obs)
act0, act, info = agent.act_forward(obs)

stats = {'TD Loss': [], 'Training Returns': [], 'Evaluation Returns': []}
# main loop
for it in range(n_iterations):
    print(f"Iteration: {it}")
    t_returns, e_returns = [],[]
    
    # training rollouts
    for _ in range(n_trains):
        rollout_return, rollout_buffer = rollout_sample(env, agent, mode="train")
        replay_buffer.push(rollout_buffer.buffer)
        t_returns.append(rollout_return)

    # Save training returns to stats
    stats['Training Returns'].extend(t_returns)
   
    # agent training
    agent.train(replay_buffer)
    stats['TD Loss'].append(agent.learning_module.TD_avg)
    np.save('P_learn1S',agent.P_learn)
    # training rollouts
    for _ in range(n_evals):
        rollout_return, rollout_buffer = rollout_sample(env, agent, mode="eval")
        e_returns.append(rollout_return)
    
    # Save evaluation returns to stats
    stats['Evaluation Returns'].extend(e_returns)

    print(f"Training rollout return: {np.mean(t_returns)}")
    # print(f"Evaluation rollout return: {np.mean(e_returns)}")

#stats = {'TD Loss': t_returns, 'Returns':  e_returns}

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

np.save('T1_train200_reduce.npy', T1)
np.save('T2_train200_reduce.npy', T2)

U_opt = np.array(policy_theta)
np.save('U_optE_reduce2', U_opt)
S = pcA.compute_sensitivity_matrix(U_opt)
W ,nv_new = pcA.compute_active_subspace(S)
np.save('W_reduce1', W)

#print(agent.P_learn)

plot_stats1(stats)

"""
T1_0 = np.load('T1_RT11.npy')
T2_0 = null_space(T1_0.T)
"""

"""
if it==0:
            J_fb =  _compute_cost(u_tilda_k, obs) 
            print("feedback_cost:", J_fb)
            if J_fb <= J_n:
                act0 = u_tilda_k[:nu]
            else:
                pass
"""