from CartPole_model import CartPole
from quadratics_cost_condense import Quadratic_stage_cost_model
import numpy as np
from nominal_mpc_condense import NominalMPC
import matplotlib.pyplot as plt
from parametric_mpc_condense import MPCfunapprox
import casadi as csd




env = CartPole('rgb_array')
env.reset()

nx = env.observation_space.shape[0]
nu = env.action_space.shape[0]

N= 100

Q = 2
Q = Q * np.diag([1e3, 1e3, 1e-2, 1e-2])

R = 2
R = R * np.diag([1e-1])

x0 =  np.array([np.pi, 1,0, 0])  

xSS=  np.array([0, 0 , 0, 0])

T1_0 = np.load('T1_Hess_dense.npy')# Load the saved T1 and T2

T2_0 = np.load('T2_Hess_dense.npy')
nv = T1_0.shape[1]

param = {"horizon":N,"x_0":x0,"Q":Q,"R":R,"S":Q , "x_SS":xSS,"n_v":nv}

cost_model = Quadratic_stage_cost_model(env, param)

# Create an instance of NominalMPC
nominal_mpc = NominalMPC(model=env, opt_params=param)

u, usol = nominal_mpc.run_open_loop_mpc()

w_0  =  csd.mtimes(T2_0.T, usol )

def plot_control_inputs(t, u, uSS, fignum):
    linewidth = 1.5
    uSSp = uSS

    # Plot control inputs
    plt.figure(fignum)

    plt.plot(t, uSSp * np.ones_like(t), linewidth=linewidth, label='Reference')
    plt.step(t[:-1], u, linewidth=linewidth, where='post', label='Trajectory')
    plt.grid(True)
    plt.xlabel('$t$ [s]', fontsize=12)
    plt.ylabel('$u^\star(t)$ [Nm]', fontsize=12)
    plt.legend()

    plt.tight_layout()
    plt.show()

tf = 2
xSS=  np.array([0, 0 , 0, 0])
uSS = 0
t_ol = np.linspace(0,tf , N+1)
plot_control_inputs(t_ol, u, uSS, 1)

agent_params= {
        
        "gamma": 0.95,
        "T1":T1_0,
        "T2":T2_0,
        "w": w_0,
        "eps": 0.25,
        "learning_params": {
            "lr": 1e-3,
            "tr": 0.2,
            "train_params": {
                "iterations":150,
                "batch_size": 60
            }, 
            "constrained_updates": True
      }
    } 

mpc = MPCfunapprox(env, agent_params, param)

# Test run 
_, obs = env.reset()
mpc.reset(obs)
act0, act, info = mpc.act_forward(obs)

v_mpc, info_v = mpc.V_value(obs)
print(f"Value function: {v_mpc}")

q_mpc, info_q = mpc.Q_value(obs, act0)
print(f"Q function: {q_mpc}")