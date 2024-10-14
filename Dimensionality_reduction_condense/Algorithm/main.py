import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import casadi as csd
from CartPole_model import CartPole
from nominal_mpc_condense import NominalMPC
from parametric_mpc_formulation_condense import ParamMPCformulation
from parametric_mpc_condense import  MPCfunapprox
from quadratics_cost_condense import Quadratic_stage_cost_model
from MPCQlearning_condense import MPCQlearning
from IPython.display import HTML
from exploration import EpsilonGreedyExploration


class MPCfunapprox_ex(MPCfunapprox, ParamMPCformulation):
    def __init__(self, model, cost_model, agent_params, opt_params, train_it,exploration_strategy, seed=1):
        # Step 1: Initialize the parent class (MPCfunapprox)
        super().__init__(model, agent_params,  opt_params)
        self.train_it = train_it
        self.cost_model = cost_model
        # Step 2: Parse agent-specific parameters and other initialization
        self._parse_agent_params(**agent_params)
        
        # Step 3: Initialize the learning module
        self.learning_module = MPCQlearning(self, self.learning_params,exploration_strategy)
        
        # Step 4: Initialize RL update formulation
        self.constraint_param_opt(self.learning_module.lr, self.learning_module.tr)
    
    def train(self):
        # Call the Qlearning train method
       self.learning_module.train()

def plot_stats(stats):
    rows = len(stats)
    cols = 1

    fig, ax = plt.subplots(rows, cols, figsize=(12, 6))

    for i, key in enumerate(stats):
        vals = stats[key]
        #vals = [np.mean(vals[i-10:i+10]) for i in range(10, len(vals)-10)]
        vals = [np.mean(vals[i-2:i+3]) for i in range(2, len(vals)-3)] 
        if len(stats) > 1:
                ax[i].plot(range(len(vals)), vals)
                ax[i].set_title(key, size=18)
               # ax[i].axhline(12677.3, color='r', linestyle='--', label='Optimal Cost')  # Add horizontal line
                #ax[i].legend()  # Add legend
        else:
            ax.plot(range(len(vals)), vals)
            ax.set_title(key, size=18)
    plt.tight_layout()
    plt.show()


def display_video(frames):
    # Copied from: https://colab.research.google.com/github/deepmind/dm_control/blob/master/tutorial.ipynb
        orig_backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        matplotlib.use(orig_backend)
        ax.set_axis_off()
        ax.set_aspect('equal')
        ax.set_position([0, 0, 1, 1])
        im = ax.imshow(frames[0])
        def update(frame):
            im.set_data(frame)
            return [im]
        anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                    interval=50, blit=True, repeat=False)
         # Save as video file
        video_path = "test_policy.mp4"
        anim.save(video_path, writer="ffmpeg", fps=20)
        print(f"Animation saved as {video_path}")
        return HTML(anim.to_html5_video())


def test_mpc_policy(env, policy, episodes=5):
    frames = []
    mode = "train"
    n_steps= 200
    Pf = policy.Pf
    p_val = policy.P_learn

    for episode in range(episodes):
        state, obs = env.reset()
        u_tilda_k ,  usol  = policy.P(obs)
        policy.Pf[2*nx + nu :(N *nu- nv)+ 2*nx + nu] = csd.mtimes(policy.T2.T, usol)
        frames.append(env.render())

        for _ in range(n_steps):
            act0, action, add_info = policy.act_forward(obs, Pf_val=Pf, P_learn=p_val,  mode=mode)
            
            next_state, _, _, _ = env.step(act0)
             #update u_tilda_k using feedback law 
            u_tilda_k = np.vstack([action[1:], action[-1, :]])

            #update w_k using utilda 
            policy.Pf[2*nx + nu :(N *nu- nv)+ 2*nx + nu] = policy.T2.T@ u_tilda_k
      
            img = env.render()
            frames.append(img) 
            obs = next_state
           
    return display_video(frames)


# Initialize the EpsilonGreedyExploration strategy
exploration_strategy = EpsilonGreedyExploration(
    epsilon=0.1,  # Initial epsilon
    strength=80,  # Perturbation strength
    hook="timestep",  # Step every timestep
    seed=42  # For reproducibility
)

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

xSS=  np.array([0, 0 , 0, 0])



T1_0 = np.load('T1_G10.npy')
T2_0 = np.load('T2_G10.npy')
nv = T1_0.shape[1]


param = {"horizon":N,"x_0":x0,"Q":Q,"R":R,"S":Q , "x_SS":xSS,"n_v":nv}

cost_model = Quadratic_stage_cost_model(env, param)

# Create an instance of NominalMPC
nominal_mpc = NominalMPC(model=env, opt_params=param)

u, usol = nominal_mpc.run_open_loop_mpc()

w_0  =  csd.mtimes(T2_0.T, usol )

n_steps = 30

seed = 1
agent_params= {
        
        "gamma": 0.95,
        "T1":T1_0,
        "T2":T2_0,
        "w": w_0,
        "eps": 0.25,
        "learning_params": {
            "lr": 1e-4,
            "tr": 0.1,
            "train_params": {
                "iterations":500,
                "batch_size": 60
            }, 
            "constrained_updates": True
      }
    } 
n_iterations = 500

# Agent init
agent = MPCfunapprox_ex(env,cost_model, agent_params,param,n_steps,exploration_strategy)

# Test run 
_, obs = env.reset()
agent.reset(obs)
act0, act, info = agent.act_forward(obs)

v_mpc, info_v = agent.V_value(obs)
print(f"Value function: {v_mpc}")

q_mpc, info_q = agent.Q_value(obs, act0)
print(f"Q function: {q_mpc}")

#main loop 
# Initialize a dictionary to store policy gradient loss and returns for each episode
stats = {'TD Loss': [], 'Returns': []}

for it in range(n_iterations):
    print(f"Iteration: {it}")
    # agent training
    agent.train()
    print(f"rollout_return: {agent.learning_module.rollout_return}")

    stats['Returns'].append(agent.learning_module.rollout_return)
    stats['TD Loss'].append(agent.learning_module.average_td)

U_opt = agent.learning_module.policy_theta
np.save('U_opt1', U_opt)

T1 =  agent.P_learn
T1 = np.array(T1).reshape(N*nu , nv , order='F')
T2 = agent.Pf[2*nx + nu + (N *nu - nv):]
T2 = np.array(T2).reshape(N*nu , (N *nu - nv), order='F')
w  = agent.Pf[2*nx + nu :(N *nu - nv)+ 2*nx + nu]
w  = np.array(w).reshape( (N *nu - nv ), 1 )

plot_stats(stats)
   
np.save('T1_nv_new.npy', T1)
np.save('T2_nv_new.npy', T2)

test_mpc_policy(env, agent)

print(agent.P_learn)