import numpy as np
import time
import matplotlib.pyplot as plt
from casadi import *


def plot_results(t, x, u, xSS, uSS, fignum):
    linewidth = 1.5
    xSSp = xSS
    uSSp = uSS

    # Plot state trajectories
    plt.figure(fignum[0])

    plt.subplot(4, 1, 1)
    plt.plot(t, xSSp[0] * np.ones_like(t), linewidth=linewidth, label='Reference')
    plt.plot(t, x[:,0], linewidth=linewidth, label='Trajectory')
    plt.grid(True)
    plt.xlabel('$t$ [s]', fontsize=12)
    plt.ylabel('$q_1(t)$ [rad]', fontsize=12)
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(t, xSSp[1] * np.ones_like(t), linewidth=linewidth, label='Reference')
    plt.plot(t, x[:,1], linewidth=linewidth, label='Trajectory')
    plt.grid(True)
    plt.xlabel('$t$ [s]', fontsize=12)
    plt.ylabel('$q_2(t)$ [rad]', fontsize=12)
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(t, xSSp[2] * np.ones_like(t), linewidth=linewidth, label='Reference')
    plt.plot(t, x[:,2], linewidth=linewidth, label='Trajectory')
    plt.grid(True)
    plt.xlabel('$t$ [s]', fontsize=12)
    plt.ylabel('$\dot{q_1}(t)$ [rad/s]', fontsize=12)
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(t, xSSp[3] * np.ones_like(t), linewidth=linewidth, label='Reference')
    plt.plot(t, x[:,3], linewidth=linewidth, label='Trajectory')
    plt.grid(True)
    plt.xlabel('$t$ [s]', fontsize=12)
    plt.ylabel('$\dot{q_2}(t)$ [rad/s]', fontsize=12)
    plt.legend()

    plt.tight_layout()

    # Plot control inputs
    plt.figure(fignum[1])

    plt.subplot(2, 1, 1)
    plt.plot(t, uSSp[0] * np.ones_like(t), linewidth=linewidth, label='Reference')
    plt.step(t[:-1], u[:,0], linewidth=linewidth, where='post', label='Trajectory')
    plt.grid(True)
    plt.xlabel('$t$ [s]', fontsize=12)
    plt.ylabel('$u_1^\star(t)$ [Nm]', fontsize=12)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, uSSp[1] * np.ones_like(t), linewidth=linewidth, label='Reference')
    plt.step(t[:-1], u[:,1], linewidth=linewidth, where='post', label='Trajectory')
    plt.grid(True)
    plt.xlabel('$t$ [s]', fontsize=12)
    plt.ylabel('$u_2^\star(t)$ [Nm]', fontsize=12)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Define system parameters
def robot_parameters() -> tuple:
    """Define system parameters and constraints."""
    # System parameters
    b1 = 200  # kg m^2/rad
    b2 = 50   # kg m^2/rad
    b3 = 23.5 # kg m^2/rad
    b4 = 25   # kg m^2/rad
    b5 = 122.5# kg m^2/rad
    c1 = -25  # Nm/s^2
    g1 = 784.8# Nm
    g2 = 245.3# Nm
    l1 = 0.5  # m
    l2 = 0.5  # m
    
    # Create a dictionary to hold all parameters
    PAR = {
        'b1': b1,
        'b2': b2,
        'b3': b3,
        'b4': b4,
        'b5': b5,
        'c1': c1,
        'g1': g1,
        'g2': g2,
        'l1': l1,
        'l2': l2
    }
    
    return PAR

def Robot_ode(x, u):
    # Extract parameters
    PAR = robot_parameters()  # Ignore constraints for now
    b1, b2, b3, b4, b5 = PAR['b1'], PAR['b2'], PAR['b3'], PAR['b4'], PAR['b5']
    c1, g1, g2, l1, l2 = PAR['c1'], PAR['g1'], PAR['g2'], PAR['l1'], PAR['l2']
   # state x = (x1 = q1, x2=q2, x3=q1_dot , x4=q2_dot)
    b11 = b1 + b2*cos(x[1]) 
    b12 =b21= b3 + b4*cos(x[1]) 
    b22 = b5
    c11 = -c1*x[3]*sin(x[1])
    c12 = -c1*(x[2]+x[3])*sin(x[1])
    c21 = c1*x[2]*sin(x[1]) 
    g_x = g1*cos(x[0]) + g2*cos(x[0] + x[1])
    g_y =g2*cos(x[0] + x[1])
    #state space equation 
    dx1 = x[2]

    dx2 = x[3]

    dx3 = (1/(b22*b11-b12*b21))*(b22*u[0]-b12*u[1] -b22*c11*x[0] + b12*c21*x[2]-b22*c12*x[3]+b12*g_y-b22*g_x)

    dx4 = (1/(b21*b12-b11*b22))*(b21*u[0]-b11*u[1]-b21*c11*x[0]+b11*c21*x[2]-b21*c12*x[3]+b11*g_y-b21*g_x)
    
    dx = vertcat( dx1, dx2 ,dx3, dx4)

    return dx 

def ERK4_no_param(f, x, u, h):
    if isinstance(f, SX):
        f = Function("f", [x, u], [f], ["x", "u"], ["xf"])

    k1 = f(x, u)
    k2 = f(x + h / 2 * k1, u)
    k3 = f(x + h / 2 * k2, u)
    k4 = f(x + h * k3, u)
    xf = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return xf

def shift(T, t0, x0, u, f):
    """
    Shift the state and time forward by one timestep.

    Args:
        T (float): The timestep.
        t0 (float): The current time.
        x0 (np.array): The current state.
        u (np.array): The control inputs.
        f (Function): The system function.

    Returns:
        tuple: The updated time, state, and control inputs.
    """
    st = x0
    con = u[0, :]
    #f_value = f(st, con)
    #st = st + T * f_value
    st =  ERK4_no_param(f, st, con, T)

    x0 = np.array(st.full()).flatten()

    t0 = t0 + T
    u0 = np.vstack([u[1:], u[-1, :]])

    return t0, x0, u0

# Defind condense vector 

def g_vec(x0 , U , N , system ,Ts):
    g=[x0]
    U = reshape(U, nu,N)
    x_k = x0
    for i in range(1,N):
        x_k = ERK4_no_param(system, x_k,  U[:,i-1], Ts) 
        g.append(x_k)
    x_N = ERK4_no_param(system, x_k,  U[:,N-1], Ts)
    g.append(x_N)
    return g

def compute_block_matrix( Q_base, R_base, N):
    # Compute the solution to the discrete-time algebraic Riccati equation using control library

    
    # Construct the block diagonal matrix Q
    Q_blocks = [Q_base] * N     #+ [Q_base]

    Q_d = np.block([[Q_blocks[i] if i == j else np.zeros_like(Q_base) for j in range(N)] for i in range(N)])
    
    # Construct the block diagonal matrix R
    R_d = np.block([[R_base if i == j else np.zeros_like(R_base) for j in range(N)] for i in range(N)])

    return  Q_d ,  R_d

#Controller frequency and Prediction horizon
Ts =  0.05    #sampling time in [s]

N =  60      #prediction horizon

tf= 3

nx= 4       #state dimension 

nu= 2       #input dimension

# System states and controls 
x = SX.sym('x', nx);    # states of the system 

u = SX.sym('u', nu);    # control of the system

dx = Robot_ode(x, u)

# Create the CasADi function
system = Function("sys", [x, u], [dx])

# Declear empty sys matrices
U = SX.sym('U',nu,N)               # Decision variables (controls)

#Parameters:initial state(x0)

P_a = SX.sym('P',nx+nx,1)

X = SX.sym('X',nx,(N+1)) # Decision variables (states)

# weighing matrices (states)
Q = 1
Q = Q * np.diag([5, 1, 0.5, 0.5])

R = 1
R = R*np.diag([1, 1])

Q_d , R_d= compute_block_matrix( Q, R, N)

G = g_vec(P_a[:nx] , U , N , system ,Ts)

X =[]

for i in range(len(G)):
    X.append(G[i][2:])

X = vertcat(*X)
G = vertcat(*G)

U = vertcat(reshape(U, -1,1))

#symbolic variable for g_vec 
g_d = SX.sym("g_d", (N)*nx)

# symbolic varaible for control input vector 

u_c = SX.sym("u_x",N*nu)

obj = g_d.T @ Q_d @ g_d + u_c.T @ R_d@ u_c 

# Create the CasADi function
objective = Function("J", [g_d, u_c], [obj])

# State constraints
lb_x = np.array([-np.finfo(np.float32).max, -np.finfo(np.float32).max , -3 * pi / 2, -3 * pi / 2])
ub_x = np.array([np.finfo(np.float32).max,  np.finfo(np.float32).max , 3 * pi / 2, 3 * pi / 2])

lbx = [lb_x[2:]]*(N + 1) 
ubx = [ub_x[2:]]*(N + 1)
lbx = vertcat(*lbx)
ubx = vertcat (*ubx)

# Input constraints
lb_u = np.array([-1000,-1000])
ub_u = np.array([1000,1000])

lbu = [lb_u]*N
ubu = [ub_u]*N
ubu = vertcat(*ubu)
lbu = vertcat (*lbu)

Opt_Vars = U

def objective_cost():
    J = objective(G[:N*nx],U) 
    return J

def inequality_constraints():
  
    hu = []   # Box constraints on active inputs
    hx = []   # Box constraints on states
    
    hu.append(lbu-U )
    hu.append(U - ubu)
    hx.append(lbx-X)
    hx.append(X - ubx)

    return  hu, hx
def equality_constraints():
    g = []  # Equality constraints initialization
    g.append(G[N*nx:]-P_a[nx:])   #terminal equality constraint
    return g


def Pi_opt_formulation():
    J = objective_cost()
    g = equality_constraints()
    hu, hx = inequality_constraints()
    G_e = vertcat(*g)
    Hu = vertcat(*hu)
    Hx = vertcat(*hx)
    G_vcsd = vertcat(*g , *hx, *hu)
    lbg = [0] * G_e.shape[0] + [-np.inf] * (Hx.shape[0] + Hu.shape[0])
    ubg = [0] * G_e.shape[0] + [0] * (Hx.shape[0] + Hu.shape[0])
    lbg_vcsd = vertcat(*lbg)
    ubg_vcsd = vertcat(*ubg)

    opts_setting = {
        "ipopt.max_iter": 500,
        "ipopt.print_level": 4,
        "print_time": 1,
        "ipopt.acceptable_tol": 1e-6,
        "ipopt.acceptable_obj_change_tol": 1e-6,
    }
    vnlp_prob = {
        "f": J,
        "x": Opt_Vars,
        "p": vertcat(P_a),
        "g": G_vcsd,
    }
    pisolver = nlpsol("vsolver", "ipopt", vnlp_prob)
    return lbg_vcsd, ubg_vcsd, G_vcsd, pisolver 

lbg_vcsd, ubg_vcsd, G_vcsd , pisolver = Pi_opt_formulation()

def run_open_loop_mpc(x0, u0 , solver ):
      # Initial control inputs and state
    xSS=  np.array([np.pi/2, 0 , 0, 0])
    u_st_0 = np.tile(u0, (N, 1))

    args_p =  np.array(
            [x0] + [xSS]
        )
    
    args_p= vertcat(*args_p)
    args_x0 = u_st_0.T.reshape(-1)
   # Solve the optimization problem
    sol = solver(x0=args_x0, p=args_p, lbg=lbg_vcsd, ubg=ubg_vcsd)
    usol = sol['x']

    # Extract the control inputs from the solution
    u = np.array(sol['x']).reshape((N , nu))
  
   # construct xsol 
    xsol = g_vec(x0 , u , N, system,Ts)
 
    xsol = vertcat(*xsol)

    xsol = np.array(xsol).reshape(N+1 , nx)
 
    # Convert lists to numpy arrays for easier handling
    return xsol, u ,usol

u0 = np.array([1000,1000])

x0 =  np.array([-5, -4, 0,  0])  

xsol, u_ol, usol  = run_open_loop_mpc(x0, u0 , pisolver)

xSS=  np.array([np.pi/2, 0 , 0, 0])
uSS = np.array([0 , 0])
t_ol = np.arange(0, tf + Ts, Ts)

plot_results(t_ol, xsol, u_ol, xSS, uSS, [1, 2])


def run_closed_loop_mpc(x0, Ts, sim_time, solver):
    x_goal = np.array([np.pi/2, 0, 0, 0])
    u0 = np.array([1000,1000])
    t0 = 0
    nx = x0.shape[0]
    t = [t0]
    x_ol = np.zeros((nx, int(sim_time / Ts) + 1))  # Open loop predicted states
    x_ol = [x0]
    mpc_i = 0
    x_cl = [x0]   # Store predicted states in the closed loop
    u_cl = []    # Store control inputs in the closed loop
    goal_tolerance = 0.01  # Define a goal tolerance
    u_st_0 = np.tile(u0, (N, 1))
    args_p = np.array([x0] + [xSS])
    args_p = vertcat(*args_p)
    cost_n = []
    
    while np.linalg.norm(x0 - x_goal, 2) > goal_tolerance and mpc_i < int(sim_time / Ts):

        args_p[:nx] = x0
        
        args_x0 =  u_st_0.T.reshape(-1)
        sol = solver(x0=args_x0, p=args_p, lbg=lbg_vcsd, ubg=ubg_vcsd)
        x_opt = sol['x']
        cost_n.append(sol['f'])
        u_0 = x_opt[:nu]
        # Extract the control inputs from the solution
        u = np.array(sol['x']).reshape(( N , nu))
        #construct x_pred 
        x_pred = g_vec(x0 , u , N, system,Ts)
        x_pred = vertcat(*x_pred)
        x0 = x_pred[nx:2*nx]
        x_pred = np.array(x_pred).reshape((N+1, nx))
        x_cl.append(x_pred)
        u_cl.append(u_0)
        t.append(t0)
        t0 = t0 + Ts
        x_ol.append(x0)
        u_st_0 = np.vstack([x_opt[1:], x_opt[-1]])

         
        mpc_i += 1

    x_ol = vertcat(*x_ol)
    x_ol = np.array(x_ol).reshape(mpc_i + 1 , nx )
    u_cl = np.array(u_cl).reshape(mpc_i , nu)
  
    
    return x_ol, u_cl, t, cost_n 

Ts = 0.01
sim_time = 3
x_ol, u_cl, t, cost_nn  = run_closed_loop_mpc(x0, Ts, sim_time, pisolver)