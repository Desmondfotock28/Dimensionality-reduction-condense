import numpy as np
from typing import Callable
from casadi import *
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from control import dare


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
    plt.ylabel('$\\theta(t)$ [rad]', fontsize=12)
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(t, xSSp[1] * np.ones_like(t), linewidth=linewidth, label='Reference')
    plt.plot(t, x[:,1], linewidth=linewidth, label='Trajectory')
    plt.grid(True)
    plt.xlabel('$t$ [s]', fontsize=12)
    plt.ylabel('$x(t)$ [m]', fontsize=12)
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(t, xSSp[2] * np.ones_like(t), linewidth=linewidth, label='Reference')
    plt.plot(t, x[:,2], linewidth=linewidth, label='Trajectory')
    plt.grid(True)
    plt.xlabel('$t$ [s]', fontsize=12)
    plt.ylabel('$\dot{\\theta}(t)$ [rad/s]', fontsize=12)
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(t, xSSp[3] * np.ones_like(t), linewidth=linewidth, label='Reference')
    plt.plot(t, x[:,3], linewidth=linewidth, label='Trajectory')
    plt.grid(True)
    plt.xlabel('$t$ [s]', fontsize=12)
    plt.ylabel('$\dot{x}(t)$ [m/s]', fontsize=12)
    plt.legend()

    plt.tight_layout()

    # Plot control inputs
    plt.figure(fignum[1])

    plt.plot(t, uSSp * np.ones_like(t), linewidth=linewidth, label='Reference')
    plt.step(t[:-1], u, linewidth=linewidth, where='post', label='Trajectory')
    plt.grid(True)
    plt.xlabel('$t$ [s]', fontsize=12)
    plt.ylabel('$u^\star(t)$ [N]', fontsize=12)
    plt.legend()

    plt.tight_layout()
    plt.show()

def select_active_inactive_subspaces_1(singular_values, singular_vectors, percentage=0.4):
    """
    Selects the singular vectors that span the active and inactive subspaces of the Hessian matrix H.
    
    Parameters:
    singular_values (numpy.ndarray): Singular values of H
    singular_vectors (numpy.ndarray): Singular vectors of H
    percentage (float): Percentage of singular values to include in the active subspace
    
    Returns:
    active_subspace (numpy.ndarray): Singular vectors that span the active subspace
    inactive_subspace (numpy.ndarray): Singular vectors that span the inactive subspace
    """
    # Determine the number of singular values to include based on the percentage
    num_singular_values = len(singular_values)
    num_active_vectors = int(np.ceil(percentage * num_singular_values))
    
    # Select the singular vectors corresponding to the largest singular values for the active subspace
    active_subspace = singular_vectors[:, :num_active_vectors]
    
    # Select the singular vectors corresponding to the smallest singular values for the inactive subspace
    inactive_subspace = singular_vectors[:, num_active_vectors:]
    
    return active_subspace, inactive_subspace

def compute_svd(H):
    """
    Computes the singular values and right singular vectors of the Hessian matrix H.
    
    Parameters:
    H (numpy.ndarray): Hessian matrix of shape (N*m, N*m)
    
    Returns:
    singular_values (numpy.ndarray): Singular values of H
    right_singular_vectors (numpy.ndarray): Right singular vectors of H
    """
    # Perform Singular Value Decomposition
    U, singular_values, Vt = np.linalg.svd(H)
    
    # For a symmetric matrix H, the right singular vectors are the columns of V (or rows of Vt)
    right_singular_vectors = Vt.T
    
    return U, singular_values, right_singular_vectors


def compute_matrices(A, B, N):
    """
    Computes the matrices Λ and Γ for the given LTI system.
    
    Parameters:
    A (numpy.ndarray): State transition matrix of shape (n, n)
    B (numpy.ndarray): Input matrix of shape (n, m)
    N (int): Horizon length
    
    Returns:
    Λ (numpy.ndarray): Matrix of shape (N*n, n)
    Γ (numpy.ndarray): Matrix of shape (N*n, N*m)
    """
    n = A.shape[0]
    m = B.shape[1]
    
    # Initialize Λ
    Λ = np.zeros((N * n, n))
    
    # Fill in Λ
    for i in range(N):
        Λ[i*n:(i+1)*n, :] = np.linalg.matrix_power(A, i+1)
    
    # Initialize Γ
    Γ = np.zeros((N * n, N * m))
    
    # Fill in Γ
    for i in range(N):
        for j in range(i+1):
            Γ[i*n:(i+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A, i-j) @ B
            
    return Λ, Γ

def compute_block_matrix( Q_base, R_base, N):
    # Compute the solution to the discrete-time algebraic Riccati equation using control library

    
    # Construct the block diagonal matrix Q
    Q_blocks = [Q_base] * N + [Q]

    Q_d = np.block([[Q_blocks[i] if i == j else np.zeros_like(Q_base) for j in range(N+1)] for i in range(N+1)])
    
    # Construct the block diagonal matrix R
    R_d = np.block([[R_base if i == j else np.zeros_like(R_base) for j in range(N)] for i in range(N)])

    return  Q_d ,  R_d



def compute_hessian(A, B, Q_base, R_base, N):
    """
    Computes the Hessian matrix H for the given LTI system and cost function.
    
    Parameters:
    A (numpy.ndarray): State transition matrix of shape (n, n)
    B (numpy.ndarray): Input matrix of shape (n, m)
    Q_base (numpy.ndarray): State cost matrix of shape (n, n)
    R_base (numpy.ndarray): Input cost matrix of shape (m, m)
    N (int): Horizon length
    
    Returns:
    H (numpy.ndarray): Hessian matrix of shape (N*m, N*m)
    """
    
    # Compute the solution to the discrete-time algebraic Riccati equation using control library
    P = dare(A, B, Q_base, R_base)[0]
    
    # Construct the block diagonal matrix Q
    Q_blocks = [Q_base] * (N - 1) + [Q]
    Q_d = np.block([[Q_blocks[i] if i == j else np.zeros_like(Q_base) for j in range(N)] for i in range(N)])
    
    # Construct the block diagonal matrix R
    R_d = np.block([[R_base if i == j else np.zeros_like(R_base) for j in range(N)] for i in range(N)])
    
    # Compute the matrix Γ
    _, Γ = compute_matrices(A, B, N)
    
    # Compute the Hessian matrix H
    H = R_d + Γ.T @ Q_d @ Γ
    
    return H 

def CartPole_parameters() -> dict:
    """Define system parameters for the two-link planar arm robot."""
    params = {
        'm': 0.1,      # kg
        'M': 1,      # kg
        'l': 0.8,      # m
        'g': 9.81     # N/kg
    }
    return params

def catPole_ode(x, u):

    params = CartPole_parameters()
    m, M, l, g = params.values()
    
    #(state space )= (x1=theta ,  x2 = z , x3= theta_dot, x4 = z_dot)
    d11 = l*(m*sin(x[0])*sin(x[0])+ M)
    d22 =  m*sin(x[0])*sin(x[0])+ M
    
    # Define the ODEs
    dx1 = x[2]
    dx2 = x[3]

    dx3 =(1/d11)*((m + M)*g*sin(x[0])-m*l*x[2]**2*sin(x[0])*cos(x[0]) - cos(x[0])*u)

    dx4 =(1/d22)*(-m*g*l*cos(x[0])*sin(x[0]) + m*l*x[2]**2*sin(x[0]) + u)

    dx = vertcat(dx1, dx2, dx3, dx4)
    return dx

# Defind condense vector 

def g_vec(x0 , U , N , system ,Ts):
    g=[x0]
    x_k = x0
    for i in range(1,N):
        x_k += Ts*system(x_k, U[:,i-1])
        g.append(x_k)
    x_N = x_k + Ts*system(x_k, U[:,N-1] )
    g.append(x_N)
    return g


def stack_gain_matrix(N,K):
    # Create identity matrices
    nu,nx = K.shape
    # Stack matrices along the diagonal
    block_matrix = np.zeros((nu * N, nx * N))
    for i in range(N):
        block_matrix[i*nu:(i+1)*nu, i*nx:(i+1)*nx] = K
    return block_matrix


#Controller frequency and Prediction horizon
Ts = 0.01    #sampling time in [s]

N =  100     #prediction horizon

tf= 2

nx= 4       #state dimension 

nu= 1       #input dimension

# System states and controls 
x = SX.sym('x', nx);    # states of the system 

u = SX.sym('u', nu);    # control of the system

dx = catPole_ode(x, u)

# Create the CasADi function
system = Function("sys", [x, u], [dx])

# Declear empty sys matrices
U = SX.sym('U',nu,N)               # Decision variables (controls)

#Parameters:initial state(x0)

P_a = SX.sym('P_a',nx,1)
    
#Objective and Constrains
Q = 2
Q = Q * np.diag([1e3, 1e3, 1e-2, 1e-2])

R = 2
R = R * np.diag([1e-1])


# Define the stage cost and terminal cost
m = 1 # mass of pendulum (kg)
M = 1  # mass of cart (kg)
g = 9.81  #  acceleration due to gravity m/s^2
l = 0.8    # length of pendulum 
# continuos-time Linearise system matrices 

A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [((m + M)*g)/(l*M), 0, 0, 0], [(-m*g)/M, 0, 0, 0]])
B = np.array([[0], [0], [-1/(l*M)], [1/M]])

# Discretization parameters
dt = 0.01

# Discrete-time system matrices using matrix exponential
A_d = np.eye(4) + dt * A
B_d = dt * B

# Terminal cost (solution to Riccati equation)
P, L, K = dare(A_d, B_d, Q, R)

K = -np.array(K)
P =  np.array(P)
# Closed-loop system
A_cl = A_d + B_d @ K
# Eigenvalues of close loop
print(np.linalg.eig(A_cl)[0])

Q_d , R_d= compute_block_matrix( Q, R, N)

G = g_vec(P_a , U , N , system ,Ts)

X =[]

for i in range(len(G)):
    X.append(G[i][1:2])

X = vertcat(*X)
G = vertcat(*G)
print(G.shape)
U = vertcat(reshape(U, -1,1))

#symbolic variable for g_vec 
g_d = SX.sym("g_d", (N+1)*nx)

# symbolic varaible for control input vector 

u_c = SX.sym("u_x",N*nu)

obj = g_d.T @ Q_d @ g_d + u_c.T @ R_d@ u_c 

# Create the CasADi function
objective = Function("J", [g_d, u_c], [obj])

# State constraints
lb_x = np.array([-np.finfo(np.float32).max,  -10, -np.finfo(np.float32).max, -np.finfo(np.float32).max])
ub_x = np.array([  np.finfo(np.float32).max,  10,  np.finfo(np.float32).max, np.finfo(np.float32).max])

lbx = [lb_x[1:2]]*(N + 1) 
ubx = [ub_x[1:2]]*(N + 1)
lbx = vertcat(*lbx)
ubx = vertcat (*ubx)

# Input constraints
lb_u = -80 
ub_u = 80

lbu = [lb_u]*N
ubu = [ub_u]*N
ubu = vertcat(*ubu)
lbu = vertcat (*lbu)

Opt_Vars = U

def objective_cost():
    J = objective(G,U) 
    return J

def inequality_constraints():
  
    hu = []   # Box constraints on active inputs
    hx = []   # Box constraints on states
    
    hu.append(lbu-U )
    hu.append(U - ubu)
    hx.append(lbx-X)
    hx.append(X - ubx)
    return  hu, hx

def Pi_opt_formulation():
    J = objective_cost()
    hu, hx = inequality_constraints()
    Hu = vertcat(*hu)
    Hx = vertcat(*hx)
    G_vcsd = vertcat(*hx, *hu)
    lbg =  [-np.inf] * (Hx.shape[0] + Hu.shape[0])
    ubg =  [0] * (Hx.shape[0] + Hu.shape[0])
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

# Define the Lagrangian
lam_g = SX.sym('lam_g', G_vcsd.size1())

Lagrangian = objective_cost() + dot(lam_g, G_vcsd)

# Compute the Hessian of the Lagrangian
Hessian_Lagrangian = hessian(Lagrangian, U)[0]

# Define a function to compute the Hessian at each optimal solution
hessian_func = Function('hessian_func', [Opt_Vars, lam_g , P_a], [Hessian_Lagrangian])

def run_open_loop_mpc(x0, u0 , solver ):
      # Initial control inputs and state
    u_st_0 = np.tile(u0, (N, 1))

    args_p =  np.array(
            [[np.pi, 1, 0, 0] ]
        )
    
    args_p= vertcat(*args_p)
    args_x0 = u_st_0.T.reshape(-1)
   # Solve the optimization problem
    sol = solver(x0=args_x0, p=args_p, lbg=lbg_vcsd, ubg=ubg_vcsd)
    usol = sol['x']

    # Extract the control inputs from the solution
    u = np.array(sol['x']).reshape((nu , N))
   # construct xsol 
    xsol = g_vec(x0 , u , N, system,Ts)
    xsol = vertcat(*xsol)
    xsol = np.array(xsol).reshape((N+1 , nx))
 
    # Convert lists to numpy arrays for easier handling
    return xsol, u ,usol

u0 = 80

x0 = np.array([np.pi, 1, 0, 0])

xsol, u_ol, usol  = run_open_loop_mpc(x0, u0 , pisolver)

xSS=  np.array([0, 0 , 0, 0])
uSS = 0
t_ol = linspace(0,tf , N+1)
plot_results(t_ol, xsol, u_ol.T, xSS, uSS, [1, 2])

#condense run close loop 

def run_closed_loop_mpc(x0, Ts, sim_time, solver):
    x_goal = np.array([0, 0, 0, 0])
    u0 = 80
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
    args_p = np.array([[np.pi, 1, 0, 0]])
    args_p = vertcat(*args_p)
    cost_n = []
    Hessians = []
    while np.linalg.norm(x0 - x_goal, 2) > goal_tolerance and mpc_i < int(sim_time / Ts):

        args_p = x0
        args_x0 =  u_st_0.T.reshape(-1)
        sol = solver(x0=args_x0, p=args_p, lbg=lbg_vcsd, ubg=ubg_vcsd)
        x_opt = sol['x']
        cost_n.append(sol['f'])
        u_0 = x_opt[:nu]
        # Extract the control inputs from the solution
        u = np.array(sol['x']).reshape((nu , N))
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

         # Compute the Hessian at the current optimal solution
        lamb_opt = sol['lam_g']
        hessian_value = hessian_func(sol['x'], lamb_opt, args_p)
        Hessians.append(hessian_value.full())
        
        mpc_i += 1

    x_ol = vertcat(*x_ol)
    x_ol = np.array(x_ol).reshape(mpc_i + 1 , nx )
    u_cl = np.array(u_cl).reshape(mpc_i , nu)
  
    
    return x_ol, u_cl, t, cost_n ,  Hessians

Ts = 0.01
sim_time = 2
x_ol, u_cl, t, cost_nn, Hessians  = run_closed_loop_mpc(x0, Ts, sim_time, pisolver)

cost_n_np = [float(cost.full().flatten()) for cost in cost_nn]
plt.figure(figsize=(10, 5))
plt.plot(cost_n_np, label='Cost_n', marker='o')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Nominal Cost')
plt.legend()
plt.grid(True)
plt.show()


#condense mpc with active subspace //need fixing

#compute eigen decomposition of the Hessian 
def compute_C():
    m = len(Hessians[0])
    C_hat = np.zeros((m, m))

    for i in range(len(Hessians)):

        C_hat += Hessians[i]

    C_hat /= len(Hessians)
    return C_hat

C_hat = compute_C()

np.save('S',C_hat)

def eigen_decomposition(C_hat):
    
    eigenvalues, eigenvectors = np.linalg.eig(C_hat)
    sort_indices = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]
    D = np.diag(eigenvalues)
    return eigenvectors, D

eigenvectors, D = eigen_decomposition(C_hat)

T1 , T2 =select_active_inactive_subspaces_1(D, eigenvectors,  percentage=0.1)

# Load the saved T1 and T2
np.save('T1_G10.npy',T1)
np.save('T2_G10.npy',T2)




nv = T1.shape[1]


# parameter for initial state and parameter w 
nP = nx  +  (N *nu - nv)
P_val = SX.sym("P_val", nP )

#Decision variables 
V = SX.sym("V", nv, 1)

# penalty  variables for inactive subspace w 
mu = SX.sym("mu", 1, 1)
lb_Mu = 0.0
ub_Mu = 1.0

U_a = T1@V-mu*T2@P_val[nx:]

U_a = reshape(U_a,nu, N)

G_a = g_vec(P_val[:nx] , U_a , N , system ,Ts)

X_a =[]

for i in range(len(G_a)):
    X_a.append(G_a[i][1:2])

X_a = vertcat(*X_a)

Opt_Vars_p = vertcat(
            reshape(V, -1, 1),
            reshape(mu, -1, 1),
        )

G_a = vertcat(*G_a)
U_a = vertcat(reshape(U_a, -1,1))

def compute_cost(U,x):
   
    # Compute stage costs
    cost = objective(x,U)

    return cost

def objective_cost_p():

    J = objective(G_a, U_a) 

    return J

def inequality_constraints_p():
    # Constraint list
    hmu = []  # initial input constraints  
    hu = []   # Box constraints on active inputs
    hx = []   # Box constraints on states
    hu.append(lbu- U_a )
    hu.append(U_a - ubu)
    hx.append(lbx-X_a)
    hx.append(X_a - ubx)
    hmu.append(lb_Mu-mu)
    hmu.append(mu-ub_Mu)

    return hmu ,hu, hx

def Pi_opt_formulation_p():
    J = objective_cost_p()
    hmu , hu, hx = inequality_constraints_p()
    Hu = vertcat(*hu)
    Hx = vertcat(*hx)
    Hmu = vertcat(*hmu)
    G_vcsd = vertcat(*hx, *hu ,*hmu)
    lbg =  [-np.inf] * (Hx.shape[0] + Hu.shape[0] + Hmu.shape[0])
    ubg =  [0] * (Hx.shape[0] + Hu.shape[0] + Hmu.shape[0])
    lbg_vcsd_p = vertcat(*lbg)
    ubg_vcsd_p = vertcat(*ubg)

    opts_setting = {
        "ipopt.max_iter": 500,
        "ipopt.print_level": 4,
        "print_time": 1,
        "ipopt.acceptable_tol": 1e-6,
        "ipopt.acceptable_obj_change_tol": 1e-6,
    }
    vnlp_prob = {
        "f": J,
        "x": Opt_Vars_p,
        "p": vertcat(P_val),
        "g": G_vcsd,
    }
    pisolver_p = nlpsol("vsolver", "ipopt", vnlp_prob)

    return lbg_vcsd_p, ubg_vcsd_p,  pisolver_p 

lbg_vcsd_p, ubg_vcsd_p,  pisolver_p = Pi_opt_formulation_p()

def run_closed_loop_activesubspace_mpc(x0, u0, Ts, sim_time, solver ):
    mu = 1
    u_tilda_k  = usol
    w_k  =  mtimes(T2.T, usol )
    t0 = 0
    nx = x0.shape[0]
    t = [t0]
    x_ol = np.zeros((nx, int(sim_time / Ts) + 1))  # Open loop predicted states
    x_ol = [x0]

    # Initialization
    mpc_i = 0
    x_cl = []    # Store predicted states in the closed loop
    u_cl = []    # Store control inputs in the closed loop
    cost_fb = []
    cost_fn = []
    goal_tolerance = 0.01  # Define a goal tolerance
    u_st_0 = np.tile(u0, (N, 1))
    u_st_0  = u_st_0.reshape(-1,1)
    V_0 = T1.T@(u_st_0 -T2@w_k)
   # Initial control inputs
    v_st_0 = np.tile(V_0 , (1, 1))
    mu_st_0 = np.tile(mu, (1, 1))
    
    # Reshape to column vectors if necessary
    v_st_0 = v_st_0.reshape(-1, 1)
    mu_st_0 = mu_st_0.reshape(-1, 1)
    x_goal =np.array([0, 0, 0,0])
    # Concatenate all three into one array
    P_init  = vertcat( 
            reshape(x0, -1, 1), 
            reshape(w_k, -1, 1)
        ) 
    while np.linalg.norm(x0 - x_goal, 2) > goal_tolerance and mpc_i < int(sim_time / Ts):
         
        P_init[:nx] = x0
        P_init[nx:] = w_k
        Opt_Vars_init = np.concatenate(( v_st_0, mu_st_0), axis=0)

        # Solve the optimization problem
        sol = solver(x0=Opt_Vars_init, p=P_init, lbg=lbg_vcsd_p, ubg=ubg_vcsd_p)
        x_opt_p = sol['x']
         # Extract the solution trajectory
        Vsol = np.array(x_opt_p[:nv]).reshape((nv,1))
        musol = np.array(x_opt_p[nv:]).reshape((1,1))
         #Reconstruct optimal control input 
        Usol = T1@Vsol + musol*T2 @w_k
        u = np.array(Usol).reshape((nu, N))
        # Reconstruct predicted trajectory 
       
        x_pred = g_vec(x0 , u , N, system,Ts)
        x_pred = vertcat(*x_pred)
        J_n = sol['f']
        cost_fn.append(J_n)
        x_pred_fb =  g_vec(x0 ,  np.array(u_tilda_k).reshape(nu,N) , N, system,Ts)
        x_pred_fb = vertcat(*x_pred_fb)

         #compute cost associated with u_tilda_k
        J_fb = compute_cost(u_tilda_k,  x_pred )
        cost_fb.append(J_fb)

        #compute cost associated v
        J_n = sol['f']
        cost_fn.append(J_n)
        
        if J_fb <= J_n:
            u_k = u_tilda_k[:nu]
            Xsol = x_pred_fb
            
        else:
            u_k = Usol[:nu]
            Xsol = x_pred
          #construct x_pred 

        x0 = Xsol[nx:2*nx]
        Xsol = np.array(Xsol).reshape((N+1, nx))
        x_cl.append(Xsol)
        
         # Store the first control action
        u_cl.append(u_k)
        # compute optimal control input 
        
        t0 = t0 + Ts
        x_ol.append(x0)   #store calculated state 

        #calculate state feedback 
        u_fb =  mtimes(K,x0)

        #update u_tilda_k 
        u_tilda_k = np.vstack([Usol[nu:], reshape(u_fb, 1, -1)])
        
        u_tilda_k =vertcat(reshape(u_tilda_k, -1, 1))
        #update w_k 
        w_k = T2.T@(u_tilda_k)

        # Prepare the initial condition for the next iteration
        v_st_0 = T1.T@(Usol-T2@w_k)
        mu_st_0 = musol

    #increment mpc counter 
        mpc_i += 1
    # Convert lists to numpy arrays for easier handling

    x_ol = np.array(vertcat(*x_ol)).reshape(mpc_i+1,nx)
    u_cl = np.array(vertcat(*u_cl)).reshape(mpc_i,nu)
    

    return x_ol ,  u_cl ,t, cost_fb, cost_fn

x_ol_p ,  u_cl_p ,t_p ,cost_fb,cost_n = run_closed_loop_activesubspace_mpc(x0, u0, Ts, sim_time, pisolver_p )

cost_n_n = [float(cost.full().flatten()) for cost in cost_nn]
cost_p_np = [float(cost.full().flatten()) for cost in cost_n]
plt.figure(figsize=(10, 5))
plt.plot(cost_n_n, label='Cost_n', marker='o')
plt.plot(cost_p_np, label='Cost_p', marker='x')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Active subspace cost dicrease')
plt.legend()
plt.grid(True)
plt.show()