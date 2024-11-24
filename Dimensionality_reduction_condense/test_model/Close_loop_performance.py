import numpy as np
from typing import Callable
from casadi import *
import matplotlib.pyplot as plt
import sympy as sp



def compute_block_matrix( Q_base, R_base, N):
    # Compute the solution to the discrete-time algebraic Riccati equation using control library

    
    # Construct the block diagonal matrix Q
    Q_blocks = [Q_base] * N + [Q_base]

    Q_d = np.block([[Q_blocks[i] if i == j else np.zeros_like(Q_base) for j in range(N+1)] for i in range(N+1)])
    
    # Construct the block diagonal matrix R
    R_d = np.block([[R_base if i == j else np.zeros_like(R_base) for j in range(N)] for i in range(N)])

    return  Q_d ,  R_d

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
        x_k += Ts*system(x_k, U[i-1,:])
        g.append(x_k)
    x_N = x_k + Ts*system(x_k, U[N-1,:] )
    g.append(x_N)
    return g


#Controller frequency and Prediction horizon
Ts = 0.01    #sampling time in [s]

N =  300     #prediction horizon

tf= 3

nx= 4       #state dimension 

nu= 1       #input dimension

# System states and controls 
x = SX.sym('x', nx);    # states of the system 

u = SX.sym('u', nu);    # control of the system

dx = catPole_ode(x, u)

# Create the CasADi function
system = Function("sys", [x, u], [dx])
    
#Objective and Constrains
Q = 1
Q = Q * np.diag([1, 1, 0.1, 0.1])

R = 1
R = R * np.diag([0.001])

Q_d , R_d= compute_block_matrix( Q, R, N)

#symbolic variable for g_vec 
g_d = SX.sym("g_d", (N+1)*nx)

u_c = SX.sym("u_x",N*nu)

obj = g_d.T @ Q_d @ g_d + u_c.T @ R_d@ u_c 

# Create the CasADi function
objective = Function("J", [g_d, u_c], [obj])


def Compute_close_loop(X,U):
    J = objective(X,U) 
    return J

def comput_close_loop_Performance(X_init, U_cl):
    Delta = 0
    for k in range(len(X_init)):
        x0 = X_init[k]
        X = g_vec(x0 , U_cl[k] , N , system ,Ts)
        X = vertcat(*X)
        U_cl_k = vertcat(reshape(U_cl[k], -1,1))
        J_k = objective(X, U_cl_k) 
        Delta += J_k 
    Delta_avg = (1/len(X_init))* Delta
    return Delta_avg

X_init = np.load('X_init_test.npy')
U_cl  = np.load('U_cl_test.npy')

#test calculation 
U_cl_p  = np.load('U_cl_a_test30_1.npy')
#U_cl_p1  = np.load('U_cl_a_test.npy')
Delta_ref = comput_close_loop_Performance(X_init, U_cl_p )
print(Delta_ref)


"""
print(U_cl[0].shape)
x0 = X_init[0]
X = g_vec(x0 , U_cl[0] , N , system ,Ts)
X = vertcat(*X)
print(X.shape)
U_cl_0 = vertcat(reshape(U_cl[0], -1,1))

J = objective(X, U_cl_0) 

print(J)
"""