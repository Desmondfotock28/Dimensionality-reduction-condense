import numpy as np
from control import dare
import sympy as sp




class GainSchedulingController:
    def __init__(self, Q, R, action_bounds=(-80, 80)):
        """
        Initialize the gain scheduling controller with weight matrices and action limits.
        
        Args:
            Q (ndarray): State weight matrix.
            R (ndarray): Control weight matrix.
            action_bounds (tuple): Min and max action values.
        """
        self.Q = Q
        self.R = R
        self.u_min, self.u_max = action_bounds  # Control action bounds
        self.gains = []  # Store gain matrices for each operating point
        self.points = []  # Store corresponding pole angles for interpolation

    def lqr_gain(self, A, B):
        """
        Solve the continuous-time LQR problem to compute the optimal gain matrix K.
        
        Args:
            A (ndarray): Linearized state matrix.
            B (ndarray): Linearized input matrix.
        
        Returns:
            K (ndarray): State feedback gain matrix.
        """
        P, L, K = dare(A, B, self.Q, self.R)
        K = -np.array(K)
        return K

    def add_operating_point(self, A, B, state_point):
        """
        Add a new operating point with its corresponding LQR gain matrix.
        
        Args:
            A (ndarray): Linearized state matrix at the operating point.
            B (ndarray): Linearized input matrix at the operating point.
            state_point (tuple): State vector for the operating point (used for scheduling).
        """
        K = self.lqr_gain(A, B)
        self.gains.append(K)
        self.points.append(state_point[0])  # Use the pole angle θ as the scheduling variable

    def interpolate_gain(self, x1):
        """
        Interpolate between gain matrices based on the pole angle (x1).
        
        Args:
            x1 (float): Current pole angle θ.
        
        Returns:
            K (ndarray): Interpolated gain matrix.
        """
        if x1 <= self.points[0]:
            return self.gains[0]
        elif x1 >= self.points[-1]:
            return self.gains[-1]

        # Perform linear interpolation between nearest points
        for i in range(len(self.points) - 1):
            if self.points[i] <= x1 <= self.points[i + 1]:
                alpha = (x1 - self.points[i]) / (self.points[i + 1] - self.points[i])
                K = (1 - alpha) * self.gains[i] + alpha * self.gains[i + 1]
                return K

    def control_action(self, state):
        """
        Compute the control action using the interpolated gain matrix.
        
        Args:
            state (ndarray): Current state vector [θ, x, θ_dot, x_dot].
        
        Returns:
            u (float): Control action, clipped to the action bounds.
        """
        x1 = state[0]  # Use the pole angle θ for gain scheduling
        K = self.interpolate_gain(x1)
        u = K @ state  # Compute the control input
        return np.clip(u, self.u_min, self.u_max)  # Clip the control to action bound
    
    def Linearise_dynamics(self, xSS, uSS ):
                # Define symbols
        theta, y, theta_dot, y_dot, u = sp.symbols('theta y theta_dot y_dot u')

            # Define constants
        m = 0.1  # mass of pendulum (kg)
        M = 1.0  # mass of cart (kg)
        g = 9.8  # acceleration due to gravity (m/s^2)
        l = 0.5  # length of pendulum (m)

            # Define the dynamics
        d11 = l * (m * sp.sin(theta)**2 + M)
        d22 = m * sp.sin(theta)**2 + M

            # ODEs
        dx1 = theta_dot
        dx2 = y_dot
        dx3 = (1 / d11) * ((m + M) * g * sp.sin(theta) - m * l * theta_dot**2 * sp.sin(theta) * sp.cos(theta) - sp.cos(theta) * u)
        dx4 = (1 / d22) * (-m * g * l * sp.cos(theta) * sp.sin(theta) + m * l * theta_dot**2 * sp.sin(theta) + u)

            # State vector
        dx = sp.Matrix([dx1, dx2, dx3, dx4])
        # Compute the Jacobians
        A = dx.jacobian([theta, y, theta_dot, y_dot])
        B = dx.jacobian([u])

        # Operating point
        op_point = {theta:xSS[0], y: xSS[1], theta_dot: xSS[2], y_dot: xSS[3], u: uSS}

        # Evaluate at the operating point
        A_evaluated = A.subs(op_point).evalf()
        B_evaluated = B.subs(op_point).evalf()
        return A_evaluated, B_evaluated



