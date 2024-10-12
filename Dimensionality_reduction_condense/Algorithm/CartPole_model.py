import numpy as np
import math
import matplotlib.pyplot as plt
from gymnasium.spaces.box import Box
from typing import Optional
from gym.error import DependencyNotInstalled
from gym import logger
import gym
from casadi import *

class CartPole(gym.Env):
    """
    Class for the CartPole environment
    ...

    Attributes
    ----------
    observation_space : Box
        State constraint set

    action_space: Box
        Input constraint set

    threshold : float
        Norm of distance from goal state to consider reached

    difficulty: {"easy" , "hard"}
        Set difficulty level by modifying threshold for the problem

    dt : float
        Sampling time

    m : float
        mass

    viewer:
        Figure object for rendering

    goal_state: array
        Target where reward is minimum

    goal_mask: array
        Scaling applied on the reward


    reward_style: {1, -1}
        Cost function if positive, Reward function if negative

    render_fl = {False, True}
        Render flag

    supports_rendering = {True, False}

    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self,  render_mode: Optional[str] = None):
        super().__init__()
        self.gravity = 9.81
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.8 # actually half the pole's length
        self.tau = 0.01  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 2*math.pi            #12 * 2 * math.pi / 360
        self.x_threshold = 10.0
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.

        # State and action space
        self.state_space = Box(
            low=np.array([-np.finfo(np.float32).max, -10, -np.finfo(np.float32).max, -np.finfo(np.float32).max]),
            high=np.array([np.finfo(np.float32).max,  10 , np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32),
        )

        self.observation_space = Box(
            low=np.array([-np.finfo(np.float32).max, -10, -np.finfo(np.float32).max, -np.finfo(np.float32).max]),
            high=np.array([np.finfo(np.float32).max,  10 , np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32),
        )

        self.action_space = Box(low=-80.0, high=80.0, shape=(1,), dtype=np.float32)
        self.action_dim = self.action_space.shape[0]
        self.obs_dim = self.observation_space.shape[0]
        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.state = None

        self.steps_beyond_terminated = None
        self.np_random, _ = gym.utils.seeding.np_random(None)

        # Goal definitions
        self.goal_state = np.array([0.0, 0.0, 0.0, 0.0])
        self.goal_mask_s = np.array([1e3, 1e3, 1e-2, 1e-2])
        self.goal_mask_a = np.array([1e-1])
        self._W = 2*np.diag(self.goal_mask_s)
        self._R = 2*np.diag(self.goal_mask_a)
        self.viewer = None

         # Initiate state
        self.reset()

    def get_obs(self, obs):
        return obs
    
    def get_initial_guess(self,  u0, Pf , P_learn, N , nv):
        """
        Generates initial guess for solver

        Parameters
        ----------
        N : int
            Prediction horizon
        """
        T1_0 =  P_learn
        T1_0 = reshape(T1_0, N*self.action_dim,nv )
        T2_0 = Pf[2*self.obs_dim + self.action_dim + (N * self.action_dim - nv): ]
        T2_0 = reshape(T2_0,N*self.action_dim,(N *self.action_dim - nv))
        w_0  = Pf[2*self.obs_dim + self.action_dim :(N * self.action_dim - nv)+ 2*self.obs_dim + self.action_dim]
        w_0 =  reshape(w_0, (N * self.action_dim - nv ), 1)
        mu = 1
        u_st_0 = np.tile(u0, (N, 1))
        u_st_0  = u_st_0.reshape(-1,1)
        #V_0 = T1_0.T@(u_st_0 -mtimes(T2_0,w_0))
        V_0 = np.zeros(nv)
        v_st_0 = np.tile(V_0 , (1, 1))
        mu_st_0 = np.tile(mu, (1, 1))
        # Reshape to column vectors if necessary
        v_st_0 = v_st_0.reshape(-1, 1)
        mu_st_0 = mu_st_0.reshape(-1, 1)

        X0 = np.concatenate((v_st_0, mu_st_0), axis=0)
        return X0
    
    def step(self, action):

        #err_msg = f"{action!r} ({type(action)}) invalid"
        #assert self.action_space.contains(action), err_msg
        #assert self.state is not None, "Call reset before using step method."
            # Define the ODEs
        
        self.state_prev = self.state.copy()
        theta, x,  theta_dot, x_dot = self.state
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        d11 = self.length*(self.masspole*sintheta*sintheta+ self.masscart)
        d22 =  self.masscart*sintheta*sintheta+ self.masscart
        thetaacc = (1/d11)*((self.total_mass)*self.gravity*sintheta-self.masspole*self.length*theta_dot**2*sintheta*costheta - costheta*action)
        xacc = (1/d22)*(-self.masspole*self.gravity*self.length*costheta*sintheta + self.masspole*self.length*theta_dot**2*sintheta + action)

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc

        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = vertcat(theta, x, theta_dot, x_dot)
        self.state = np.array(self.state)
        self.obs = self.state.copy()
        rew = self.reward_fn(self.state_prev, action)
        info = ""
        return (self.state, self.obs, rew, info)
    
    def reward_fn(self, state, action):
        """
        Compute reward for one timestep

        """
        state = state if state is not None else self.state.copy()
        r = state.T @ self._W @ state + self._R*action**2  #Need to defined reward style
        return r
    
    def _parse_env_params(
        self,
        start_loc=[np.pi, 1.0, 0.0, 0.0],
        init_std=0.2,
        render=False,
    ):
        """Parse simulation parameters"""
        self.start_loc = start_loc
        self.init_std = init_std
        self.render_fl = render


    def ode(self, x, u):
         #(state space )= (x1=theta ,  x2 = z , x3= theta_dot, x4 = z_dot)
        m, M, l, g = self.masspole, self.masscart,self.length,self.gravity
        d11 = l*(m*sin(x[0])*sin(x[0])+ M)
        d22 =  m*sin(x[0])*sin(x[0])+ M
    
    # Define the ODEs
        dx1 = x[2]
        dx2 = x[3]

        dx3 =(1/d11)*((m + M)*g*sin(x[0])-m*l*x[2]**2*sin(x[0])*cos(x[0]) - cos(x[0])*u)

        dx4 =(1/d22)*(-m*g*l*cos(x[0])*sin(x[0]) + m*l*x[2]**2*sin(x[0]) + u)

        dx = vertcat(dx1, dx2, dx3, dx4)
        return dx
    
    def get_model(self):
        # Symabolic defn
        x = SX.sym("x", self.obs_dim)
        u = SX.sym("u", self.action_dim)

        # Model of system dynamics
        model_dyn = Function(
            "model_dyn",
            [x, u],
            [x + self.tau*self.ode(x, u)],
        )
        return model_dyn
  

    def reset(self,seed = None):
        """
        Resets the state of the system and the noise generator

        Returns the state of the system

        """
        self.state =  np.array([np.pi, 1, 0, 0])

        if seed is not None:
            np.random.seed(seed)

        self.state += np.random.uniform(low=-0.05, high=0.05, size=(4,)) # Adding small noise
        self.state = self.state.clip(
            self.observation_space.low, self.observation_space.high
        )
        self.state_prev = self.state.copy()
        self.obs = self.state.copy()

        if self.render_mode == "human":
            self.render()
        return self.state, self.obs
    
    def render(self):
        if self.render_mode is None:
            logger.warning(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (4 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        theta, x, _ , _ = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]

            # Clamp values for rendering
        cartx = np.clip(cartx, 0, self.screen_width)
        carty = np.clip(carty, 0, self.screen_height)

        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-theta)
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        
        # Clamp pole coordinates
        pole_coords = [(np.clip(c[0], 0, self.screen_width), np.clip(c[1], 0, self.screen_height)) for c in pole_coords]
        
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.screen = None




