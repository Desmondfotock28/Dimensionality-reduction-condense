# Dimensionality Reduction  Reinforcement Learning 

## Introduction

MPC algorithms must address the exponential growth of states and control variables when determining optimal control in high-dimensional spaces. Over the years, many approaches have been developed to reduce the computational complexity of the full problem while maintaining control performance. These include the move block strategy [1], SVD decomposition [2], orthonormal basis functions [3], and sensitivity-based methods such as the Active Subspaces Method [4].

Among them, NMPC in active subspaces presents a promising approach by projecting the optimization problem into a lower-dimensional input space while maintaining recursive feasibility
and stability. This approach depends on certain projector matrices to solve the optimization problem in the reduced-dimensional subspaces. However, there is no underlying concept for
selecting these projector matrices in order to achieve closed-loop optimality.

This framework involves using Reinforcement Learning (RL) with Principal Component Analysis(PCA) to tune the parameter $T_1$ for the  Nonlinear Model Predictive Control (NMPC) in Active Subspaces (Dimensionality Reduction with Recursive Feasibility Guarantees) algorithm [4]. Here:

- $T_1$ is the projector of the active subspace.

The idea of integrating RL with MPC was first proposed in [5] and has demonstrated effectiveness in various applications using different learning algorithms and sound theory, as seen in [6], [7], [8], and [9]. This framework merges two powerful control techniques into a single data-driven approach:

- **Model Predictive Control (MPC):** A well-known control methodology that uses a prediction model to forecast the future behavior of the environment and compute the optimal action.

- **Reinforcement Learning (RL):** A machine learning paradigm that has achieved significant success in recent years (e.g., games such as Chess and Go) and is highly adaptable to unknown and complex-to-model environments.
  
The figure below illustrates the main concept behind this learning-based control approach. The MPC controller, parametrized with $T_1$ in its objective, predictive model, and constraints, serves a dual purpose: acting as a policy provider (i.e., generating an action for the environment based on the current state) and as a function approximator for the state and action value functions (i.e., predicting the expected return when following the current control policy from a given state or state-action pair). PCA is used to determine the optimal active subspace dimension, which is subsequently fine-tuned by the RL algorithm. Simultaneously, the RL algorithm adjusts the MPC parametrization to enhance the controller's closed-loop performance and achieve a (sub)optimal policy.

<div align="center">
  <img src="https://github.com/Desmondfotock28/Dimensionality-Reduction-with-Reinforcement-learning/blob/main/FRAMEWORK.png" alt="mpcrl-diagram" height="300">
</div>
