import autograd.numpy as np
import pymanopt
from pymanopt.manifolds import Grassmann
from pymanopt.optimizers import TrustRegions

SUPPORTED_BACKENDS = ("autograd", "numpy")


def create_cost_and_derivatives(manifold, matrix, backend):
    euclidean_gradient = euclidean_hessian = None

    if backend == "autograd":
        @pymanopt.function.autograd(manifold)
        def cost(X):
            return -np.trace(X.T @ matrix @ X)

    elif backend == "numpy":
        @pymanopt.function.numpy(manifold)
        def cost(X):
            return -np.trace(X.T @ matrix @ X)

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(X):
            return -2 * matrix @ X

        @pymanopt.function.numpy(manifold)
        def euclidean_hessian(X, H):
            return -2 * matrix @ H

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, euclidean_gradient, euclidean_hessian


def run(nv, backend=SUPPORTED_BACKENDS[0], matrix=None, quiet=True):
    subspace_dimension = nv
    if matrix is None:
        num_rows = 128
        matrix = np.random.normal(size=(num_rows, num_rows))
        matrix = 0.5 * (matrix + matrix.T)  # Symmetrize the matrix
    else:
        num_rows = matrix.shape[0]

    # Define the Grassmann manifold and create the cost function
    manifold = Grassmann(num_rows, subspace_dimension)
    cost, euclidean_gradient, euclidean_hessian = create_cost_and_derivatives(
        manifold, matrix, backend
    )
    
    # Setup optimization problem
    problem = pymanopt.Problem(
        manifold,
        cost,
        euclidean_gradient=euclidean_gradient,
        euclidean_hessian=euclidean_hessian,
    )
    
    # Run the optimizer to estimate the dominant subspace
    optimizer = TrustRegions(verbosity=2 * int(not quiet))
    estimated_spanning_set = optimizer.run(problem).point

    # Compute the true dominant subspace from eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    column_indices = np.argsort(eigenvalues)[-subspace_dimension:]  # Indices of largest eigenvalues
    true_spanning_set = eigenvectors[:, column_indices]

    # Print geodesic distance between true and estimated subspace
    geodesic_distance = manifold.dist(true_spanning_set, estimated_spanning_set)
    print("Geodesic distance between true and estimated dominant subspace:", geodesic_distance)

    # Print the estimated and true dominant eigenvectors
    #print("\nEstimated dominant subspace (from optimization):\n", estimated_spanning_set)
    #print("\nTrue dominant eigenvectors (from eigenvalue decomposition):\n", true_spanning_set)
    print(eigenvalues[0:3])
    return estimated_spanning_set, true_spanning_set, geodesic_distance


# Load the matrix from a file and run the algorithm
matrix = np.load('S.npy')
nv = 3
estimated_subspace, true_subspace, distance = run(nv, backend=SUPPORTED_BACKENDS[0], matrix=matrix, quiet=False)

np.save('dominant_active', estimated_subspace)

