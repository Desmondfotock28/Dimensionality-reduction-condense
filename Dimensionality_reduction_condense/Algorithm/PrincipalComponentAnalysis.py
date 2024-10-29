import numpy as np

class PrincipalComponentAnalysis:
    def __init__(self, threshold=0.02):
        """
        Initialize with:
        U_opt: Optimal control trajectories from MPC (list of arrays).
        J_cl: Baseline closed-loop performance metric.
        threshold: Minimum variance percentage (default is 2% or 0.02).
        """
        self.threshold = threshold
        self.nv_new = None  # New active subspace dimension
        
    def compute_sensitivity_matrix(self,U_opt):
        """Compute the sensitivity matrix S (U_hat)."""
        m = len(U_opt[0])
        U_hat = np.zeros((m, m))

        for u in U_opt:
            U_hat += u @ u.T

        S = U_hat / len(U_opt) 

        return S    
    
    def compute_PCA(self,S):
        """Compute eigenvalues and eigenvectors of the sensitivity matrix S."""
    
        # Eigen decomposition of S
        eig_vals, eig_vecs = np.linalg.eig(S)

        # Sort eigenvalues and eigenvectors by decreasing eigenvalue
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

        # Calculate cumulative variance contribution
        eigv_sum = sum(eig_vals)
        variance = []
        for i,j in enumerate(eig_pairs):
            variance.append((j[0]/eigv_sum).real)
        
        variance = np.array(variance)
        counter =0
        # Determine nv_new: the number of eigenvalues with at least 2% variance
        for var in variance:
            if var>= self.threshold:
                counter+=1
            else:
                pass
        self.nv_new = counter

        #self.nv_new = np.argmax(variance >= self.threshold) + 1
        return  eig_vecs 
    
    def compute_active_subspace(self,S):
        eig_vecs  = self.compute_PCA(S)
        
        # Select the top-nv_new eigenvectors
        W = np.hstack([eig_vecs[:, i].reshape(-1, 1) for i in range(self.nv_new)])

        return W ,self.nv_new
