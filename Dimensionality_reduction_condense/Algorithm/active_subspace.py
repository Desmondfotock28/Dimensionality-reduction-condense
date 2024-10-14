import numpy as np
import matplotlib.pyplot as plt


# Step 3
# Find Eigen Value And Eigen Vector for sensitivity matrix S  
# It's Nm*Nm so we will have Nm eigen values and Nm corresponding vector
N= 100
U_opt = np.load('U_opt1.npy')

def compute_U():
    m = len(U_opt[0])
    U_hat = np.zeros((m, m))

    for i in range(len(U_opt)):

         U_hat += U_opt[i]@U_opt[i].T

    U_hat /= len(U_opt)
    return U_hat

U_hat = compute_U()
print(U_hat.shape)

S =U_hat
eig_vals, eig_vecs = np.linalg.eig(S)


for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(N,1)   
    print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
    print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))

# 4.1. Sorting the eigenvectors by decreasing eigenvalues
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0])

# 4.2. 
# “explained variance” as percentage:
print('Variance explained:\n')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))

# 4.3. 
# Choosing nv eigenvectors with the largest eigenvalues
# After sorting the eigenpairs by decreasing eigenvalues, 
# it is now time to construct our Nm×nv-dimensional eigenvector matrix T1
# here Nm×nv: based on the nv most informative eigenpairs
# and thereby reducing the initial Nm-dimensional feature space into a nv-dimensional feature subspace.

W = np.hstack((eig_pairs[0][1].reshape(N,1), eig_pairs[1][1].reshape(N,1) ,eig_pairs[2][1].reshape(N,1)))
print('Matrix W:\n', W.real)

np.save('dominant_active', W)
# 4.4. 
#compute the activity score 