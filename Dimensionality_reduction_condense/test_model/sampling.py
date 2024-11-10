import numpy as np

# Parameters
num_samples = 200
threshold = 0.1  # Define the distance threshold

# Initialize an empty list to store the candidate states
candidate_states = []

# Function to check if a new state is within the threshold of any existing state
def is_too_close(new_state, existing_states, threshold):
    for state in existing_states:
        if np.linalg.norm(new_state - state) < threshold:
            return True
    return False

# Sampling loop with uniqueness check
while len(candidate_states) < num_samples:
    new_state = np.array([
        np.random.uniform(low=0, high=np.pi),  # X[0]
        np.random.uniform(low=0, high=3),      # X[1]
        0,                                     # X[2]
        0                                      # X[3]
    ])
    
    # Check if the new state is too close to any existing state
    if not is_too_close(new_state, candidate_states, threshold):
        candidate_states.append(new_state)

# Convert the list to a NumPy array for consistency
candidate_states = np.array(candidate_states)
np.save('train_state.npy',candidate_states)
print("Unique sampled candidate states:\n", candidate_states)