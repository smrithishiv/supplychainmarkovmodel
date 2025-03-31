import pandas as pd
import numpy as np

# Step 1: Load the transition matrix from CSV with headers
df = pd.read_csv("outputs/initial_model.csv", index_col=0)  # First column is treated as index

# Step 2: Convert DataFrame to NumPy array (excluding headers)
P = df.to_numpy()

# Step 3: Steady-state function
def compute_steady_state(P):
    P = np.array(P)
    n = P.shape[0]
    A = P.T - np.eye(n)  # Transpose P and subtract identity matrix
    A = np.vstack([A, np.ones(n)])  # Append row of ones to ensure sum of probabilities = 1
    b = np.zeros(n + 1)
    b[-1] = 1  # Right-hand side constraint for probability sum
    steady_state, *_ = np.linalg.lstsq(A, b, rcond=None)  # Solve least squares system
    return steady_state

# Step 4: Compute steady state
steady_state = compute_steady_state(P)

# Step 5: Create a DataFrame for results
result_df = pd.DataFrame({
    'State': df.index,  # Preserve state labels from original CSV
    'Probability': steady_state
})

print(result_df["Probability"])
result_df.to_csv("outputs/steady_state_probabilities.csv", index=False)  # Save to CSV file