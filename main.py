import pandas as pd
import numpy as np

# Step 1: Load the transition matrix from CSV
df = pd.read_csv("mse431matrix.csv", header=None)  # Replace with your file path

# Step 2: Convert DataFrame to NumPy array
P = df.to_numpy()

# Step 3: Steady-state function
def compute_steady_state(P):
    P = np.array(P)
    n = P.shape[0]
    A = P.T - np.eye(n)
    A = np.vstack([A, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1
    steady_state, *_ = np.linalg.lstsq(A, b, rcond=None)
    return steady_state

# Step 4: Compute steady state
steady_state = compute_steady_state(P)

# Step 5: Create a DataFrame for results
result_df = pd.DataFrame({
    'State': range(1, len(steady_state) + 1),
    'Probability': steady_state
})

print(result_df["Probability"])
result_df.to_csv("steady_state_probabilities.csv", index=False)  # Save to CSV file