# Hasan F. Durkaya
# 504241526

import numpy as np

# Data Generation.
np.random.seed(None)  #None seed form random seed

sequence_number = 10  # Number of sequences
true_p_A, true_p_B = 0.7, 0.4  # True probabilities for coins A and B
true_pi_A, true_pi_B = 0.6, 0.4  # Unequal probabilities of choosing each coin

# Conf flip sequences
# Initialize empty list to store all sequences
sequences = []

for _ in range(sequence_number):
    
    coin = np.random.choice(['A', 'B'], p=[true_pi_A, true_pi_B])           # Randomly select either coin A or B with equal probability
    p = true_p_A if coin == 'A' else true_p_B                   # Set probability based on which coin was selected
    flips = np.random.choice(['H', 'T'], size = 10, p=[p, 1 - p])            # Generate sequence of H/T flips using the selected coin's probability
    sequences.append(flips)


# Generated Sequences.
print("Generated Sequences:")
for i, sequence in enumerate(sequences):
    print(f"Sequence {i + 1}: {''.join(sequence)}")

# EM algorithm
# Initialize parameters randomly
p_A, p_B = np.random.rand(), np.random.rand()   # Random initial probabilities for coins A and B
pi_A = np.random.rand()                         # Random initial probability of choosing coin A
pi_B = 1 - pi_A                                 # Probability of choosing coin B (must sum to 1)

max_iter = 100                  # Maximum number of iterations.
tolerance = 1e-6                # Convergence tolerance.

for iteration in range(max_iter):
    # E-step: Compute responsibilities (expected contributions)
    heads_A, tails_A, heads_B, tails_B = 0, 0, 0, 0
    total_r_A, total_r_B = 0, 0

    for sequence in sequences:
        
        n_heads = np.sum(sequence == 'H')    #Number of heads in the sequence.
        n_tails = len(sequence) - n_heads    #Number of tails in the sequence.

        # Probabilities of the sequence given each coin
        prob_A = (p_A**n_heads) * ((1 - p_A)**n_tails)
        prob_B = (p_B**n_heads) * ((1 - p_B)**n_tails)
        weighted_prob_A = pi_A * prob_A
        weighted_prob_B = pi_B * prob_B
        total_prob = weighted_prob_A + weighted_prob_B

         # Responsibilities calculated.
        r_A = weighted_prob_A / total_prob
        r_B = weighted_prob_B / total_prob

        # Accumulating weighted counts.
        heads_A += r_A * n_heads
        tails_A += r_A * n_tails
        heads_B += r_B * n_heads
        tails_B += r_B * n_tails

        total_r_A += r_A
        total_r_B += r_B

    #M-step: Updating parameters.
    new_p_A = heads_A / (heads_A + tails_A)
    new_p_B = heads_B / (heads_B + tails_B)
    new_pi_A = total_r_A / sequence_number
    new_pi_B = 1 - new_pi_A

    # Checking for convergence
    if (
        abs(new_p_A - p_A) < tolerance
        and abs(new_p_B - p_B) < tolerance
        and abs(new_pi_A - pi_A) < tolerance
    ):
        break

    p_A, p_B, pi_A, pi_B = new_p_A, new_p_B, new_pi_A, new_pi_B     # Updating parameters for next iteration.

# Display estimated parameters
print("\nEstimated Parameters:")
print(f"p_A: {p_A:.4f}")
print(f"p_B: {p_B:.4f}")
print(f"pi_A: {pi_A:.4f}")
print(f"pi_B: {pi_B:.4f}")
