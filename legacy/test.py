from itertools import product

# Define the possible values for each position in the tuple
possible_values = [-0.1, 0, 0.1]

# Generate all possible combinations using itertools.product
actions_combinations = product(possible_values, repeat=6)

# Create the dictionary with actions as keys and None as values
actions_dict = {idx: action for idx, action in enumerate(actions_combinations)}

# Check the length of the dictionary to verify it contains 729 entries
print(len(actions_dict))  # Output should be 729

# Printing a sample entry to verify
print(actions_dict[729])  # Output should be (-0.1, -0.1, -0.1, -0.1, -0.1, -0.1)
