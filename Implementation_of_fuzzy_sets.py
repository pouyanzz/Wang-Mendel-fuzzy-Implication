import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Define the triangular membership function
def triangular_mf(x, a, b, c):
    """
    Calculate the membership degree for a triangular fuzzy set.
    
    Parameters:
        x: Input value or array.
        a: Start of the triangle.
        b: Peak of the triangle.
        c: End of the triangle.
    
    Returns:
        Membership degree for each x.
    """
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))
# Function to fuzzify the input using triangular membership functions
def fuzzify_input(x, fuzzy_sets):
    membership_degrees = {}
    for label, mf in fuzzy_sets.items():
        membership_degrees[label] = triangular_mf(x, mf[0], mf[1], mf[2])
    return membership_degrees
# Defuzzification using Center of Average method
def defuzzify_center_of_average(fuzzy_output_sets, memberships):
    """
    Perform defuzzification using the Center of Average method.
    
    Parameters:
        fuzzy_output_sets: A dictionary where keys are labels and values are the centers of fuzzy sets.
        memberships: A dictionary where keys are labels and values are the membership degrees.
    
    Returns:
        Crisp defuzzified value.
    """
    numerator = 0
    denominator = 0
    
    for label, center in fuzzy_output_sets.items():
        membership = memberships.get(label, 0)  # Get membership degree
        numerator += membership * center
        denominator += membership
    
    if denominator == 0:
        return 0  # Avoid division by zero, return default crisp output
    
    return numerator / denominator


x = np.linspace(-5, 5, 1000)  # For inputs x1 and x2
f = np.linspace(0, 50, 1000)  # For output F(x1, x2)

# Define triangular fuzzy sets for inputs (x1 and x2)
fuzzy_sets = {
    "NB": triangular_mf(x, -5, -5, -3.33),
    "NM": triangular_mf(x, -5, -3.33, -1.67),
    "NS": triangular_mf(x, -3.33, -1.67, 0),
    "ZR": triangular_mf(x, -1.67, 0, 1.67),
    "PS": triangular_mf(x, 0, 1.67, 3.33),
    "PM": triangular_mf(x, 1.67, 3.33, 5),
    "PB": triangular_mf(x, 3.33, 5, 5),
}
# Define the centers of the fuzzy output sets
output_centers = {
    "NB": 4.16,  # Approximate center of the "NB" triangular set (0 to 8.33)
    "NM": 12.5,  # Approximate center of the "NM" triangular set (8.33 to 16.67)
    "NS": 20.83, # Approximate center of the "NS" triangular set (16.67 to 25)
    "ZR": 29.16, # Approximate center of the "ZR" triangular set (25 to 33.33)
    "PS": 37.5,  # Approximate center of the "PS" triangular set (33.33 to 41.67)
    "PM": 45.83, # Approximate center of the "PM" triangular set (41.67 to 50)
    "PB": 50     # Approximate center of the "PB" triangular set (50)
}
# Plot the fuzzy sets for inputs
plt.figure(figsize=(10, 6))
for label, mf in fuzzy_sets.items():
    plt.plot(x, mf, label=label)
plt.title("Fuzzy Sets for x1 and x2")
plt.xlabel("x")
plt.ylabel("Membership Degree")
plt.legend()
plt.grid()
plt.show()

# Define triangular fuzzy sets for the output (F(x1, x2))
fuzzy_sets_output = {
    "NB": triangular_mf(f, 0, 0, 8.33),
    "NM": triangular_mf(f, 0, 8.33, 16.67),
    "NS": triangular_mf(f, 8.33, 16.67, 25),
    "ZR": triangular_mf(f, 16.67, 25, 33.33),
    "PS": triangular_mf(f, 25, 33.33, 41.67),
    "PM": triangular_mf(f, 33.33, 41.67, 50),
    "PB": triangular_mf(f, 41.67, 50, 50),
}

# Plot the fuzzy sets for the output
plt.figure(figsize=(10, 6))
for label, mf in fuzzy_sets_output.items():
    plt.plot(f, mf, label=label)
plt.title("Fuzzy Sets for F(x1, x2)")
plt.xlabel("F(x1, x2)")
plt.ylabel("Membership Degree")
plt.legend()
plt.grid()
plt.show()


rule_base = {
    ("NB", "NB"): "PB",
    ("NB", "NM"): "PM",
    ("NB", "NS"): "PS",
    ("NB", "ZR"): "ZR",
    ("NB", "PS"): "NS",
    ("NB", "PM"): "NM",
    ("NB", "PB"): "NB",

    ("NM", "NB"): "PB",
    ("NM", "NM"): "PM",
    ("NM", "NS"): "PS",
    ("NM", "ZR"): "ZR",
    ("NM", "PS"): "NS",
    ("NM", "PM"): "NM",
    ("NM", "PB"): "NB",

    # Adjusting NS and ZR interactions
    ("NS", "NB"): "PS",
    ("NS", "NM"): "PS",
    ("NS", "NS"): "ZR",
    ("NS", "ZR"): "ZR",
    ("NS", "PS"): "NS",
    ("NS", "PM"): "NM",
    ("NS", "PB"): "ZR",

    # ZR interactions
    ("ZR", "NB"): "ZR",
    ("ZR", "NM"): "ZR",
    ("ZR", "NS"): "ZR",
    ("ZR", "ZR"): "ZR",
    ("ZR", "PS"): "ZR",
    ("ZR", "PM"): "ZR",
    ("ZR", "PB"): "ZR",

    # PS interactions
    ("PS", "NB"): "ZR",
    ("PS", "NM"): "ZR",
    ("PS", "NS"): "ZR",
    ("PS", "ZR"): "ZR",
    ("PS", "PS"): "ZR",
    ("PS", "PM"): "ZR",
    ("PS", "PB"): "ZR",

    # PB interactions
    ("PM","NB"):"NB",
    ("PM","NM"):"NM",
    ("PM","NS"):"NM",
    ("PM","ZR"):"PM",
    ("PM","PS"):"PM",
    ("PM","PM"):"PB",
    ("PM","PB"):"PB",
    ("PB","NB"):"NB",
    ("PB","NM"):"NM",
    ("PB","NS"):"NS",
    ("PB","ZR"):"PM",
    ("PB","PS"):"PM",
    ("PB","PM"):"PB",
    ("PB","PB"):"PB"
}


# Generate the fuzzy rules using the combinations of fuzzy sets
rules = []
for input1_label in fuzzy_sets.keys():
    for input2_label in fuzzy_sets.keys():
        output_label = rule_base.get((input1_label, input2_label), "ZR")  # Default to "ZR" if no rule is defined
        rules.append((input1_label, input2_label, output_label))

# Print the generated rules
for rule in rules:
    print(f"If x1 is {rule[0]} and x2 is {rule[1]}, then the output is {rule[2]}")
    
    
data = pd.read_csv("training_data.csv")

# Apply fuzzification and defuzzification to the entire dataset
crisp_outputs = []
for index, row in data.iterrows():
    x1_input = row['x1']
    x2_input = row['x2']
    
    # Fuzzify inputs
    x1_memberships = fuzzify_input(x1_input, fuzzy_sets)
    x2_memberships = fuzzify_input(x2_input, fuzzy_sets)
    
    # Here we use some simple rule for membership intersection
    output_memberships = {}
# Apply rules and aggregate outputs
output_memberships = {output: 0 for output in fuzzy_sets_output}
for rule in rules:
    input1_label, input2_label, output_label = rule
    membership_x1 = x1_memberships[input1_label]
    membership_x2 = x2_memberships[input2_label]
    output_memberships[output_label] = max(output_memberships[output_label], min(membership_x1, membership_x2))

# Defuzzify to get crisp output
crisp_output = defuzzify_center_of_average(output_centers, output_memberships)

print(crisp_output)