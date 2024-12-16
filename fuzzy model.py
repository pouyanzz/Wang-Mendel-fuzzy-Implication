import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class FuzzyModel():
    def __init__(self, training_data, num_fuzzy_sets=7):
        """
        Initialize the fuzzy model with dynamically calculated fuzzy sets.
        
        Parameters:
            training_data: List of tuples (x1, x2, y).
            num_fuzzy_sets: Number of fuzzy sets to create for inputs and outputs.
        """
        # Extract input and output data
        x1_data = [x[0] for x in training_data]
        x2_data = [x[1] for x in training_data]
        y_data = [x[2] for x in training_data]
        
        # Calculate fuzzy sets for inputs and output
        self.input_fuzzy_sets_x1 = self.calculate_fuzzy_sets(x1_data, num_fuzzy_sets)
        self.input_fuzzy_sets_x2 = self.calculate_fuzzy_sets(x2_data, num_fuzzy_sets)
        self.output_fuzzy_sets = self.calculate_fuzzy_sets(y_data, num_fuzzy_sets)
        
        # Calculate centers for output fuzzy sets
        self.output_centers = {label: (a + b + c) / 3 for label, (a, b, c) in self.output_fuzzy_sets.items()}
    def calculate_fuzzy_sets(self, data, num_sets=7):
        """
        Automatically calculate fuzzy set ranges for inputs and outputs.
        
        Parameters:
            data: List or numpy array of data values (input or output).
            num_sets: Number of fuzzy sets to create.
        
        Returns:
            Dictionary of fuzzy sets with triangular membership function parameters.
        """
        # Determine the range of the data
        min_val = np.min(data)
        max_val = np.max(data)
        sets = ["NB","NM","NS","ZR","PS","PM","PB"]
        # Calculate step size for dividing the range
        step = (max_val - min_val) / (num_sets - 1)
        
        # Generate triangular fuzzy sets
        fuzzy_sets = {}
        for i in range(len(sets)):
            label = f"{sets[i]}"  # Label for the fuzzy set
            if i == 0:
                fuzzy_sets[label] = (min_val, min_val, min_val + step)
            elif i == num_sets - 1:
                fuzzy_sets[label] = (max_val - step, max_val, max_val)
            else:
                fuzzy_sets[label] = (min_val + (i - 1) * step, min_val + i * step, min_val + (i + 1) * step)
        
        return fuzzy_sets

    def generate_rules(self, training_data):
        """
        Generate fuzzy rules using the Wang-Mendel method.
        
        Parameters:
            training_data: List of tuples (x1, x2, y), where:
                        x1, x2 are inputs, and y is the output.
        
        Returns:
            A dynamically generated rule base.
        """
        # Initialize a list to store rules with weights
        rules = []

        # Step 1: Generate all possible rules
        for x1, x2, y in training_data:
            # Fuzzify inputs using respective fuzzy sets
            memberships_x1 = self.fuzzify_input(x1, self.input_fuzzy_sets_x1)
            memberships_x2 = self.fuzzify_input(x2, self.input_fuzzy_sets_x2)

            # Fuzzify the output using output fuzzy sets
            memberships_y = self.fuzzify_input(y, self.output_fuzzy_sets)

            # Identify the fuzzy set with the maximum membership for each variable
            x1_label = max(memberships_x1, key=memberships_x1.get)
            x2_label = max(memberships_x2, key=memberships_x2.get)
            y_label = max(memberships_y, key=memberships_y.get)

            # Compute the rule weight based on input memberships only
            weight = min(memberships_x1[x1_label], memberships_x2[x2_label])


            # Append the rule: antecedent, consequent, and weight
            rules.append(((x1_label, x2_label), y_label, weight))

        # Step 2: Resolve conflicts by keeping the highest-weight rule for each antecedent
        rule_dict = {}
        for (antecedent, consequent, weight) in rules:
            if antecedent not in rule_dict or rule_dict[antecedent][1] < weight:
                rule_dict[antecedent] = (consequent, weight)

        # Convert the rule dictionary into a simplified rule base
        self.rule_base = {k: v[0] for k, v in rule_dict.items()}

        # Return the rule base for verification or further processing
        return self.rule_base

    def calculate_mse(self, predictions, targets):
        """
        Calculate Mean Squared Error (MSE) between predicted and actual target values.

        :param predictions: List or numpy array of predicted outputs (y'_i).
        :param targets: List or numpy array of actual outputs (y_i).
        :return: MSE value.
        """
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        squared_differences = (predictions - targets) ** 2
        
        mse = np.sum(squared_differences) / (2 * len(targets))

        
        return mse
    def triangular_mf(self, x, a, b, c):
        """
        Calculate the membership degree for a triangular fuzzy set.
        Handles cases where a == b (half-triangle) or b == c.
        """
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-6
        return np.maximum(0, np.minimum(
            (x - a) / (b - a + epsilon) if b != a else (1 if x >= a else 0),
            (c - x) / (c - b + epsilon) if b != c else (1 if x <= c else 0)
        ))

    def fuzzify_input(self, x, fuzzy_sets):
        membership_degrees = {}
        for label, (a, b, c) in fuzzy_sets.items():
            membership_degrees[label] = self.triangular_mf(x, a, b, c)
        return membership_degrees


    def fuzzify_output(self):
        """
        Fuzzify output based on predefined triangular membership functions.
        
        Returns:
            Dictionary with centers of output fuzzy sets.
        """
        return self.output_centers

    def defuzzify_center_of_average(self, memberships):
        """
        Perform defuzzification using the Center of Average method.
        
        Parameters:
            memberships: A dictionary where keys are labels and values are the membership degrees.
        
        Returns:
            Crisp defuzzified value.
        """
        numerator = sum(membership * self.output_centers[label] for label, membership in memberships.items())
        denominator = sum(memberships.values())
        
        if denominator == 0:
            return 0
        
        return numerator / denominator

    def evaluate_rules(self, input_memberships_x1, input_memberships_x2):
        """
        Evaluate the rule base to compute output fuzzy memberships.
        
        Parameters:
            input_memberships_x1: Fuzzy memberships for input x1.
            input_memberships_x2: Fuzzy memberships for input x2.
        
        Returns:
            Dictionary of output fuzzy memberships (aggregated for defuzzification).
        """
        # Initialize output fuzzy memberships
        output_memberships = {label: 0 for label in self.output_fuzzy_sets.keys()}

        # Iterate over the dynamically generated rule base
        for (input1_label, input2_label), output_label in self.rule_base.items():
            # Check if the rule's antecedent matches the current memberships
            if input1_label in input_memberships_x1 and input2_label in input_memberships_x2:
                # Compute the firing strength (minimum of input memberships)
                firing_strength = min(input_memberships_x1[input1_label], input_memberships_x2[input2_label])
                
                # Aggregate output memberships (take max for overlapping rules)
                output_memberships[output_label] = max(output_memberships[output_label], firing_strength)

        return output_memberships


    def plot_fuzzy_sets(self):
        """
        Plot the fuzzy sets for inputs x1, x2, and output y.
        """
        # Vectorize the triangular_mf function to apply it element-wise on an array
        vectorized_mf = np.vectorize(self.triangular_mf)
        
        # Create subplots
        fig, ax = plt.subplots(3, 1, figsize=(10, 12))

        # Plot fuzzy sets for x1
        ax[0].set_title('Fuzzy Sets for x1')
        for label, (a, b, c) in self.input_fuzzy_sets_x1.items():
            x = np.linspace(a , c , 500)
            y = vectorized_mf(x, a, b, c)
            ax[0].plot(x, y, label=f'{label}')
        ax[0].legend(loc='upper right')
        ax[0].set_xlabel('x1')
        ax[0].set_ylabel('Membership Degree')

        # Plot fuzzy sets for x2
        ax[1].set_title('Fuzzy Sets for x2')
        for label, (a, b, c) in self.input_fuzzy_sets_x2.items():
            x = np.linspace(a , c, 500)
            y = vectorized_mf(x, a, b, c)
            ax[1].plot(x, y, label=f'{label}')
        ax[1].legend(loc='upper right')
        ax[1].set_xlabel('x2')
        ax[1].set_ylabel('Membership Degree')

        # Plot fuzzy sets for output
        ax[2].set_title('Fuzzy Sets for Output y')
        for label, (a, b, c) in self.output_fuzzy_sets.items():
            x = np.linspace(a , c , 500)
            y = vectorized_mf(x, a, b, c)
            ax[2].plot(x, y, label=f'{label}')
        ax[2].legend(loc='upper right')
        ax[2].set_xlabel('y')
        ax[2].set_ylabel('Membership Degree')

        # Show the plots
        plt.tight_layout()
        plt.show()

    def train_plot_3d_output(self, test_cases):
        """Plot the crisp outputs from test cases in a 3D plot."""
        
        # Extracting x1 and x2 values and corresponding outputs
        x1_values = [case[0] for case in test_cases]
        x2_values = [case[1] for case in test_cases]
        actual_output = [case[2] for case in test_cases]
        
        # Assuming predictions is already calculated in your main code
        predictions = []
        targets = []
        for input_value_x1, input_value_x2, actual_output in zip(x1_values, x2_values, actual_output):
            memberships_x1 = self.fuzzify_input(input_value_x1,self.input_fuzzy_sets_x1)
            memberships_x2 = self.fuzzify_input(input_value_x2,self.input_fuzzy_sets_x2)
            output_memberships = self.evaluate_rules(memberships_x1,memberships_x2)
            crisp_output = self.defuzzify_center_of_average(output_memberships)
            predictions.append(crisp_output)
            targets.append(actual_output)
        # Create a meshgrid for plotting
        mse = model.calculate_mse(predictions, targets)
        print(f"Mean Squared Error (MSE): {mse}")
        X1_grid, X2_grid = np.meshgrid(np.unique(x1_values), np.unique(x2_values))
        
        # Reshape predictions to match the meshgrid shape
        Z_grid = np.array(predictions).reshape(X1_grid.shape)

        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(X1_grid, X2_grid, Z_grid, cmap='viridis', edgecolor='none')
        
        ax.set_title(f'Crisp Output Surface Plot, MSE: {mse}')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Crisp Output')
        
        plt.show()
        
if __name__ == "__main__":
    data = pd.read_csv("training_data.csv")
    data= data.to_records(index=False)
    model = FuzzyModel(data)

    x1_fuzzy_set = model.input_fuzzy_sets_x1
    x2_fuzzy_set = model.input_fuzzy_sets_x2

    # Generate rules using Wang-Mendel method
    model.generate_rules(data)
    print("\nGenerated Rule Base:")
    for rule, output in model.rule_base.items():
        print(f"If x1 is {rule[0]} and x2 is {rule[1]}, then output is {output}")

    # Define test cases (with corresponding actual target outputs)
    test_cases = pd.read_csv("test_data.csv")
    test_cases = test_cases.to_records(index=False)


    model.plot_fuzzy_sets()
    model.train_plot_3d_output(test_cases)

