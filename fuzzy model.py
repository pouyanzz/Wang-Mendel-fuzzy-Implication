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
        x1_data = [x[0] for x in training_data]
        x2_data = [x[1] for x in training_data]
        y_data = [x[2] for x in training_data]
        
        self.input_fuzzy_sets_x1 = self.calculate_fuzzy_sets(x1_data, num_fuzzy_sets)
        self.input_fuzzy_sets_x2 = self.calculate_fuzzy_sets(x2_data, num_fuzzy_sets)
        self.output_fuzzy_sets = self.calculate_fuzzy_sets(y_data, num_fuzzy_sets)
        
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
        min_val = np.min(data)
        max_val = np.max(data)
        sets = ["NB","NM","NS","ZR","PS","PM","PB"]
        step = (max_val - min_val) / (num_sets - 1)
        
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
        rules = []

        for x1, x2, y in training_data:
            memberships_x1 = self.fuzzify_input(x1, self.input_fuzzy_sets_x1)
            memberships_x2 = self.fuzzify_input(x2, self.input_fuzzy_sets_x2)

            memberships_y = self.fuzzify_input(y, self.output_fuzzy_sets)

            x1_label = max(memberships_x1, key=memberships_x1.get)
            x2_label = max(memberships_x2, key=memberships_x2.get)
            y_label = max(memberships_y, key=memberships_y.get)

            weight = min(memberships_x1[x1_label], memberships_x2[x2_label])


            rules.append(((x1_label, x2_label), y_label, weight))

        rule_dict = {}
        for (antecedent, consequent, weight) in rules:
            if antecedent not in rule_dict or rule_dict[antecedent][1] < weight:
                rule_dict[antecedent] = (consequent, weight)

        self.rule_base = {k: v[0] for k, v in rule_dict.items()}

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
        return np.maximum(0, np.minimum(
            (x - a) / (b - a ) if b != a else (1 if x >= a else 0),
            (c - x) / (c - b) if b != c else (1 if x <= c else 0)
        ))

    def fuzzify_input(self, x, fuzzy_sets):
        membership_degrees = {}
        for label, (a, b, c) in fuzzy_sets.items():
            membership_degrees[label] = self.triangular_mf(x, a, b, c)
        return membership_degrees


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
        output_memberships = {label: 0 for label in self.output_fuzzy_sets.keys()}

        for (input1_label, input2_label), output_label in self.rule_base.items():
            if input1_label in input_memberships_x1 and input2_label in input_memberships_x2:
                firing_strength = min(input_memberships_x1[input1_label], input_memberships_x2[input2_label])
                
                output_memberships[output_label] = max(output_memberships[output_label], firing_strength)

        return output_memberships


    def plot_fuzzy_sets(self):

        vectorized_mf = np.vectorize(self.triangular_mf)
        
        fig, ax = plt.subplots(3, 1, figsize=(10, 12))

        ax[0].set_title('Fuzzy Sets for x1')
        for label, (a, b, c) in self.input_fuzzy_sets_x1.items():
            x = np.linspace(a , c , 500)
            y = vectorized_mf(x, a, b, c)
            ax[0].plot(x, y, label=f'{label}')
        ax[0].legend(loc='upper right')
        ax[0].set_xlabel('x1')
        ax[0].set_ylabel('Membership Degree')

        ax[1].set_title('Fuzzy Sets for x2')
        for label, (a, b, c) in self.input_fuzzy_sets_x2.items():
            x = np.linspace(a , c, 500)
            y = vectorized_mf(x, a, b, c)
            ax[1].plot(x, y, label=f'{label}')
        ax[1].legend(loc='upper right')
        ax[1].set_xlabel('x2')
        ax[1].set_ylabel('Membership Degree')

        ax[2].set_title('Fuzzy Sets for Output y')
        for label, (a, b, c) in self.output_fuzzy_sets.items():
            x = np.linspace(a , c , 500)
            y = vectorized_mf(x, a, b, c)
            ax[2].plot(x, y, label=f'{label}')
        ax[2].legend(loc='upper right')
        ax[2].set_xlabel('y')
        ax[2].set_ylabel('Membership Degree')

        plt.savefig("Fuzzy sets Plots")
        plt.tight_layout()
        plt.show()

    def predict_plot_3d_output(self, test_cases):
        x1_values = [case[0] for case in test_cases]
        x2_values = [case[1] for case in test_cases]
        actual_output = [case[2] for case in test_cases]
        
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
        
        fig1 = plt.figure(figsize=(12, 6))

        ax1 = fig1.add_subplot(121, projection='3d')
        ax1.scatter(x1_values, x2_values, predictions, c='blue', marker='o', label='Predicted')
        ax1.set_title('3D Scatter Plot of Predicted Values')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_zlabel('Predicted Output')
        ax1.legend()

        ax2 = fig1.add_subplot(122, projection='3d')
        ax2.scatter(x1_values, x2_values, targets, c='red', marker='^', label='Target')
        ax2.set_title('3D Scatter Plot of Target Values')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_zlabel('Target Output')
        ax2.legend()

        plt.tight_layout()
        plt.savefig("3D_Scatter_Target_vs_Predicted.jpg")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(targets)), targets, label='Target Values', color='red', linestyle='-', marker='o')
        plt.plot(range(len(predictions)), predictions, label='Predicted Values', color='blue', linestyle='--', marker='x')
        
        plt.title(f'Target vs Predicted Values (MSE: {mse:.4f})')
        plt.xlabel('Test Case Index')
        plt.ylabel('Output Values')
        plt.legend()
        plt.grid(True)

        plt.savefig("Line_Plot_Target_vs_Predicted.jpg")
        plt.show()


        
if __name__ == "__main__":
    data = pd.read_csv("training_data.csv")
    data= data.to_records(index=False)
    model = FuzzyModel(data)



    # Generate rules using Wang-Mendel method
    model.generate_rules(data)
    print("\nGenerated Rule Base:")
    for rule, output in model.rule_base.items():
        print(f"If x1 is {rule[0]} and x2 is {rule[1]}, then output is {output}")

    test_cases = pd.read_csv("test_data.csv")
    test_cases = test_cases.to_records(index=False)


    model.plot_fuzzy_sets()
    model.predict_plot_3d_output(test_cases)

