import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_data_and_plot(num_points, filename, title):
    # Generate evenly spaced points between -5 and 5
    x_values = np.linspace(-5, 5, num_points)
    x1, x2 = np.meshgrid(x_values, x_values)

    # Calculate the output F(x1, x2) = x1^2 + x2^2
    output = x1**2 + x2**2

    # Create a pandas DataFrame
    data = pd.DataFrame({'x1': x1.flatten(), 'x2': x2.flatten(), 'F(x1, x2)': output.flatten()})

    # Save to a CSV file
    data.to_csv(filename, index=False)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1, x2, output, cmap='viridis')  # Use original meshgrid arrays
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.title(f'{title} 3D Surface Plot')
    plt.savefig(f'{title} 3D Surface Plot.jpg')

    plt.show()


# Generate and plot training data
generate_data_and_plot(41, "training_data.csv", "Training")

# Generate and plot test data
generate_data_and_plot(168, "test_data.csv", "Test")
