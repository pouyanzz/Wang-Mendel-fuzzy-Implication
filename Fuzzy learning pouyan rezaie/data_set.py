import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_data_and_plot(num_points, filename, title, is_training=True):
    if is_training:
        x_values = np.linspace(-5, 5, num_points)
        x1, x2 = np.meshgrid(x_values, x_values)

        output = x1**2 + x2**2

        data = pd.DataFrame({'x1': x1.flatten(), 'x2': x2.flatten(), 'F(x1, x2)': output.flatten()})

    else:
        x1_values = np.random.uniform(-5, 5, num_points)
        x2_values = np.random.uniform(-5, 5, num_points)
        output = x1_values**2 + x2_values**2

        data = pd.DataFrame({'x1': x1_values, 'x2': x2_values, 'F(x1, x2)': output})

    data.to_csv(filename, index=False)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if is_training:
        ax.plot_surface(x1, x2, output, cmap='viridis') 
    else:
        ax.scatter(data['x1'], data['x2'], data['F(x1, x2)'], color='red', marker='o')  

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.title(f'{title} 3D Surface Plot' if is_training else f'{title} Test Data Scatter Plot')
    plt.savefig(f'{title} {"3D Surface Plot" if is_training else "Test Data Scatter Plot"}.jpg')

    plt.show()

generate_data_and_plot(41, "training_data.csv", "Training", is_training=True)

generate_data_and_plot(168, "test_data.csv", "Test", is_training=False)
