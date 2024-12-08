import sys
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import argparse


# Function to draw a cuboid-like structure with faces and edges
def hollow(ax, x, y, z, x_size, y_size, z_size, color='skyblue', linewidth=1, alpha=1, label=None, padding_top_hover=10):
    x_pos, y_pos, z_pos = x, y, z
    vertices = np.array([
        [x_pos, y_pos, z_pos], 
        [x_pos + x_size, y_pos, z_pos], 
        [x_pos + x_size, y_pos + y_size, z_pos],
        [x_pos, y_pos + y_size, z_pos],
        [x_pos, y_pos, z_pos + z_size],
        [x_pos + x_size, y_pos, z_pos + z_size],
        [x_pos + x_size, y_pos + y_size, z_pos + z_size],
        [x_pos, y_pos + y_size, z_pos + z_size]
    ])
    
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    for edge in edges:
        ax.plot3D(*zip(*vertices[edge]), color=color, linewidth=linewidth, alpha=alpha)

    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[3], vertices[7], vertices[4]]
    ]
    
    poly3d = Poly3DCollection(faces, color=color, alpha=alpha, linewidth=linewidth)
    ax.add_collection3d(poly3d)
    
    if label:
        label_position = [x_pos + x_size / 2, y_pos + y_size / 2, z_pos + z_size + padding_top_hover]
        offset_x = 0.1
        offset_y = 0.1
        label_position[0] *= 1 + offset_x
        label_position[1] *= 1 + offset_y
        ax.text(label_position[0], label_position[1], label_position[2], label, color='black', fontsize=12, ha='center', va='center')

    return ax

# Function to visualize the model layers as cubes in 3D
def visualize_model_layers_as_cubes(model, title, scale=0.1, gap=5, layergap=100, length_cube=10):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    layer_names = []
    layer_shapes = []

    _maxx = 0
    for i, layer in enumerate(model.layers):
        layer_names.append(layer.name)
        if hasattr(layer, 'output'):
            layer_shape = layer.output.shape
            _maxx = max(_maxx, max([i for i in layer_shape if i is not None]))
            layer_shapes.append(layer_shape)
        else:
            layer_shapes.append("N/A")

    nextX = 0
    _maxx *= scale * 1.1
    for i, shape in enumerate(layer_shapes):
        if shape != "N/A":
            startCuboidX = nextX
            nextX += scale * gap * 2
            if isinstance(shape, tuple):
                x_size, y_size, z_size = scale, scale, scale
                for dim in shape:
                    if dim is None: continue
                    nextX += scale * (gap + length_cube)
                    x_size = scale * length_cube
                    y_size = scale * dim
                    z_size = scale * dim
                    ax.bar3d(nextX, -0.5 * y_size, -0.5 * z_size, x_size, y_size, z_size, color=np.random.rand(3), linewidth=100 * scale, alpha=0.5)
                    ax.bar3d(nextX + scale, -0.5 * scale, -0.5 * scale, scale * (gap - length_cube), scale, scale, color='red', linewidth=1 * scale)
            endCuboidX = nextX + scale * gap * 2
            nextX += scale * (gap + layergap)
            hollow(ax, startCuboidX, -0.5 * _maxx, -0.5 * _maxx, endCuboidX - startCuboidX, _maxx, _maxx, color=np.random.rand(3), linewidth=scale, alpha=0.1, label=layer_names[i])
            ax.bar3d(endCuboidX, -0.5 * scale, -0.5 * scale, scale * (layergap), scale, scale, color='red', linewidth=1 * scale)
            
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Y (constant)")
    ax.set_zlabel("Z (depth)")
    
    plt.show()

# Function to load and visualize models
def visualize_model_from_filepath(filepath,scale=None):
    try:
        model = tf.keras.models.load_model(filepath)
        if scale is not None:
            visualize_model_layers_as_cubes(model, f"Model: {os.path.basename(filepath)}", scale=scale)
        else:
            visualize_model_layers_as_cubes(model, f"Model: {os.path.basename(filepath)}")
    except Exception as e:
        print(f"Error loading model from {filepath}: {e}")

# Main function to handle arguments or prompt for file path
def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Visualize Keras model layers as cubes in 3D.")
    
    # Add file path argument
    parser.add_argument("--filepath", type=str, help="File path to the h5 model file.")
    # Add scale argument with a default value
    parser.add_argument("--scale", type=float, default=0.1, help="Scale factor for the model layers (default: 0.1)")
    
    # Parse arguments
    args = parser.parse_args()
    scale = args.scale
    filepath = args.filepath
    
    # Check if file paths are provided via command-line arguments
    if len(sys.argv) > 1:
        # Assume file paths are passed as the remaining arguments
        if os.path.exists(filepath):
            print(f"Visualizing model from {filepath} with scale {scale}")
            visualize_model_from_filepath(filepath, scale=scale)
        else:
                print(f"Error: The file {filepath} does not exist.")
    else:
        # Prompt for file path if none is provided
        filepath = input("Enter the h5 file path: ").strip()
        if os.path.exists(filepath):
            visualize_model_from_filepath(filepath, scale=scale)
        else:
            print("Error: The file does not exist.")

if __name__ == "__main__":
    main()
