import numpy as np
from scipy.ndimage import gaussian_filter
import imageio

# Function to apply Gaussian blur and normalize the image
def load(filename):
    image_input = imageio.v3.imread(filename, mode='L')
    image_input = gaussian_filter(image_input, 1)
    image_input = image_input.astype(np.float32)  # Convert image to float32 before division
    
    # Normalize to [0, 1]
    image_input /= np.max(image_input)
    
    print("image_input", np.min(image_input), np.max(image_input), np.mean(image_input))  # Print min, max and mean values

    return image_input

# Function to save the image
def save(filename, image_output):
    print("image_output_raw", np.min(image_output), np.max(image_output), np.mean(image_output))  # Print min, max and mean values
    
    # Normalize to [0, 1]
    image_output = (image_output - np.min(image_output)) / (np.max(image_output) - np.min(image_output))
    
    print("image_output", np.min(image_output), np.max(image_output), np.mean(image_output))  # Print min, max and mean values
    
    # Convert image to 8-bit integer format
    image_output = (image_output * 255).astype(np.uint8)
    imageio.v3.imwrite(filename, image_output)