import numpy as np
from scipy.ndimage import gaussian_filter
import imageio

# Function to apply Gaussian blur and normalize the image
def load(filename):
    image = imageio.v3.imread(filename, mode='L')
    image = gaussian_filter(image, 1)
    image = image.astype(np.float32)  # Convert image to float32 before division
    image /= np.max(image)
    return image

# Function to save the image
def save(filename, image_input):
    print("save_image", np.min(image_input), np.max(image_input), np.mean(image_input))  # Print min, max and mean values
    
    # Convert image to 8-bit integer format
    image_input = (image_input * 255).astype(np.uint8)
    imageio.v3.imwrite(filename, image_input)