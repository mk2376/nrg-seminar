import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

import image

# Function to compute gradient of image
def gradient(image_input):
    # Gaussian smoothing
    smoothed = gaussian_filter(image_input, sigma=1.0)
    
    # Sobel filters for computing image gradient
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8

    # Compute image gradient
    grad_x = convolve2d(smoothed, sobel_x, mode='same', boundary='fill', fillvalue=0)
    grad_y = convolve2d(smoothed, sobel_y, mode='same', boundary='fill', fillvalue=0)

    return grad_x, grad_y

# Function to compute divergence of vector field
def divergence(field_x, field_y):
    # Use central difference for divergence calculation
    divergence_x = convolve2d(field_x, np.array([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]), mode='same', boundary='fill', fillvalue=0)
    divergence_y = convolve2d(field_y, np.array([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]), mode='same', boundary='fill', fillvalue=0)

    return divergence_x + divergence_y

# Function to generate vector field from binary image
def generate_vector_field(image_input):
    # Compute image gradient
    grad_x, grad_y = gradient(image_input)

    # Create vector field
    field_x = -grad_y  # y-gradient points in x-direction
    field_y = grad_x   # x-gradient points in y-direction

    return field_x, field_y

# Function to compute divergence field from binary image
def generate(image_input):
    # Generate vector field
    field_x, field_y = generate_vector_field(image_input)

    # Compute divergence
    divergence_field = divergence(field_x, field_y)
    
    # Invert white spots into black (so the divergence is a single color) (currently [-0.5, 0.5])
    divergence_field = -np.abs(divergence_field)
    
    # Convert it into [0, 1] range (currently [-0.5, 0])
    divergence_field = -divergence_field*2
    
    # divergence_field has some spots with less intensity which we have to correct
    divergence_field = np.where(divergence_field > 0.05, 1, divergence_field)
    
    # Convert it back into [-0.5, 0] range
    divergence_field = -divergence_field/2
    
    print("divergence_field")
    image.save('outputs/divergence_field.jpg', divergence_field)

    return divergence_field
