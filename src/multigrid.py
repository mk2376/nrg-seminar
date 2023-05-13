import os
import argparse
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import imageio

# Function to compute gradient of image
def gradient(image):
    # Sobel filters for computing image gradient
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8

    # Compute image gradient
    grad_x = convolve2d(image, sobel_x, mode='same', boundary='fill', fillvalue=0)
    grad_y = convolve2d(image, sobel_y, mode='same', boundary='fill', fillvalue=0)

    return grad_x, grad_y

# Function to compute divergence of vector field
def divergence(field_x, field_y):
    # Use central difference for divergence calculation
    divergence_x = convolve2d(field_x, np.array([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]), mode='same', boundary='fill', fillvalue=0)
    divergence_y = convolve2d(field_y, np.array([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]), mode='same', boundary='fill', fillvalue=0)

    return divergence_x + divergence_y

# Function to generate vector field from binary image
def generate_vector_field(image):
    # Compute image gradient
    grad_x, grad_y = gradient(image)

    # Create vector field
    field_x = -grad_y  # y-gradient points in x-direction
    field_y = grad_x   # x-gradient points in y-direction

    return field_x, field_y

# Function to compute divergence field from binary image
def generate_divergence_field(image):
    # Generate vector field
    field_x, field_y = generate_vector_field(image)

    # Compute divergence
    divergence_field = divergence(field_x, field_y)
    
    print("divergence_field", np.min(divergence_field), np.max(divergence_field), np.mean(divergence_field))  # Print min, max and mean values
    save_image('assets/divergence_field.png', divergence_field)

    return divergence_field

# Function to apply Gaussian blur and normalize the image
def load_image(filename):
    image = imageio.v3.imread(filename, mode='L')
    image = gaussian_filter(image, 1)
    image = image.astype(np.float32)  # Convert image to float32 before division
    image /= np.max(image)
    return image

# Function to save the image
def save_image(filename, image):
    print("save_image", np.min(image), np.max(image), np.mean(image))  # Print min, max and mean values
    
    # Convert image to 8-bit integer format
    image = (image * 255).astype(np.uint8)
    imageio.v3.imwrite(filename, image)

# Function to perform relaxation in the multigrid method
def relax_kernel(program, queue, u, u_new, N, M):
    global_work_size = (N, M)
    local_work_size = None # (2, 2)  # or whatever value is appropriate for your hardware
    program.relax_kernel(queue, global_work_size, local_work_size, u.data, u_new.data, N, M)

# Function to calculate residual
def residual_kernel(program, queue, u, f, res, N, M):
    global_work_size = (N, M)
    local_work_size = None # (2, 2)  # or whatever value is appropriate for your hardware
    program.residual_kernel(queue, global_work_size, local_work_size, u.data, f.data, res.data, N, M)

# Function to perform restriction in the multigrid method
def restrict_kernel(program, queue, res, r2, N, M):
    halfN = np.int32(N//2);
    halfM = np.int32(M//2);
    global_work_size = (halfN, halfM)
    local_work_size = None # (2, 2) # or whatever value is appropriate for your hardware
    program.restrict_kernel(queue, global_work_size, local_work_size, res.data, r2.data, N, M)

# Function to perform prolongation in the multigrid method
def prolong_kernel(program, queue, r2, u, N, M):
    halfN = np.int32(N//2);
    halfM = np.int32(M//2);
    global_work_size = (halfN, halfM)
    local_work_size = None # (2, 2)  # or whatever value is appropriate for your hardware
    program.prolong_kernel(queue, global_work_size, local_work_size, u.data, r2.data, N, M)

# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Apply Poisson image editing using multigrid solver.')
    parser.add_argument('input_file', help='Input image file path')
    args = parser.parse_args()

    # Get input file name and extension
    filename, file_extension = os.path.splitext(args.input_file)

    # Read image and initialize parameters
    input_image = load_image(args.input_file)
    N = np.int32(input_image.shape[0])
    M = np.int32(input_image.shape[1])
    halfN = np.int32(N//2);
    halfM = np.int32(M//2);
    
    num_cycles = 100
    num_relaxations = 1

    # OpenCL context
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    # Allocate memory
    u = cl_array.to_device(queue, np.zeros((N, M), dtype=np.float32))
    f = cl_array.to_device(queue, generate_divergence_field(input_image).astype(np.float32))  # Convert to float32
    u_new = cl_array.to_device(queue, np.zeros((N, M), dtype=np.float32))
    r2 = cl_array.to_device(queue, np.zeros((halfN, halfM), dtype=np.float32))  # For restriction
    r2_new = cl_array.to_device(queue, np.zeros((halfN, halfM), dtype=np.float32))  # For relaxation

    # Load kernel
    with open('src/multigrid.cl', 'r') as file_obj:
        fstr = "".join(file_obj.readlines())
    program = cl.Program(context, fstr).build()
    
    # Run multigrid
    for cycle in range(num_cycles):
        for relax in range(num_relaxations):
            relax_kernel(program, queue, u, u_new, N, M)
            u, u_new = u_new, u
            
        # Calculate residual
        res = cl_array.to_device(queue, np.zeros((N, M), dtype=np.float32))  # Buffer for the residual
        residual_kernel(program, queue, u, f, res, N, M)

        # Apply restriction
        restrict_kernel(program, queue, res, r2, N, M) # Downsample to the next grid level
        # print("r2", np.min(r2.get()), np.max(r2.get()), np.mean(r2.get()))  # Print min, max and mean values
        
        # Relaxation on the coarser grid
        for relax in range(num_relaxations):
            relax_kernel(program, queue, r2, r2_new, halfN, halfM)
            r2, r2_new = r2_new, r2

        # Apply prolongation
        prolong_kernel(program, queue, r2, u, N, M)
        # print("u", np.min(u.get()), np.max(u.get()), np.mean(u.get()))  # Print min, max and mean values
        
        # Perform relaxation on the original grid
        for relax in range(num_relaxations):
            relax_kernel(program, queue, u, u_new, N, M)
            u, u_new = u_new, u

    # Save output image
    output_image = u.get()
    output_file = filename + '_output' + file_extension
    save_image(output_file, output_image)

# If this script is the main module, run the main function
if __name__ == "__main__":
    main()
