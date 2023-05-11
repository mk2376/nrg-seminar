import os
import argparse
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from scipy.ndimage import gaussian_filter
import imageio

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
def relax_kernel(program, queue, u, f, unew, N, M):
    program.relax(queue, (N, M), None, u.data, f.data, unew.data, N, M)

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
    print("input_image", np.min(input_image), np.max(input_image), np.mean(input_image))  # Print min, max and mean values
    N = np.int32(input_image.shape[0])
    M = np.int32(input_image.shape[1])
    print("NxM", N, M)
    num_cycles = 1
    num_relaxations = 1

    # OpenCL context
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    # Allocate memory
    u = cl_array.to_device(queue, np.zeros((N, M), dtype=np.float32))
    f = cl_array.to_device(queue, input_image)
    unew = cl_array.to_device(queue, np.zeros((N, M), dtype=np.float32))

    # Load kernel
    with open('src/multigrid.cl', 'r') as file_obj:
        fstr = "".join(file_obj.readlines())
    program = cl.Program(context, fstr).build()

    # Run multigrid
    for cycle in range(num_cycles):
        for relax in range(num_relaxations):
            relax_kernel(program, queue, u, f, unew, N, M)
            u, unew = unew, u

    # Save output image
    output_image = u.get()
    output_file = filename + '_output' + file_extension
    save_image(output_file, output_image)

# If this script is the main module, run the main function
if __name__ == "__main__":
    main()
