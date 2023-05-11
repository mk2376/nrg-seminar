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
    imageio.v3.imsave(filename, image)

# Function to perform relaxation in the multigrid method
def relax_kernel(queue, u, f, unew, program):
    program.relax(queue, u.shape, None, u.data, f.data, unew.data, np.int32(u.shape[0]))

# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Apply Poisson image editing using multigrid solver.')
    parser.add_argument('input_file', help='Input image file path')
    args = parser.parse_args()

    # Read image and initialize parameters
    input_image = load_image(args.input_file)
    N = input_image.shape[0]
    num_cycles = 5
    num_relaxations = 5

    # OpenCL context
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    # Allocate memory
    mf = cl.mem_flags
    u = cl_array.to_device(queue, np.zeros((N, N), dtype=np.float32))
    f = cl_array.to_device(queue, np.zeros((N, N), dtype=np.float32))
    unew = cl_array.to_device(queue, np.zeros((N, N), dtype=np.float32))

    # Load kernel
    with open('src/multigrid.cl', 'r') as f:
        fstr = "".join(f.readlines())
    program = cl.Program(context, fstr).build()

    # Run multigrid
    for cycle in range(num_cycles):
        for relax in range(num_relaxations):
            relax_kernel(queue, u, f, unew, program)
            u, unew = unew, u

    # Save output image
    output_image = u.get()
    output_file = args.input_file.rsplit('.', 1)[0] + '_output.png'
    save_image(output_file, output_image)

# If this script is the main module, run the main function
if __name__ == "__main__":
    main()
