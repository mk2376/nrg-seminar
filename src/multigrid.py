import os
import argparse
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

import image
import divergence_field
import kernels

tolerance = 1e-5  # Set your desired tolerance
max_level = 1  # Set your desired maximum level

num_cycles = 1
num_relaxations = 1

# Function to apply multigrid V-cycle
def multigrid_vcycle(program, queue, u, f, N, M, level, prev_res_norm):
    u_new = cl_array.to_device(queue, np.zeros((N, M), dtype=np.float32))
    
    # Pre-relaxation
    for i in range(num_relaxations):
        kernels.relax(program, queue, u, u_new, f, N, M)
        u, u_new = u_new, u
        
        print("u", np.min(u.get()), np.max(u.get()), np.mean(u.get()))  # Print min, max and mean values
        output_image = u.get()
        output_file = "output_" + str(i) + ".jpg"
        image.save(output_file, output_image)

    # Initialize residual norm
    res_norm = np.inf

    if level < max_level:
        # Calculate residual
        res = cl_array.to_device(queue, np.zeros((N, M), dtype=np.float32))  # Buffer for the residual
        kernels.residual(program, queue, u, f, res, N, M)

        res_norm = np.linalg.norm(res.get())

        # Apply restriction
        halfN = np.int32(N//2)
        halfM = np.int32(M//2)
        f_coarse = cl_array.to_device(queue, np.zeros((halfN, halfM), dtype=np.float32))  # RHS at the coarser level
        res_coarse = cl_array.to_device(queue, np.zeros((halfN, halfM), dtype=np.float32))  # Residual at the coarser level
        kernels.restrict(program, queue, f, f_coarse, N, M)
        kernels.restrict(program, queue, res, res_coarse, N, M)

        # Recursive call
        correction_coarse, res_norm_level = multigrid_vcycle(program, queue, res_coarse, f_coarse, halfN, halfM, level+1, prev_res_norm)
        if res_norm_level < res_norm:
            res_norm = res_norm_level

        # Apply prolongation
        kernels.prolong(program, queue, correction_coarse, u, N, M)

    # Post-relaxation
    for _ in range(num_relaxations):
        kernels.relax(program, queue, u, u_new, f, N, M)
        u, u_new = u_new, u
    
    return u, res_norm

# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Apply Poisson image editing using multigrid solver.')
    parser.add_argument('input_file', help='Input image file path')
    args = parser.parse_args()

    # Get input file name and extension
    filename, file_extension = os.path.splitext(args.input_file)

    # Read image and initialize parameters
    input_image = image.load(args.input_file)
    N = np.int32(input_image.shape[0])
    M = np.int32(input_image.shape[1])

    # OpenCL context
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    # Allocate memory
    u = cl_array.to_device(queue, np.zeros((N, M), dtype=np.float32))
    f = cl_array.to_device(queue, divergence_field.generate(input_image).astype(np.float32))  # Convert to float32

    # Load kernel
    with open('src/kernels.cl', 'r') as file_obj:
        fstr = "".join(file_obj.readlines())
    program = cl.Program(context, fstr).build()
    
    # Stopping criteria
    prev_res_norm = np.inf  # Initialize previous residual norm to infinity
    
    # Run multigrid
    for _ in range(num_cycles):
        u, res_norm = multigrid_vcycle(program, queue, u, f, N, M, 0, prev_res_norm)

        print("res_norm", res_norm, "res_norm < tolerance", res_norm < tolerance, "res_norm >= prev_res_norm", res_norm >= prev_res_norm)

        # Break if the solution has converged or is not improving
        if res_norm < tolerance or res_norm >= prev_res_norm:
            print("multigrid_vcycle has converged")
            break
        
        prev_res_norm = res_norm

    # Save output image
    output_image = u.get()
    output_file = filename + '_output' + file_extension
    image.save(output_file, output_image)

# If this script is the main module, run the main function
if __name__ == "__main__":
    main()
