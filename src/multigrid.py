import os
import argparse
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from scipy.ndimage import gaussian_filter

import image
import divergence_field
import kernels

tolerance = 1e-5  # Set desired tolerance
max_level = 10  # Set your desired maximum level

max_num_cycles = 100
num_relaxations = 1 # Baseline

# Function to apply multigrid V-cycle
def multigrid_vcycle(program, queue, u, f, N, M, level, prev_res_norm):
    # Initialize residual norm
    res_norm = np.inf
    
    # Make sure we are not going too deep
    if N < 50 or M < 50:
        return u, res_norm
    
    u_new = cl_array.to_device(queue, np.zeros((N, M), dtype=np.double))
    
    # Pre-relaxation
    for i in range(num_relaxations):
        kernels.relax(program, queue, u, u_new, f, N, M)
        u, u_new = u_new, u
        
        # output_image = u.get()
        # output_file = "outputs/relax_" + str(level) + "_" + str(i) + ".jpg"
        # image.save(output_file, output_image)

    if level < max_level:
        # Calculate residual
        res = cl_array.to_device(queue, np.zeros((N, M), dtype=np.double))  # Buffer for the residual
        kernels.residual(program, queue, u, f, res, N, M)

        res_norm = np.linalg.norm(res.get())
        factor = 2
        
        # Apply restriction
        halfN = np.int32(N//factor)
        halfM = np.int32(M//factor)
        f_coarse = cl_array.to_device(queue, np.zeros((halfN, halfM), dtype=np.double))  # RHS at the coarser level
        res_coarse = cl_array.to_device(queue, np.zeros((halfN, halfM), dtype=np.double))  # Residual at the coarser level
        kernels.restrict(program, queue, f, f_coarse, N, M)
        kernels.restrict(program, queue, res, res_coarse, N, M)
        
        # output_image = f_coarse.get()
        # output_file = "outputs/restrict_f_" + str(level) + ".jpg"
        # image.save(output_file, output_image)
        
        # output_image = res_coarse.get()
        # output_file = "outputs/restrict_res_" + str(level) + ".jpg"
        # image.save(output_file, output_image)

        # Recursive call
        correction_coarse, res_norm_level = multigrid_vcycle(program, queue, res_coarse, f_coarse, halfN, halfM, level+1, prev_res_norm)
        if res_norm_level < res_norm:
            res_norm = res_norm_level
            
        # output_image = correction_coarse.get()
        # output_file = "outputs/correction_coarse_" + str(level) + ".jpg"
        # image.save(output_file, output_image)

        # Apply prolongation
        kernels.prolong(program, queue, correction_coarse, u, N, M)
        
        # output_image = u.get()
        # output_file = "outputs/prolong_" + str(level) + ".jpg"
        # image.save(output_file, output_image)

    # Post-relaxation
    # for _ in range(num_relaxations-level):
    #    kernels.relax(program, queue, u, u_new, f, N, M)
    #    u, u_new = u_new, u
    
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
    u = cl_array.to_device(queue, np.zeros((N, M), dtype=np.double))
    f = cl_array.to_device(queue, divergence_field.generate(input_image).astype(np.double)) # Convert to double

    # Load kernel
    with open('src/kernels.cl', 'r') as file_obj:
        fstr = "".join(file_obj.readlines())
    program = cl.Program(context, fstr).build()
    
    # Stopping criteria
    prev_res_norm = np.inf  # Initialize previous residual norm to infinity
    
    # Run multigrid
    for i in range(max_num_cycles):
        u, res_norm = multigrid_vcycle(program, queue, u, f, N, M, 0, prev_res_norm)

        print("res_norm", res_norm, "prev_res_norm", prev_res_norm, "res_norm < tolerance", res_norm < tolerance, "res_norm >= prev_res_norm", res_norm >= prev_res_norm)

        # Break if the solution has converged or is not improving
        if res_norm < tolerance or res_norm >= prev_res_norm:
            print("multigrid_vcycle has converged after", i+1, "cycles")
            break
        
        prev_res_norm = res_norm

    # Save intermediate image
    output_image = u.get()
    output_file = "outputs/intermediate_output" + file_extension
    image.save(output_file, output_image)

    # Normalize to [0, 1]
    output_image = (output_image - np.min(output_image)) / (np.max(output_image) - np.min(output_image))
    
    # Invert
    output_image = (output_image - 1)*-1
    
    # Filter
    output_image = np.where(output_image < 0.78, 0, output_image)
    
    output_file = "outputs/final_output" + file_extension
    image.save(output_file, output_image)

# If this script is the main module, run the main function
if __name__ == "__main__":
    main()
