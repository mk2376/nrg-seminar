import numpy as np

# Function to perform relaxation in the multigrid method
def relax(program, queue, u, u_new, f, N, M):
    # Compute grid spacing
    h_x = np.double(1.0 / (N - 1))
    h_y = np.double(1.0 / (M - 1))
    
    global_work_size = (N, M)
    local_work_size = None # (2, 2)  # or whatever value is appropriate for your hardware
    program.relax_kernel(queue, global_work_size, local_work_size, u.data, u_new.data, f.data, N, M, h_x, h_y)

# Function to calculate residual
def residual(program, queue, u, f, res, N, M):
    global_work_size = (N, M)
    local_work_size = None # (2, 2)  # or whatever value is appropriate for your hardware
    program.residual_kernel(queue, global_work_size, local_work_size, u.data, f.data, res.data, N, M)

# Function to perform restriction in the multigrid method
def restrict(program, queue, res, r2, N, M):
    factor = 2;
    
    halfN = np.int32(N//factor);
    halfM = np.int32(M//factor);
    global_work_size = (halfN, halfM)
    local_work_size = None # (2, 2) # or whatever value is appropriate for your hardware
    program.restrict_kernel(queue, global_work_size, local_work_size, res.data, r2.data, N, M)

# Function to perform prolongation in the multigrid method
def prolong(program, queue, correction_coarse, u, N, M):
    factor = 2;
        
    halfN = np.int32(N//factor);
    halfM = np.int32(M//factor);
    global_work_size = (halfN, halfM)
    local_work_size = None # (2, 2)  # or whatever value is appropriate for your hardware
    program.prolong_kernel(queue, global_work_size, local_work_size, correction_coarse.data, u.data, N, M)
