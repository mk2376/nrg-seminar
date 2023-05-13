// OpenCL kernel to perform relaxation step
__kernel void relax_kernel(__global float* u, __global float* u_new, int N, int M) {
    // Calculate the current thread index
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x > 0 && x < N-1 && y > 0 && y < M-1) {
        float u_new_value = 0.25f * (u[(x-1)*M + y] + u[(x+1)*M + y] + u[x*M + (y-1)] + u[x*M + (y+1)]);
        u_new[x*M + y] = u_new_value;
    } else {
        u_new[x*M + y] = u[x*M + y];
    }
}

// OpenCL kernel to calculate residual
__kernel void residual_kernel(__global float* u, __global float* f, __global float* res, int N, int M) {
    // Calculate the current thread index
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Compute residual only for interior points
    if (x > 0 && x < N-1 && y > 0 && y < M-1) {
        // Apply finite difference for Laplacian
        float laplacian_u = u[(x-1)*M + y] + u[(x+1)*M + y] + u[x*M + (y-1)] + u[x*M + (y+1)] - 4.0f*u[x*M + y];

        // Compute residual
        res[x*M + y] = laplacian_u - f[x*M + y];
    } else {
        res[x*M + y] = 0.0f;  // No residual at boundary
    }
}

// OpenCL kernel to perform restriction step
__kernel void restrict_kernel(__global float* r, __global float* r2, int N, int M) {
    // Calculate the current thread index
    int x = get_global_id(0);
    int y = get_global_id(1);

    int x2 = x*2;
    int y2 = y*2;

    int halfN = N / 2;
    int halfM = M / 2;

    // Check if the thread index is within the valid range
    if (x < halfN && y < halfM) {
        // Perform the restriction operation by averaging a 2x2 block
        r2[x*halfM + y] = 0.25f * (
            r[x2*M + y2] + 
            r[x2*M + (y2 + 1)] +
            r[(x2 + 1)*M + y2] +
            r[(x2 + 1)*M + (y2 + 1)]
        );
    }
}

// OpenCL kernel to perform prolongation step
__kernel void prolong_kernel(__global float* u, __global float* r2, int N, int M) {
    // Calculate the current thread index
    int x = get_global_id(0);
    int y = get_global_id(1);

    int x2 = x*2;
    int y2 = y*2;

    int halfN = N / 2;
    int halfM = M / 2;

    // Check if the thread index is within the valid range
    if (x < halfN && y < halfM) {
        // Perform the prolongation operation by nearest-neighbor interpolation
        float new_value = r2[x*halfM + y];

        u[x2*M + y2] = new_value;
        u[x2*M + (y2 + 1)] = new_value;
        u[(x2 + 1)*M + y2] = new_value;
        u[(x2 + 1)*M + (y2 + 1)] = new_value;
    }
}
