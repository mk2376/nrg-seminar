// OpenCL kernel to perform relaxation step
__kernel void relax_kernel(__global float* u, __global float* f, __global float* unew, int N, int M) {
    // Calculate the current thread index
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x > 0 && x < N-1 && y > 0 && y < M-1) {
        float u_new_value = 0.25f * (u[(x-1)*M + y] + u[(x+1)*M + y] + u[x*M + (y-1)] + u[x*M + (y+1)] - f[x*M + y]);
        unew[x*M + y] = u_new_value;
    } else {
        unew[x*M + y] = u[x*M + y];
    }
}

// OpenCL kernel to perform restriction step
__kernel void restrict_kernel(__global float* r, __global float* r2, int N, int M) {
    // Calculate the current thread index
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Check if the thread index is within the valid range
    if (x > 0 && x < N - 1 && y > 0 && y < M - 1) {
        // Perform the restriction operation by averaging four neighboring values
        r2[(y / 2) * (M / 2) + (x / 2)] = 0.25f * (r[(y - 1) * N + x] + r[(y + 1) * N + x] + r[y * N + x - 1] + r[y * N + x + 1]);
    }
}

// OpenCL kernel to perform prolongation step
__kernel void prolong_kernel(__global float* u, __global float* u2, int N, int M) {
    // Calculate the current thread index
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Check if the thread index is within the valid range
    if (x > 0 && x < N - 1 && y > 0 && y < M - 1) {
        // Perform the prolongation operation by linear interpolation
        u2[y * 2 * M + x * 2] = 0.5f * (u[(y - 1) * N + x] + u[y * N + x]);
        u2[(y * 2 + 1) * M + x * 2] = 0.5f * (u[y * N + x] + u[(y + 1) * N + x]);
        u2[y * 2 * M + x * 2 + 1] = 0.5f * (u[y * N + x] + u[y * N + x + 1]);
        u2[(y * 2 + 1) * M + x * 2 + 1] = 0.5f * (u[(y + 1) * N + x] + u[y * N + x + 1]);
    }
}
