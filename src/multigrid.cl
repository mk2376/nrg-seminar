// OpenCL kernel to perform relaxation step
__kernel void relax(__global float* u, __global float* f, __global float* unew, int N, int M) {
    // Calculate the current thread index
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Check if the thread index is within the valid range
    if (x > 0 && x < N - 1 && y > 0 && y < M - 1) {
        // Perform the relaxation operation
        unew[y * M + x] = 0.25f * (u[(y - 1) * M + x] + u[(y + 1) * M + x] +
                                   u[y * M + x - 1] + u[y * M + x + 1] - f[y * M + x]);
    }
}

// OpenCL kernel to perform restriction step
__kernel void restrict_kernel(__global float* r, __global float* r2, int N) {
    // Calculate the current thread index
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Check if the thread index is within the valid range
    if (x > 0 && x < N - 1 && y > 0 && y < N - 1) {
        // Perform the restriction operation by averaging four neighboring values
        r2[(y / 2) * (N / 2) + (x / 2)] = 0.25f * (r[(y - 1) * N + x] + r[(y + 1) * N + x] + r[y * N + x - 1] + r[y * N + x + 1]);
    }
}

// OpenCL kernel to perform prolongation step
__kernel void prolong(__global float* u, __global float* u2, int N) {
    // Calculate the current thread index
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Check if the thread index is within the valid range
    if (x > 0 && x < N - 1 && y > 0 && y < N - 1) {
        // Perform the prolongation operation by linear interpolation
        u2[y * 2 * N + x * 2] = 0.5f * (u[(y - 1) * N + x] + u[y * N + x]);
        u2[(y * 2 + 1) * N + x * 2] = 0.5f * (u[y * N + x] + u[(y + 1) * N + x]);
        u2[y * 2 * N + x * 2 + 1] = 0.5f * (u[y * N + x] + u[y * N + x + 1]);
        u2[(y * 2 + 1) * N + x * 2 + 1] = 0.5f * (u[(y + 1) * N + x] + u[y * N + x + 1]);
    }
}
