# nrg-seminar

Scattered data approximation with multigrid relaxation

## Instructions

- Interpolate/approximate scattered data
- We want a fast iterative approach so that it can run on a GPU
- Use a Poisson equation
- Certain classes of PDEs (including Poisson) converge slowly
- Low-frequency errors are eliminated slowly, high-frequency errors are eliminated quickly
- Employ a multi-resolution approach (multigrid) so that all error modes are eliminated quickly

In many cases in computer graphics, we are faced with unstructured data samples, such as point clouds or path
tracing rays. Given a finite set of scattered samples, your task is to guess the function values in the entire domain.
Many methods exist for this task, taking significantly different approaches. One of them involves solving the Laplacian
equation on a grid (e.g., pixels in an image). However, just straightforwardly discretizing the equation results in
extremely slow convergence. The goal of this seminar is to explore and implement the multigrid method on the GPU.
The multigrid method effectively accelerates the convergence of low-frequency signals by discretizing and solving the
equation in different resolutions and then systematically distributing the residual error.

References:
[1] K. Anjyo, J. P. Lewis, and F. Pighin, “Scattered data interpolation for computer graphics”,
ACM SIGGRAPH 2014 Courses, ACM Press, 2014, doi: 10.1145/2614028.2615425
