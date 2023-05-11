#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <filesystem>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// CUDA kernels for multigrid relaxation, restriction, and prolongation
__global__ void relaxKernel(float *u, float *f, float *unew, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < N - 1 && y > 0 && y < N - 1) {
        unew[y * N + x] = 0.25f * (u[(y - 1) * N + x] + u[(y + 1) * N + x] +
                                   u[y * N + x - 1] + u[y * N + x + 1] - f[y * N + x]);
    }
}

__global__ void restrictKernel(float *r, float *r2, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < N / 2 && y > 0 && y < N / 2) {
        r2[y * (N / 2) + x] = 0.25f * (r[(2 * y - 1) * N + 2 * x] + r[(2 * y + 1) * N + 2 * x] +
                                      r[2 * y * N + 2 * x - 1] + r[2 * y * N + 2 * x + 1]);
    }
}

__global__ void prolongKernel(float *u, float *u2, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < N - 1 && y > 0 && y < N - 1) {
        float value = 0.25f * (u2[(y / 2) * (N / 2) + x / 2] +
                               (x % 2 == 1 ? u2[(y / 2) * (N / 2) + (x / 2) + 1] : 0) +
                               (y % 2 == 1 ? u2[((y / 2) + 1) * (N / 2) + x / 2] : 0) +
                               (x % 2 == 1 && y % 2 == 1 ? u2[((y / 2) + 1) * (N / 2) + (x / 2) + 1] : 0));
        u[y * N + x] += value;
    }
}

// Multigrid solver function
void multigrid(float *u, float *f, int N, int num_cycles, int num_relaxations) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    dim3 gridDim2((N / 2 + blockDim.x - 1) / blockDim.x, (N / 2 + blockDim.y - 1) / blockDim.y);

    float *d_u, *d_f, *d_unew, *d_r, *d_r2, *d_u2;
    cudaMalloc(&d_u, N * N * sizeof(float));
    cudaMalloc(&d_f, N * N * sizeof(float));
    cudaMalloc(&d_unew, N * N * sizeof(float));
    cudaMalloc(&d_r, N * N * sizeof(float));
    cudaMalloc(&d_r2, (N / 2) * (N / 2) * sizeof(float));
    cudaMalloc(&d_u2, (N / 2) * (N / 2) * sizeof(float));

    cudaMemcpy(d_u, u, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, f, N * N * sizeof(float), cudaMemcpyHostToDevice);

    for (int cycle = 0; cycle < num_cycles; ++cycle) {
        for (int relax = 0; relax < num_relaxations; ++relax) {
            relaxKernel<<<gridDim, blockDim>>>(d_u, d_f, d_unew, N);
            cudaDeviceSynchronize();
            std::swap(d_u, d_unew);
        }

        relaxKernel<<<gridDim, blockDim>>>(d_u, d_f, d_r, N);
        cudaDeviceSynchronize();

        restrictKernel<<<gridDim2, blockDim>>>(d_r, d_r2, N);
        cudaDeviceSynchronize();

        cudaMemcpy(d_u2, d_r2, (N / 2) * (N / 2) * sizeof(float), cudaMemcpyDeviceToDevice);

        if (N > 4) {
            multigrid(d_u2, d_r2, N / 2, num_cycles, num_relaxations);
        }

        prolongKernel<<<gridDim, blockDim>>>(d_u, d_u2, N);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(u, d_u, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_u);
    cudaFree(d_f);
    cudaFree(d_unew);
    cudaFree(d_r);
    cudaFree(d_r2);
    cudaFree(d_u2);
}

// Helper function to compute gradient field from an image
void computeGradientField(const cv::Mat &image, cv::Mat &gradient) {
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            float gradientX = x < image.cols - 1 ? image.at<float>(y, x + 1) - image.at<float>(y, x) : 0;
            float gradientY = y < image.rows - 1 ? image.at<float>(y + 1, x) - image.at<float>(y, x) : 0;
            gradient.at<cv::Vec2f>(y, x) = cv::Vec2f(gradientX, gradientY);
        }
    }
}

int main(int argc, char *argv[]) {
     if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image_path>" << std::endl;
        return -1;
    }

    // Read the input image
    std::string inputImagePath = argv[1];
    cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);

    if (inputImage.empty()) {
        std::cerr << "Could not open or find the image: " << inputImagePath << std::endl;
        return -1;
    }

    // Convert input image to floating-point format and normalize to [0, 1]
    inputImage.convertTo(inputImage, CV_32F, 1.0 / 255.0);

    // Compute the gradient field of the input image
    cv::Mat gradientField(inputImage.size(), CV_32FC2);
    computeGradientField(inputImage, gradientField);

    // Solve the Poisson equation using multigrid method
    int N = inputImage.rows;
    int num_cycles = 5;
    int num_relaxations = 5;
    float *u = (float *)calloc(N * N, sizeof(float));
    float *f = (float *)calloc(N * N, sizeof(float));

    // Convert the gradient field to a single-channel image for the Poisson solver
    for (int i = 0; i < N * N; ++i) {
        f[i] = gradientField.at<cv::Vec2f>(i)[0] + gradientField.at<cv::Vec2f>(i)[1];
    }

    multigrid(u, f, N, num_cycles, num_relaxations);

    // Normalize the output and save as an image
    float minVal, maxVal;
    cv::Mat outputImage(inputImage.size(), CV_32F, u);
    cv::minMaxLoc(outputImage, &minVal, &maxVal);
    outputImage = (outputImage - minVal) / (maxVal - minVal);

    free(u);
    free(f);

    // Convert the floating-point output image back to 8-bit format
    outputImage.convertTo(outputImage, CV_8U, 255.0);

    // Create the output file name by appending "_output" to the input file name
    std::filesystem::path inputPath(inputImagePath);
    std::string outputImagePath = (inputPath.parent_path() / (inputPath.stem().string() + "_output")).string() + inputPath.extension().string();

    cv::imwrite(outputImagePath, outputImage);

    return 0;
}
