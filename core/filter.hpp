#ifndef FILTER_HPP
#define FILTER_HPP

#include <Eigen/Dense>
#include <filesystem>

namespace chr
{
    // Convolution kernel filter class
    class filter
    {
    private:
        size_t core_size_; // Kernel size
        size_t channels_;  // Number of input channels
    public:
        double bias;                          // Bias term
        std::vector<Eigen::MatrixXd> kernels; // Collection of kernel matrices
    public:
        filter(size_t channels, size_t core_size);
        void initialize_gausz(double stddev);      // Gaussian initialization
        void initialize_xavier(size_t input_size); // Xavier initialization
        void initialize_He(size_t input_size);     // He initialization
    };
}

#endif // !FILTER_HPP
