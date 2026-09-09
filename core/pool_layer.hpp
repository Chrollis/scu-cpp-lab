#ifndef POOL_LAYER_HPP
#define POOL_LAYER_HPP

#include <Eigen/Dense>

namespace chr
{
    enum class pooling_type
    {
        max,    // Max pooling
        average // Average pooling
    };

    class pool_layer
    {
    private:
        size_t core_size_;                         // Pooling kernel size
        size_t stride_;                            // Stride
        pooling_type type_;                        // Pooling type
        std::vector<Eigen::MatrixXd> input_;       // Input data
        std::vector<Eigen::MatrixXd> feature_map_; // Pooled feature maps
        std::vector<Eigen::MatrixXd> record_;      // Recorded locations of max pooling (used for backpropagation)
    public:
        pool_layer(size_t core_size, size_t stride, pooling_type type = pooling_type::max);
        std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd> &input);
        std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd> &gradient);

    private:
        void max_pooling(const Eigen::MatrixXd &input, Eigen::MatrixXd &output, Eigen::MatrixXd &record) const;
        void average_pooling(const Eigen::MatrixXd &input, Eigen::MatrixXd &output) const;
        Eigen::MatrixXd max_backward(const Eigen::MatrixXd &gradient, const Eigen::MatrixXd &record) const;
        Eigen::MatrixXd average_backward(const Eigen::MatrixXd &gradient);
    };

    // Example:
    // input: 16 * 10 * 10 (the same size of the record)
    // settings:
    // > core size: 2
    // > stride: 2
    // output: 16 * 5 * 5 (the same size of the feature map)
}

#endif // !POOL_LAYER_HPP
