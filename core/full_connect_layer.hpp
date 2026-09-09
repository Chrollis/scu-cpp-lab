#ifndef FULL_CONNECT_LAYER_HPP
#define FULL_CONNECT_LAYER_HPP

#include "activation_function.hpp"
#include <Eigen/Dense>

namespace chr
{
    class full_connect_layer
    {
    private:
        size_t in_size_;                 // Input dimension
        size_t out_size_;                // Output dimension
        activation_function afunc_;      // Activation function
        Eigen::MatrixXd weights_;        // Weight matrix
        Eigen::VectorXd biases_;         // Bias vector
        Eigen::VectorXd input_;          // Input data
        Eigen::VectorXd gradient_;       // Gradient
        Eigen::VectorXd feature_vector_; // Feature vector (after activation)
        Eigen::VectorXd linear_outcome_; // Linear output (before activation)
    public:
        full_connect_layer(size_t in_size, size_t out_size, activation_function_type activate_type = activation_function_type::relu);
        Eigen::VectorXd forward(const Eigen::VectorXd &input);
        Eigen::VectorXd backward(const Eigen::VectorXd &gradient, double learning_rate, bool is_output_layer = false, size_t label = 0);
        void weights_update(double learning_rate);
        void save(std::ostream &file) const;
        void load(std::istream &file);

    private:
        void initialize_weights();
    };

    // Example:
    // input: 400 * 1
    // output : 120 * 1
    // weights: 120 * 400
    // bias: 120 * 1
}

#endif // !FULL_CONNECT_LAYER_HPP
