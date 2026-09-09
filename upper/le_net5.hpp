#ifndef LE_NET5_HPP
#define LE_NET5_HPP

#include "cnn_base.h"
#include "convolve_layer.hpp"
#include "full_connect_layer.hpp"
#include "pool_layer.hpp"

namespace chr
{
    class le_net5 : public cnn_base
    {
    private:
        // Network layer definitions (LeNet-5 structure)
        convolve_layer conv1_;   // First convolution layer
        pool_layer pool1_;       // First pooling layer
        convolve_layer conv2_;   // Second convolution layer
        pool_layer pool2_;       // Second pooling layer
        full_connect_layer fc1_; // First fully connected layer
        full_connect_layer fc2_; // Second fully connected layer
        full_connect_layer fc3_; // Third fully connected layer (output layer)

    public:
        le_net5();
        Eigen::VectorXd forward(const std::vector<Eigen::MatrixXd> &input) override;
        std::vector<Eigen::MatrixXd> backward(size_t label, double learning_rate) override;
        void save(const std::filesystem::path &path) override;
        void load(const std::filesystem::path &path) override;
        std::string model_type() const override { return "LeNet-5"; }
    };
}

#endif // !LE_NET5_HPP
