#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP

#include <cmath>
#include <stdexcept>

namespace chr
{

    // Supported activation function types.
    enum class activation_function_type
    {
        relu,
        lrelu,
        sigmoid,
        tanh,
        softmax
    };

    // Functor holding an activation function type.
    // operator() evaluates the activation function (used in forward pass).
    // operator[] evaluates the derivative of the activation function (used in backward pass).
    class activation_function
    {
    public:
        // Construct from an activation function type.
        activation_function(activation_function_type type = activation_function_type::relu)
            : type_(type)
        {
        }

        // Activation function itself.
        double operator()(double x) const
        {
            switch (type_)
            {
            case activation_function_type::relu:
                // ReLU: max(0, x)
                return x > 0.0 ? x : 0.0;
            case activation_function_type::lrelu:
                // Leaky ReLU: x for x > 0, 0.01 * x otherwise
                return x > 0.0 ? x : 0.01 * x;
            case activation_function_type::sigmoid:
                return 1.0 / (1.0 + std::exp(-x));
            case activation_function_type::tanh:
                return std::tanh(x);
            case activation_function_type::softmax:
                // Return the value as-is; softmax needs the whole vector.
                // This is handled by applying softmax over the full vector elsewhere.
                return x;
            default:
                throw std::runtime_error("unknown activation function type");
            }
        }

        // Derivative of the activation function.
        double operator[](double x) const
        {
            switch (type_)
            {
            case activation_function_type::relu:
                // ReLU derivative: 1 for x > 0, 0 otherwise
                return x > 0.0 ? 1.0 : 0.0;
            case activation_function_type::lrelu:
                // Leaky ReLU derivative: 1 for x > 0, 0.01 otherwise
                return x > 0.0 ? 1.0 : 0.01;
            case activation_function_type::sigmoid:
            {
                // sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
                double s = operator()(x);
                return s * (1.0 - s);
            }
            case activation_function_type::tanh:
            {
                // tanh derivative: 1 - tanh(x)^2
                double t = std::tanh(x);
                return 1.0 - t * t;
            }
            case activation_function_type::softmax:
                // Softmax derivative is handled by the caller for the whole vector.
                return 1.0;
            default:
                throw std::runtime_error("unknown activation function type");
            }
        }

    private:
        activation_function_type type_;
    };

}

#endif // !ACTIVATION_FUNCTION_HPP
