#ifndef CNN_BASE_H
#define CNN_BASE_H

#include "mnist_data.hpp"
#include <QObject>

namespace chr
{

    class cnn_base : public QObject
    {
        Q_OBJECT
    public:
        virtual ~cnn_base() = default;
        virtual Eigen::VectorXd forward(const std::vector<Eigen::MatrixXd> &input) = 0;
        virtual std::vector<Eigen::MatrixXd> backward(size_t label, double learning_rate) = 0;
        double train(const std::vector<mnist_data> &dataset, size_t epochs, double learning_rate, bool show_detail = 0);
        size_t predict(const Eigen::VectorXd &output); // Prediction function
        static std::string model_type_of(const std::filesystem::path &path);
        virtual void save(const std::filesystem::path &path) = 0;
        virtual void load(const std::filesystem::path &path) = 0;
        virtual std::string model_type() const = 0;

    signals:
        void inform(const QString &output);
        void train_details(double progress, double loss, size_t correct, size_t total);

    protected:
        Eigen::VectorXd flatten(const std::vector<Eigen::MatrixXd> &matrixs);                                                  // Flatten operation (multi-dimensional to 1D)
        std::vector<Eigen::MatrixXd> counterflatten(const Eigen::VectorXd &vector, size_t channels, size_t rows, size_t cols); // Reverse flatten operation (1D to multi-dimensional)
        double cross_entropy_loss(const Eigen::VectorXd &output, size_t label);                                                // Cross-entropy loss function
    };
}

#endif // CNN_BASE_H
