#ifndef MNIST_DATA_HPP
#define MNIST_DATA_HPP

#include "image_process.hpp"
#include <filesystem>

namespace chr
{
    // MNIST dataset data class, used to store handwritten digit images and labels
    class mnist_data
    {
    private:
        Eigen::MatrixXd image_; // 28x28 image matrix, pixel values normalized to 0 or 1
        size_t label_;          // Digit label (0-9)
    public:
        mnist_data(const Eigen::MatrixXd &image, size_t label);
        mnist_data(Eigen::MatrixXd &&image, size_t label);
        const Eigen::MatrixXd &image() const { return image_; }                            // Get the image matrix
        cv::Mat cv_image() const { return image_process::eigen_matrix_to_cv_mat(image_); } // Convert to an OpenCV image format
        size_t label() const { return label_; }                                            // Get the label
        bool is_legal() const;                                                             // Check whether the data is legal (28x28 size)
    public:
        static unsigned swap_endian(unsigned val);                                                  // Byte order conversion (big endian to little endian)
        static unsigned check_mnist_file(std::ifstream &mnist_images, std::ifstream &mnist_labels); // Validate the MNIST file format
    public:
        // Read data from the MNIST files
        static std::vector<mnist_data> obtain_data(const std::filesystem::path &mnist_image_path, const std::filesystem::path &mnist_label_path, size_t offset = 0, size_t size = 60000);
        static void write_data(const std::filesystem::path &image_path, const std::filesystem::path &label_path, const std::vector<mnist_data> &datas);
    };
}

#endif // !MNIST_DATA_HPP
