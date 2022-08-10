//
// Created by Dima Zhylko on 12/05/2020.
//

#ifndef VULKAN_PERCEPTRON_CONVOLUTION_H
#define VULKAN_PERCEPTRON_CONVOLUTION_H
#include "layer.h"
#include <string>
#include "vulkan_init.h"
#include <iostream>

class Convolution: public Layer{
    Tensor mWeight;
    Tensor mBias;

    float mScale;

    std::string initializer{};

    struct dims{
        uint32_t batch_size;
        uint32_t inp_dim;
        uint32_t output_dim;
    } dim;

public:
    Convolution(VkDevice device, uint32_t queueFamilyIndex, VkPhysicalDevice physicalDevice,
               int batch_size, int input_dim, int output_dim, VkBuffer input, float mScale=1, const std::string& initializer="He-et-al");

    void forward(VkQueue& queue) override;
    void backward(VkQueue& queue) override;
    void forward_initialize(VkQueue& queue) override;
    void backward_initialize(VkBuffer& d_out) override;

    Tensor& get_bias();
    Tensor& get_mWeight();

    [[nodiscard]] const dims& get_dims() const {return dim;};
    uint32_t get_output_dim() override {return dim.output_dim;}

    uint64_t get_output_offset() override {return forward_offsets[2];}

    uint32_t get_input_dim() override {return dim.inp_dim;}

    std::vector<std::vector<float>> get_results();
    std::vector<std::pair<Tensor, Tensor>> get_trainable_parameters() override;

    ~Convolution();
};
#endif //VULKAN_PERCEPTRON_DENSE_H
