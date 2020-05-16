//
// Created by Dima Zhylko on 15/05/2020.
//

#ifndef VULKAN_PERCEPTRON_RELU_H
#define VULKAN_PERCEPTRON_RELU_H

#include <layer.h>
#include <vulkan_init.h>

class ReLULayer: public Layer {
    struct dims{
        uint32_t batch_size;
        uint32_t inp_dim;
    } dim;

public:

    ReLULayer(VkDevice device, uint32_t queueFamilyIndex, VkPhysicalDevice physicalDevice,
         int batch_size, int input_dim, VkBuffer input);

    void forward(VkQueue& queue) override;
    void backward(VkQueue& queue) override;
    void forward_initialize(VkQueue& queue) override;
    void backward_initialize(VkBuffer& d_out) override;

    uint64_t get_output_offset() override{return forward_offsets[0];}
    uint32_t get_output_dim() override{return dim.inp_dim;};
};
#endif //VULKAN_PERCEPTRON_RELU_H