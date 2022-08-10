#ifndef VULKAN_PERCEPTRON_OLP_H
#define VULKAN_PERCEPTRON_OLP_H
#include <vector>
#include <string>
#include "layer.h"
#include "dense.h"
#include "convolution.h"
#include "relu.h"
#include "softmax.h"
#include <vulkan_init.h>
#include <iostream>

class OLP {
    friend class Trainer;
    std::vector<Layer*> mLayers{};
    VkInstance mInstance;
    VkDebugUtilsMessengerEXT mDebugMessenger;
    VkPhysicalDevice mPhysicalDevice;
    uint32_t mQueueFamilyIndex;
    VkDevice mDevice;
    VkQueue mQueue;
    VkDeviceMemory mDeviceMemory;

    VkBuffer mInput;
    VkBuffer mD_input;
    std::vector<uint64_t> mOffsets;

    uint32_t mBatch_size;
    uint32_t mInput_size;

public:
    OLP();

    OLP(uint32_t mInput_size, uint32_t mBatch_size);

    void add(uint32_t layer_dim, uint32_t mInput_size=0, uint32_t mBatch_size=0);

    void forward_initialize();
    void forward(const std::vector<std::vector<float>>& batch);

    VkBuffer& get_output() {return mLayers[mLayers.size()-1]->get_output();}
    uint32_t get_output_dim() {return mLayers[mLayers.size()-1]->get_output_dim();}

    VkDeviceMemory& get_output_memory() {return mLayers[mLayers.size()-1]->get_forward_device_memory();}

    VkBuffer& get_d_output() {return mLayers[mLayers.size()-1]->get_d_output();}

    uint64_t get_output_offset(){return mLayers[mLayers.size()-1]->get_output_offset();}

    uint32_t get_batch_size(){return mBatch_size;}
    uint32_t get_layer_count(){return mLayers.size();}

    std::vector<std::pair<Tensor, Tensor>> get_trainable_parameters();

    float evaluate(std::vector<std::vector<float>>& X);

    ~OLP();

    // DEBUG
    VkDevice &get_device() {return mDevice;}
    VkPhysicalDevice& get_physicalDevice() {return mPhysicalDevice;}
    uint32_t get_queue_index() {return mQueueFamilyIndex;}
};

#endif //VULKAN_PERCEPTRON_OLP_H
