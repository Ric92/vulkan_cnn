//
// Created by Dima Zhylko on 12/05/2020.
//

#include <convolution.h>
#include <random>

void Convolution::forward(VkQueue &queue) {
    submitTask(queue, &forwardCommandBuffer);
}

void Convolution::forward_initialize(VkQueue &queue) {

    std::vector<VkBuffer*> buffers{&mWeight.get_buffer(), &mBias.get_buffer(), &output};
    allocateAndBindBuffers(device, physicalDevice, buffers, forwardDeviceMemory, forward_offsets);

    createPipelineLayout(device, 4, forwardSetLayout, forwardPipelineLayout, sizeof(dims));
    createComputePipeline(device, "../shaders/dense.comp.spv", forwardPipelineLayout, forwardPipeline);

    buffers.insert(buffers.begin(), &input);

    allocateDescriptorSet(device, buffers, forwardDescriptorPool, forwardSetLayout, forwardDescriptorSet);
    createCommandPoolAndBuffer(device, queueFamilyIndex, forwardCommandPool, forwardCommandBuffer);

    recordComputePipeline(forwardCommandBuffer, forwardPipelineLayout, sizeof(dims), reinterpret_cast<void*>(&dim),
            forwardPipeline,forwardDescriptorSet, (dim.batch_size+15)/16, (dim.output_dim+15)/16, 1);
}

Convolution::Convolution(VkDevice device, uint32_t queueFamilyIndex, VkPhysicalDevice physicalDevice,
        int batch_size, int input_dim, int output_dim, VkBuffer input, float mScale, const std::string& initializer) {
    this->mScale = mScale;

    this->input = input;
    this->initializer = initializer;
    dim.batch_size = batch_size;
    dim.inp_dim = input_dim;
    dim.output_dim = output_dim;

    this->device = device;
    this->queueFamilyIndex = queueFamilyIndex;
    this->physicalDevice = physicalDevice;

    createBuffer(device, queueFamilyIndex, mWeight.get_buffer(), dim.inp_dim, dim.output_dim);
    mWeight.set_height(dim.inp_dim);
    mWeight.set_width(dim.output_dim);

    createBuffer(device, queueFamilyIndex, mBias.get_buffer(), dim.output_dim, 1);
    mBias.set_height(dim.output_dim);
    mBias.set_width(1);

    createBuffer(device, queueFamilyIndex, output, dim.batch_size, dim.output_dim);
}


std::vector<std::vector<float>> Convolution::get_results(){
    char *data = nullptr;
    if(vkMapMemory(this->device, forwardDeviceMemory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&data)) != VK_SUCCESS){
        throw std::runtime_error("failed to map device memory");
    }

    uint32_t* p_labels = reinterpret_cast<uint32_t*>(data + forward_offsets[0]);

    vkUnmapMemory(this->device, forwardDeviceMemory);
}

Tensor &Convolution::get_bias() {
    return mBias;
}

Tensor &Convolution::get_mWeight() {
    return mWeight;
}

Convolution::~Convolution() {
    vkDestroyCommandPool(device, forwardCommandPool, nullptr);

    vkFreeMemory(device, forwardDeviceMemory, nullptr);

    vkDestroyBuffer(device, output, nullptr);
    vkDestroyBuffer(device, mWeight.get_buffer(), nullptr);
    vkDestroyBuffer(device, mBias.get_buffer(), nullptr);

    vkDestroyDescriptorPool(device, forwardDescriptorPool, nullptr);

    vkDestroyPipeline(device, forwardPipeline, nullptr);

    vkDestroyPipelineLayout(device, forwardPipelineLayout, nullptr);

    vkDestroyDescriptorSetLayout(device, forwardSetLayout, nullptr);

}

void Convolution::backward_initialize(VkBuffer& d_out){}

std::vector<std::pair<Tensor, Tensor>> Convolution::get_trainable_parameters(){}

void Convolution::backward(VkQueue& queue){}