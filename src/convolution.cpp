//
// Created by Dima Zhylko on 12/05/2020.
//

#include <convolution.h>
#include <random>

void Convolution::forward(VkQueue &queue) {
    submitTask(queue, &forwardCommandBuffer);
}

void Convolution::forward_initialize(VkQueue &queue) {

    std::vector<VkBuffer*> buffers{&weight.get_buffer(), &bias.get_buffer(), &output};
    allocateAndBindBuffers(device, physicalDevice, buffers, forwardDeviceMemory, forward_offsets);

    createPipelineLayout(device, 4, forwardSetLayout, forwardPipelineLayout, sizeof(dims));
    createComputePipeline(device, "../shaders/dense.comp.spv", forwardPipelineLayout, forwardPipeline);

    buffers.insert(buffers.begin(), &input);

    allocateDescriptorSet(device, buffers, forwardDescriptorPool, forwardSetLayout, forwardDescriptorSet);
    createCommandPoolAndBuffer(device, queueFamilyIndex, forwardCommandPool, forwardCommandBuffer);

    recordComputePipeline(forwardCommandBuffer, forwardPipelineLayout, sizeof(dims), reinterpret_cast<void*>(&dim),
            forwardPipeline,forwardDescriptorSet, (dim.batch_size+15)/16, (dim.output_dim+15)/16, 1);

    // TODO: actual He-et-al initialization
    char* data = nullptr;
    if(vkMapMemory(device, forwardDeviceMemory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&data)) != VK_SUCCESS){
        throw std::runtime_error("failed to map device memory");
    }

    if(initializer == "He-et-al"){
        float* weight = reinterpret_cast<float*>(data + forward_offsets[0]);
        float* bias = reinterpret_cast<float*>(data + forward_offsets[1]);

        for(int i = 0;i<dim.output_dim;i++){
            bias[i] = 0;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0.0, 2.0/dim.inp_dim);

        for(int i = 0;i<dim.inp_dim;i++){
            for(int j = 0;j<dim.output_dim;j++){
                weight[i*dim.output_dim + j] = dis(gen);
            }
        }
    } else {
        throw std::invalid_argument("unknown initializer");
    }
    vkUnmapMemory(device, forwardDeviceMemory);

}

Convolution::Convolution(VkDevice device, uint32_t queueFamilyIndex, VkPhysicalDevice physicalDevice,
        int batch_size, int input_dim, int output_dim, VkBuffer input, float scale, const std::string& initializer) {
    this->scale = scale;

    this->input = input;
    this->initializer = initializer;
    dim.batch_size = batch_size;
    dim.inp_dim = input_dim;
    dim.output_dim = output_dim;

    this->device = device;
    this->queueFamilyIndex = queueFamilyIndex;
    this->physicalDevice = physicalDevice;

    createBuffer(device, queueFamilyIndex, weight.get_buffer(), dim.inp_dim, dim.output_dim);
    weight.set_height(dim.inp_dim);
    weight.set_width(dim.output_dim);

    createBuffer(device, queueFamilyIndex, bias.get_buffer(), dim.output_dim, 1);
    bias.set_height(dim.output_dim);
    bias.set_width(1);

    createBuffer(device, queueFamilyIndex, output, dim.batch_size, dim.output_dim);
}

Tensor &Convolution::get_bias() {
    return bias;
}

Tensor &Convolution::get_weight() {
    return weight;
}

Convolution::~Convolution() {
    vkDestroyCommandPool(device, forwardCommandPool, nullptr);

    vkFreeMemory(device, forwardDeviceMemory, nullptr);

    vkDestroyBuffer(device, output, nullptr);
    vkDestroyBuffer(device, weight.get_buffer(), nullptr);
    vkDestroyBuffer(device, bias.get_buffer(), nullptr);

    vkDestroyDescriptorPool(device, forwardDescriptorPool, nullptr);

    vkDestroyPipeline(device, forwardPipeline, nullptr);

    vkDestroyPipelineLayout(device, forwardPipelineLayout, nullptr);

    vkDestroyDescriptorSetLayout(device, forwardSetLayout, nullptr);

}

void Convolution::backward_initialize(VkBuffer& d_out){}

std::vector<std::pair<Tensor, Tensor>> Convolution::get_trainable_parameters(){}

void Convolution::backward(VkQueue& queue){}