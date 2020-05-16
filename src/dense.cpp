//
// Created by Dima Zhylko on 12/05/2020.
//

#include <dense.h>
#include <random>

VkBuffer &DenseLayer::get_weight() {
    return weight;
}

VkBuffer &DenseLayer::get_bias() {
    return bias;
}

void DenseLayer::forward(VkQueue &queue) {
    submitTask(queue, &forwardCommandBuffer);
}

void DenseLayer::forward_initialize(VkQueue &queue) {

    std::cout<<"dense forward init"<<std::endl;

    std::vector<VkBuffer*> buffers{&weight, &bias, &output};
    allocateAndBindBuffers(device, physicalDevice, buffers, forwardDeviceMemory, forward_offsets);

    createPipelineLayout(device, 4, forwardSetLayout, forwardPipelineLayout, sizeof(dims));
    createComputePipeline(device, "../shaders/dense.comp.spv", forwardPipelineLayout, forwardPipeline);

    buffers.insert(buffers.begin(), &input);

    allocateDescriptorSet(device, buffers, forwardDescriptorPool, forwardSetLayout, forwardDescriptorSet);
    createCommandPoolAndBuffer(device, queueFamilyIndex, forwardCommandPool, forwardCommandBuffer);

    recordComputePipeline(forwardCommandBuffer, forwardPipelineLayout, sizeof(dims), reinterpret_cast<void*>(&dim),
            forwardPipeline,forwardDescriptorSet, (dim.batch_size+15)/16, (dim.output_dim+15)/16, 1);

    // TODO: actual xavier initialization
    char* data = nullptr;
    if(vkMapMemory(device, forwardDeviceMemory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&data)) != VK_SUCCESS){
        throw std::runtime_error("failed to map device memory");
    }

    std::cout<<"obtain pointers"<<std::endl;

    float* weight = reinterpret_cast<float*>(data + forward_offsets[0]);
    float* bias = reinterpret_cast<float*>(data + forward_offsets[1]);

    std::cout<<"init bias"<<std::endl;
    for(int i = 0;i<dim.output_dim;i++){
        bias[i] = 0;
    }
    std::cout<<"end init bias"<<std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, 1.0);

    std::cout<<"init weight"<<std::endl;
    for(int i = 0;i<dim.inp_dim;i++){
        for(int j = 0;j<dim.output_dim;j++){
            weight[i*dim.output_dim + j] = scale * dis(gen);
        }
    }
    std::cout<<"end init weight"<<std::endl;
    vkUnmapMemory(device, forwardDeviceMemory);

}

DenseLayer::DenseLayer(VkDevice device, uint32_t queueFamilyIndex, VkPhysicalDevice physicalDevice,
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

    createBuffer(device, queueFamilyIndex, weight, dim.inp_dim, dim.output_dim);
    createBuffer(device, queueFamilyIndex, bias, dim.output_dim, 1);
    createBuffer(device, queueFamilyIndex, output, dim.batch_size, dim.output_dim);
}

void DenseLayer::backward(VkQueue &queue) {
    submitTask(queue, &backwardCommandBuffer, false);
    submitTask(queue, &backwardWeightCommandBuffer, false);
    submitTask(queue, &backwardBiasCommandBuffer);
}

void DenseLayer::backward_initialize(VkBuffer &d_out) {
    d_output = d_out;

    createBuffer(device, queueFamilyIndex, d_input, dim.batch_size, dim.inp_dim);
    createBuffer(device, queueFamilyIndex, d_weight, dim.inp_dim, dim.output_dim);
    createBuffer(device, queueFamilyIndex, d_bias, dim.output_dim, 1);

    std::vector<VkBuffer*> buffers{&d_input, &d_weight, &d_bias};

    allocateAndBindBuffers(device, physicalDevice, buffers, backwardDeviceMemory, backward_offsets);

    createPipelineLayout(device, 7, backwardSetLayout, backwardPipelineLayout, sizeof(dims));
    createComputePipeline(device, "../shaders/d_dense.comp.spv", backwardPipelineLayout, backwardPipeline);
    createComputePipeline(device, "../shaders/d_dense_w.comp.spv", backwardPipelineLayout, backwardWeightPipeline);
    createComputePipeline(device, "../shaders/d_dense_b.comp.spv", backwardPipelineLayout, backwardBiasPipeline);


    buffers.insert(buffers.begin(), &input);
    buffers.insert(buffers.begin(), &weight);
    buffers.insert(buffers.begin(), &bias);
    buffers.push_back(&d_output);

    allocateDescriptorSet(device, buffers, backwardDescriptorPool, backwardSetLayout, backwardDescriptorSet);
    createCommandPoolAndBuffer(device, queueFamilyIndex, backwardCommandPool, backwardCommandBuffer);
    createCommandPoolAndBuffer(device, queueFamilyIndex, backwardWeightCommandPool, backwardWeightCommandBuffer);
    createCommandPoolAndBuffer(device, queueFamilyIndex, backwardBiasCommandPool, backwardBiasCommandBuffer);

    recordComputePipeline(backwardCommandBuffer, backwardPipelineLayout, sizeof(dims), reinterpret_cast<void*>(&dim),
                          backwardPipeline,backwardDescriptorSet, (dim.batch_size+15)/16, (dim.inp_dim+15)/16, 1);

    recordComputePipeline(backwardWeightCommandBuffer, backwardPipelineLayout, sizeof(dims), reinterpret_cast<void*>(&dim),
                          backwardWeightPipeline,backwardDescriptorSet, (dim.inp_dim+15)/16, (dim.output_dim+15)/16, 1);

    recordComputePipeline(backwardBiasCommandBuffer, backwardPipelineLayout, sizeof(dims), reinterpret_cast<void*>(&dim),
                          backwardBiasPipeline,backwardDescriptorSet, (dim.output_dim+31)/32, 1, 1);
}