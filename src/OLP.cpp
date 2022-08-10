#include <OLP.h>

OLP::OLP(uint32_t mInput_size, uint32_t mBatch_size) {
    setup_vulkan(mInstance, mDebugMessenger, mPhysicalDevice, mQueueFamilyIndex, mDevice, mQueue);

    add(256, mInput_size, mBatch_size);
    
}

void OLP::add(uint32_t layer_dim, uint32_t mInput_size, uint32_t mBatch_size) {
    if(mLayers.empty() && (mInput_size == 0 || mBatch_size == 0)){
        throw std::invalid_argument("first layer should specify mInput and batch size greater than 0");
    }

    uint32_t input_dim;

    VkBuffer input_buffer;

    if(mLayers.empty()){
        input_dim = mInput_size;
        this->mBatch_size = mBatch_size;
        this->mInput_size = mInput_size;

        createBuffer(mDevice, mQueueFamilyIndex, mInput, mBatch_size, mInput_size);
        //createBuffer(mDevice, mQueueFamilyIndex, mD_input, mBatch_size, mInput_size);
        input_buffer = mInput;

    } else {
        input_buffer = mLayers[mLayers.size()-1]->get_output();
        input_dim = mLayers[mLayers.size()-1]->get_output_dim();
    }

    Convolution* d = new Convolution(mDevice, mQueueFamilyIndex, mPhysicalDevice, this->mBatch_size, input_dim, layer_dim, input_buffer);

    mLayers.push_back(d);
}

void OLP::forward_initialize(){
    std::vector<VkBuffer*> buffers{&mInput};
    allocateAndBindBuffers(mDevice, mPhysicalDevice, buffers, mDeviceMemory, mOffsets);

    for(Layer* layer : mLayers){
        layer->forward_initialize(mQueue);
    }
}

void OLP::forward(const std::vector<std::vector<float> > &batch) {
    char *data = nullptr;
    if(vkMapMemory(mDevice, mDeviceMemory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&data)) != VK_SUCCESS){
        throw std::runtime_error("failed to map mDevice memory");
    }
    float* batch_data = reinterpret_cast<float*>(data + mOffsets[0]);

    if(batch.size() != this->mBatch_size || batch[0].size() != this->mInput_size){
        throw std::invalid_argument("batch size or mInput dimension is wrong");
    }

    for(int i = 0;i<this->mBatch_size;i++){
        for(int j=0;j<this->mInput_size;j++){
            batch_data[i*this->mInput_size + j] = batch[i][j];
        }
    }

    vkUnmapMemory(mDevice, mDeviceMemory);

    for(Layer* layer : mLayers){
        layer->forward(mQueue);
    }
}

OLP::OLP() {
    setup_vulkan(mInstance, mDebugMessenger, mPhysicalDevice, mQueueFamilyIndex, mDevice, mQueue);
}

std::vector<std::pair<Tensor, Tensor>> OLP::get_trainable_parameters() {
    std::vector<std::pair<Tensor, Tensor>> params;

    for(Layer* layer : mLayers){
        std::vector<std::pair<Tensor, Tensor>> layer_params = layer->get_trainable_parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }

    return params;
}

OLP::~OLP() {

    for(Layer* layer : mLayers){
        delete layer;
    }

    vkFreeMemory(mDevice, mDeviceMemory, nullptr);

    vkDestroyBuffer(mDevice, mInput, nullptr);

    vkDestroyDevice(mDevice, nullptr);

    DestroyDebugUtilsMessengerEXT(mInstance, mDebugMessenger, nullptr);

    vkDestroyInstance(mInstance, nullptr);
}

float OLP::evaluate(std::vector<std::vector<float>> &X) {
    if(X.empty()){
        throw std::invalid_argument("X should have > 0");
    }
    forward(X);
}



