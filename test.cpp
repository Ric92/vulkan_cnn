#include <iostream>
#include <string>
#include <vulkan_init.h>
#include <OLP.h>
#include <fstream>
#include <sstream>


int main(int argc, char** argv) {

    uint32_t x_dim = 200;

    int val_dataset_size;
    std::string val_data_path;
    std::string val_labels_path;

    uint32_t batch_size = 1;
    OLP OLP(x_dim, batch_size);

    OLP.forward_initialize();
    std::cout<< "Initialized OLP" << std::endl;

    std::vector<std::vector<float>> img;
    std::cout<<"accuracy after training: "<<OLP.evaluate(img)<<std::endl;

    return 0;
}
