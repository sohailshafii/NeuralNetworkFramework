#pragma once
#include <vector>

class Model {
public:
    Model(unsigned int numInputFeatures, const std::vector<unsigned int>& numUnits);
    
private:
    typedef struct LayerInfo {
        // flattened weights matrix and biases vector
        std::vector<float> weights;
        std::vector<float> biases;
    } LayerInfo;
    std::vector<unsigned int> numUnitsPerLayer;
    std::vector<LayerInfo> layers;
};
