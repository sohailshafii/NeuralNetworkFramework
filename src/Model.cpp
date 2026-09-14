#include "Model.h"


Model::Model(unsigned int numInputFeatures, const std::vector<unsigned int>& numUnitsPerLayer) {
    this->numUnitsPerLayer = numUnitsPerLayer;
    
    size_t numLayers = numUnitsPerLayer.size();
    layers.resize(numLayers);
    layers[0].weights.resize(numUnitsPerLayer[0] * numInputFeatures);
    layers[0].biases.resize(numUnitsPerLayer[0]);
    // L0 x num_features
    for (size_t i = 1; i < numLayers; i++) {
        // L(i) x L(i + 1)
        layers[i].weights.resize(numUnitsPerLayer[i] * numUnitsPerLayer[i - 1]);
        layers[i].biases.resize(numUnitsPerLayer[i]);
    }
}
