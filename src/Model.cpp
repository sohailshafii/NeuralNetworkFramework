#include "Model.h"
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>

Model::Model(unsigned int numInputFeatures, const std::vector<unsigned int>& numUnitsPerLayer) {
    if (numUnitsPerLayer.empty()) {
        throw std::invalid_argument("Model needs at least one layer.");
    }
    this->numInputFeatures = numInputFeatures;
    this->numUnitsPerLayer = numUnitsPerLayer;
    
    size_t numLayers = numUnitsPerLayer.size();
    layers.resize(numLayers);
    layers[0].weights.resize(numUnitsPerLayer[0] * numInputFeatures);
    layers[0].biases.resize(numUnitsPerLayer[0]);
    // L0 x num_features
    for (size_t i = 1; i < numLayers; i++) {
        // L(i) x L(i - 1)
        layers[i].weights.resize(numUnitsPerLayer[i] * numUnitsPerLayer[i - 1]);
        layers[i].biases.resize(numUnitsPerLayer[i]);
    }
}

void Model::Train(const std::vector<float>& data, bool parallel, unsigned int seed) {
    InitializeWeightsAndBiases(seed);
}

void Model::InitializeWeightsAndBiases(unsigned int seed) {
    size_t numLayers = numUnitsPerLayer.size();
    // seeded once, outside the loop, so the layers draw from one continuous
    // stream. Re-seeding per layer would make same-sized layers identical
    std::mt19937 generator(seed);
    
    for (size_t i = 0; i < numLayers; i++) {
        // He initialization: normal noise scaled by sqrt(2 / fanIn), which keeps
        // the activation variance roughly steady as it passes through the layers.
        size_t fanIn = (i == 0) ? numInputFeatures : numUnitsPerLayer[i - 1];
        float standardDeviation = std::sqrt(2.0f / static_cast<float>(fanIn));
        std::normal_distribution<float> distribution(0.0f, standardDeviation);
        
        std::vector<float>& weights = layers[i].weights;
        for (size_t w = 0; w < weights.size(); w++) {
            weights[w] = distribution(generator);
        }
        // biases start at zero. Unlike zero weights, this causes no symmetry
        // problem -- the random weights already break it.
        std::fill(layers[i].biases.begin(), layers[i].biases.end(), 0.0f);
    }
}
