#pragma once
#include <vector>
#include <memory>
#include <ThreadPool.h>

class Model {
public:
    static constexpr unsigned int defaultSeed = 42;
    Model(unsigned int numInputFeatures, const std::vector<unsigned int>& numUnits);
    
    void Train(const std::vector<float>& data, bool parallel = false,
               unsigned int seed = defaultSeed);
    
private:
    typedef struct LayerInfo {
        // flattened weights matrix and biases vector
        std::vector<float> weights;
        std::vector<float> biases;
    } LayerInfo;
    unsigned int numInputFeatures;
    std::vector<unsigned int> numUnitsPerLayer;
    std::vector<LayerInfo> layers;
    std::shared_ptr<ThreadPool> pool;
    
    void InitializeWeightsAndBiases(unsigned int seed, bool parallel);
};
