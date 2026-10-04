#pragma once
#include <vector>
#include <memory>

class ThreadPool;

class Model {
public:
    static constexpr unsigned int defaultSeed = 42;
    Model(unsigned int numInputFeatures, const std::vector<unsigned int>& numUnits);
    
    void Train(const std::vector<float>& data,
               ThreadPool* threadPool,
               bool parallel = false,
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
    
    void InitializeWeightsAndBiases(ThreadPool* threadPool, unsigned int seed, bool parallel);
};
