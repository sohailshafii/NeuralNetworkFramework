#pragma once
#include <vector>

class ThreadPool;
class ImagesLoader;

class Model {
public:
    using DecayCallback = float(*)(float, unsigned int, float, unsigned int);
    
    enum class Optimizer {
        GradientDescent, Momentum, Adam
    };
    
    typedef struct ModelParams {
        float learningRate = 0.0007f;
        unsigned int miniBatchSize = 64;
        float beta = 0.9f;
        float beta1 = 0.9f;
        float beta2 = 0.99f;
        float epsilon = 1e-8;
        unsigned int numEpochs = 5000;
        
        bool printCost = true;
        DecayCallback decay = nullptr;
        float decayRate = 1.0f;
        unsigned int timeInterval = 100;
        
        Optimizer optimizer = Optimizer::GradientDescent;
    } ModelParams;
    
    static constexpr unsigned int defaultSeed = 42;
    Model(unsigned int numInputFeatures, const std::vector<unsigned int>& numUnits);
    
    void Train(const ImagesLoader& imagesLoader,
               ThreadPool& threadPool,
               const ModelParams& modelParams,
               bool parallel = false,
               unsigned int seed = defaultSeed);
    
    typedef struct LayerInfo {
        // flattened weights matrix and biases vector
        std::vector<float> weights;
        std::vector<float> biases;
    } LayerInfo;
    
    const std::vector<LayerInfo>& GetLayers() const { return layers; }
private:
    unsigned int numInputFeatures;
    std::vector<unsigned int> numUnitsPerLayer;
    std::vector<LayerInfo> layers;
    
    typedef struct MiniBatch {
        std::vector<float> miniBatchX;
        std::vector<float> miniBatchY;
    } MiniBatch;
    
    void CreateRandomMiniBatches(const ImagesLoader& imagesLoader,
                                 std::vector<MiniBatch>& miniBatches,
                                 ThreadPool& threadPool,
                                 unsigned int miniBatchSize,
                                 bool parallel,
                                 unsigned int seed = 0);
    
    void InitializeWeightsAndBiases(ThreadPool& threadPool, unsigned int seed, bool parallel);
    
    friend class ModelTestsFixture;
};
