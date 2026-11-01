#include "Model.h"
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <cassert>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include "ThreadPool.h"
#include "ImagesLoader.h"

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
        // L(i) x L(i - 1). Multiplies by input layer and produces
        // output layer.
        layers[i].weights.resize(numUnitsPerLayer[i] * numUnitsPerLayer[i - 1]);
        layers[i].biases.resize(numUnitsPerLayer[i]);
    }    
}

void Model::Train(const ImagesLoader& imagesLoader,
                  ThreadPool& threadPool,
                  const ModelParams& modelParams,
                  bool parallel, unsigned int seed) {
    InitializeWeightsAndBiases(threadPool, seed, parallel);
    
    ModelParams currentParams = modelParams;
    std::vector<float> costs;
    // adam update counter
    unsigned int t = 0;
    // TODO: seed for minibatches
    // number of training examples -- get from images loader
    unsigned long m = imagesLoader.GetNumImages();
    
    if (modelParams.optimizer == Optimizer::GradientDescent) {
        // TODO implement if required
    }
    else if (modelParams.optimizer == Optimizer::Momentum) {
        // TODO implement momentum
    }
    else if (modelParams.optimizer == Optimizer::Adam) {
        // TODO: implement adam
    }
    
    std::vector<float> yValues;
    
    unsigned int currentSeed = seed;
    std::vector<MiniBatch> miniBatches;
    for (unsigned int currEpoch = 0; currEpoch < modelParams.numEpochs; currEpoch++) {
        // Define the random minibatches. We increment the seed to distinctly reshuffle
        // the data set after each epoch.
        currentSeed++;
        
        CreateRandomMiniBatches(imagesLoader, miniBatches, threadPool, modelParams.miniBatchSize,
                                parallel, currentSeed);
        
        // TODO: loop per mini batch
        
        // TODO cost avg
        
        // TODO decay
        
        // TODO print
    }
    
    // TODO: set params and final cost
}

void Model::CreateRandomMiniBatches(const ImagesLoader& imagesLoader,
                                    std::vector<MiniBatch>& miniBatches,
                                    ThreadPool& threadPool,
                                    unsigned int miniBatchSize,
                                    bool parallel,
                                    unsigned int seed) {
    miniBatches.clear();
    
    size_t m = imagesLoader.GetNumImages();
    std::vector<size_t> samplesPerm(m);
    // for from 0 to samples.end()-1
    std::iota(samplesPerm.begin(), samplesPerm.end(), 0u);
    // shuffle them using a seed
    std::mt19937 randGen(seed);
    std::shuffle(samplesPerm.begin(), samplesPerm.end(), randGen);
    
    // create re-usable chunks to store features, and also output chunks.
    unsigned int numFeatures = imagesLoader.GetNumberOfFeaturesPerImage();
    // a portion of the data will be complete, while remaining will need to set up manually
    size_t numCompleteMiniBatches = floor(m / miniBatchSize);
    const auto& originalYValues = imagesLoader.GetYValues();
    const auto& originalXValues = imagesLoader.GetData();
    
    miniBatches.resize(numCompleteMiniBatches);
    if (parallel) {
        auto batchWorker = [&originalXValues, &originalYValues, &miniBatches,
                            &samplesPerm, numFeatures, miniBatchSize]
                            (size_t batchBeginIndex, size_t batchEndIndex) {
            size_t samplesPerBatch = numFeatures * miniBatchSize;
            std::vector<float> featureBatch(samplesPerBatch);
            std::vector<float> outputBatch(miniBatchSize);
            
            size_t featSampleSizeBytes = numFeatures * sizeof(float);
            for (size_t batchIndex = batchBeginIndex; batchIndex < batchEndIndex; batchIndex++) {
                // loop through samples in batch
                size_t batchSampleBeginIndex = batchIndex * miniBatchSize;
                size_t batchSampleEndIndex = batchSampleBeginIndex + miniBatchSize;
                
                for (size_t batchSampleI = batchSampleBeginIndex; batchSampleI < batchSampleEndIndex;
                     batchSampleI++) {
                    // key: apply perm!
                    size_t permIndex = samplesPerm[batchSampleI];
                    size_t currFeatureOffset = permIndex * numFeatures;
                    
                    size_t offsetOutSampleBatch = (batchSampleI - batchSampleBeginIndex) * numFeatures;
                    size_t offsetOutBatch = (batchSampleI - batchSampleBeginIndex);
                    memcpy(featureBatch.data() + offsetOutSampleBatch,
                           originalXValues.data() + currFeatureOffset,
                           featSampleSizeBytes);
                    memcpy(outputBatch.data() + offsetOutBatch,
                           originalYValues.data() + permIndex,
                           sizeof(float));
                }
                
                miniBatches[batchIndex] = MiniBatch{featureBatch, outputBatch};
            }
        };
        threadPool.ParallelFor(numCompleteMiniBatches, batchWorker);
    }
    else {
        size_t samplesPerBatch = numFeatures * miniBatchSize;
        std::vector<float> featureBatch(samplesPerBatch);
        std::vector<float> outputBatch(miniBatchSize);
        
        size_t featSampleSizeBytes = numFeatures * sizeof(float);
        for (size_t batchIndex = 0; batchIndex < numCompleteMiniBatches; batchIndex++) {
            // loop through samples in batch
            size_t batchSampleBeginIndex = batchIndex * miniBatchSize;
            size_t batchSampleEndIndex = batchSampleBeginIndex + miniBatchSize;
            
            for (size_t batchSampleI = batchSampleBeginIndex; batchSampleI < batchSampleEndIndex;
                 batchSampleI++) {
                // key: apply perm!
                size_t permIndex = samplesPerm[batchSampleI];
                size_t currFeatureOffset = permIndex * numFeatures;
                
                size_t offsetOutSampleBatch = (batchSampleI - batchSampleBeginIndex) * numFeatures;
                size_t offsetOutBatch = (batchSampleI - batchSampleBeginIndex);
                memcpy(featureBatch.data() + offsetOutSampleBatch,
                       originalXValues.data() + currFeatureOffset,
                       featSampleSizeBytes);
                memcpy(outputBatch.data() + offsetOutBatch,
                       originalYValues.data() + permIndex,
                       sizeof(float));
            }
            
            miniBatches[batchIndex] = MiniBatch{featureBatch, outputBatch};
        }
    }
    
    // handling case where last minibatch is incomplete
    if (m % miniBatchSize != 0) {
        size_t lastFeatureOffset = numCompleteMiniBatches * numFeatures * miniBatchSize;
        size_t lastOutputOffset = numCompleteMiniBatches * miniBatchSize;
        size_t numRemainingSamples = m - lastOutputOffset;
        std::vector<float> lastFeatChunk(numFeatures * numRemainingSamples);
        std::vector<float> lastOutputChunk(numRemainingSamples);
        memcpy(lastFeatChunk.data(), originalXValues.data() + lastFeatureOffset,
               numFeatures * numRemainingSamples * sizeof(float));
        memcpy(lastOutputChunk.data(), originalYValues.data() + lastOutputOffset,
               numRemainingSamples * sizeof(float));
        miniBatches.emplace_back(MiniBatch{lastFeatChunk, lastOutputChunk});
    }
}

void Model::InitializeWeightsAndBiases(ThreadPool& threadPool, unsigned int seed, bool parallel) {
    size_t numLayers = numUnitsPerLayer.size();
    
    for (size_t i = 0; i < numLayers; i++) {
        // He initialization: normal noise scaled by sqrt(2 / fanIn), which keeps
        // the activation variance roughly steady as it passes through the layers.
        size_t fanIn = (i == 0) ? numInputFeatures : numUnitsPerLayer[i - 1];
        float standardDeviation = std::sqrt(2.0f / static_cast<float>(fanIn));
        
        std::vector<float>& weights = layers[i].weights;
        // Weights are drawn in fixed-size chunks, each from its own generator seeded by
        // (seed, layer, chunk). Results depend only on the seed and chunk size, never on
        // how many threads ran, so a run reproduces on any machine and the serial and
        // parallel paths produce identical weights.
        // Chunk size trades off two costs. Each chunk seeds its own mt19937, which
        // costs tens of microseconds, so chunks must be big enough that drawing the
        // numbers dominates the seeding. But they must be small enough that a large
        // layer yields more chunks than threads, or some workers sit idle.
        // Any multiple of 16 floats (one 64-byte cache line) keeps chunk seams off
        // shared cache lines; this size is far above that for the reasons above.
        constexpr size_t chunkSize = 65536;
        // let's say you have 100000 weights. chunk size is 65536.
        // that's about 1.5 chunks. You need to take the remainder into account.
        // So we add chunkSize-1 to round up. This is similar to
        //  ceil(a / b) written as (a + b - 1) / b. Why -1? Well if you had
        // 131,072 weights, which is exactly 2x the chunk size, adding chunkSize would
        // cause the num chunks to be 3 after the division. if we add chunkSize-1,
        // we would still be at 2.
        size_t numChunks = (weights.size() + chunkSize - 1) / chunkSize;
        
        // Fills chunks [chunkBegin, chunkEnd). Used by both paths below.
        auto fillChunks = [&](size_t chunkBegin, size_t chunkEnd) {
            for (size_t c = chunkBegin; c < chunkEnd; c++) {
                std::seed_seq seq{seed, static_cast<unsigned int>(i), static_cast<unsigned int>(c)};
                std::mt19937 generator(seq);
                std::normal_distribution<float> distribution(0.0f, standardDeviation);
                size_t begin = c * chunkSize;
                size_t end = std::min(begin + chunkSize, weights.size());
                for (size_t w = begin; w < end; w++) { weights[w] = distribution(generator); }
            }
        };
        
        if (parallel) {
            threadPool.ParallelFor(numChunks, fillChunks);
        }
        else {
            fillChunks(0, numChunks);
        }
        // biases start at zero. Unlike zero weights, this causes no symmetry
        // problem -- the random weights already break it.
        std::fill(layers[i].biases.begin(), layers[i].biases.end(), 0.0f);
    }
}
