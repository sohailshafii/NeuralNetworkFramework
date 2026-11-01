#include <gtest/gtest.h>

#include <vector>
#include <filesystem>
#include <memory>

#include "Model.h"
#include "ThreadPool.h"
#include "ImagesLoader.h"
// implementation lives in ImagesLoader.cpp, so only the declarations here.
#include "stb_image_write.h"

class ModelTestsFixture : public ::testing::Test {
protected:
    std::shared_ptr<ImagesLoader> trainingImages;

    void SetUp() override {
        std::filesystem::path folder = std::filesystem::temp_directory_path() /
                 ("nnf_loader_test_" + std::to_string(::getpid()));
        std::filesystem::create_directories(folder);

        struct Spec { const char* name; int width; int height; };
        const std::string positiveLabel = "cat";
        const Spec specs[] = {
            {"cat_0.png", 10, 8},
            {"dog_1.png", 64, 64},
            {"cat_2.png", 100, 70},
            {"bird_3.png", 3, 5},
            {"cat_4.png", 128, 96},
            {"dog_5.png", 33, 17},
        };
        int seed = 0;
        for (const Spec& spec : specs) {
            std::vector<unsigned char> pixels(spec.width * spec.height * EXPECTED_CHANNELS);
            for (size_t i = 0; i < pixels.size(); i++) {
                // deterministic but non-uniform content
                pixels[i] = static_cast<unsigned char>((i * 7 + seed * 13) % 256);
            }
            seed++;
            std::string path = (folder / spec.name).string();
            int ok = stbi_write_png(path.c_str(), spec.width, spec.height, EXPECTED_CHANNELS,
                                    pixels.data(), spec.width * EXPECTED_CHANNELS);
            ASSERT_NE(ok, 0) << "failed to write " << path;
        }
        
        ThreadPool pool(4);
        trainingImages = std::make_shared<ImagesLoader>(positiveLabel, folder.string(), pool, true);
        std::filesystem::remove_all(folder);
    }
    
    void CreateRandomMiniBatches(Model& model,
                                 std::vector<std::vector<float>>& batchesX,
                                 std::vector<std::vector<float>>& batchesY,
                                 ThreadPool& threadPool,
                                 unsigned int miniBatchSize,
                                 bool parallel,
                                 unsigned int seed) {
        std::vector<Model::MiniBatch> miniBatches;
        model.CreateRandomMiniBatches(*trainingImages, miniBatches, threadPool, 64, false);
        batchesX.resize(miniBatches.size());
        batchesY.resize(miniBatches.size());
        for(size_t i = 0; i < miniBatches.size(); i++) {
            batchesX[i].resize(miniBatches[i].miniBatchX.size() * sizeof(float));
            batchesY[i].resize(miniBatches[i].miniBatchY.size() * sizeof(float));
            memcpy(batchesX[i].data(), miniBatches[i].miniBatchX.data(),
                   miniBatches[i].miniBatchX.size() * sizeof(float));
            memcpy(batchesY[i].data(), miniBatches[i].miniBatchY.data(),
                   miniBatches[i].miniBatchY.size() * sizeof(float));
        }
    }

    void TearDown() override {
    }
};

TEST_F(ModelTestsFixture, ParallelBatchesMatchesSerial) {
    ThreadPool pool(4);
    // layer 0 is large enough to span several chunks; the rest are small.
    const unsigned int numInputFeatures = 12288;
    const std::vector<unsigned int> numUnitsPerLayer = {20, 7, 5, 1};
    const std::vector<float> noData;

    Model serial(numInputFeatures, numUnitsPerLayer);
    
    Model::ModelParams modelParams {
       .learningRate = 0.0007f,
       .miniBatchSize = 64,
       .beta = 0.9f,
       .beta1 = 0.9f,
       .beta2 = 0.99f,
       .epsilon = 1e-8,
       .numEpochs = 30,
           
       .printCost = true,
       .decay = nullptr,
       .decayRate = 1.0f,
       .timeInterval = 100,
           
       .optimizer = Model::Optimizer::GradientDescent
    };
    auto trainStart = std::chrono::steady_clock::now();
    /*model.Train(trainingData, pool,
                modelParams,
                trainingArgs.runParallel);*/
   /* const ImagesLoader& imagesLoader,
                      ThreadPool& threadPool,
                      const ModelParams& modelParams,
                      bool parallel, unsigned int seed*/
    
    /*CreateRandomMiniBatches(const ImagesLoader& imagesLoader,
     std::vector<MiniBatch>& miniBatches,
     ThreadPool& threadPool,
     unsigned int miniBatchSize,
     bool parallel,
     unsigned int seed)*/
    std::vector<std::vector<float>> miniBatchesSamplesSerial, miniBatchesOutSerial;
    std::vector<std::vector<float>> miniBatchesSamplesParallel, minBatchesOutParallel;
    CreateRandomMiniBatches(serial, miniBatchesSamplesSerial,
                            miniBatchesOutSerial, pool, 64, false, 42);
    CreateRandomMiniBatches(serial, miniBatchesSamplesParallel,
                            minBatchesOutParallel, pool, 64, true, 42);
    
    size_t numTotalBatchesSerial = miniBatchesSamplesSerial.size();
    size_t numTotalBatchesParallel = miniBatchesSamplesParallel.size();
    
    ASSERT_EQ(numTotalBatchesSerial, numTotalBatchesParallel);
    
    for (size_t i = 0; i < numTotalBatchesSerial; i++) {
        auto sizeFeatSerial = miniBatchesSamplesSerial[i].size();
        auto sizeOutSerial = miniBatchesOutSerial[i].size();
        auto sizeFeatParallel = miniBatchesSamplesParallel[i].size();
        auto sizeOutParallel = minBatchesOutParallel[i].size();
        
        ASSERT_EQ(sizeFeatSerial, sizeFeatParallel);
        ASSERT_EQ(sizeOutSerial, sizeOutParallel);
        
        EXPECT_EQ(miniBatchesSamplesSerial[i], miniBatchesSamplesParallel[i]);
        EXPECT_EQ(miniBatchesOutSerial[i], minBatchesOutParallel[i]);
    }
}

