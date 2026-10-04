#include <gtest/gtest.h>

#include <vector>

#include "Model.h"
#include "ThreadPool.h"

namespace {

TEST(ModelTest, ParallelInitMatchesSerial) {
    ThreadPool pool(4);
    // layer 0 is large enough to span several chunks; the rest are small.
    const unsigned int numInputFeatures = 12288;
    const std::vector<unsigned int> numUnitsPerLayer = {20, 7, 5, 1};
    const std::vector<float> noData;

    Model serial(numInputFeatures, numUnitsPerLayer);
    Model parallel(numInputFeatures, numUnitsPerLayer);
    serial.Train(noData, &pool, false, 42);
    parallel.Train(noData, &pool, true, 42);

    const auto& serialLayers = serial.GetLayers();
    const auto& parallelLayers = parallel.GetLayers();
    ASSERT_EQ(serialLayers.size(), numUnitsPerLayer.size());
    ASSERT_EQ(serialLayers.size(), parallelLayers.size());

    for (size_t i = 0; i < serialLayers.size(); i++) {
        EXPECT_EQ(serialLayers[i].weights, parallelLayers[i].weights) << "layer " << i;
        EXPECT_EQ(serialLayers[i].biases, parallelLayers[i].biases) << "layer " << i;
    }
}

} // namespace
