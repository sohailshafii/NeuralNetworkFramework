#include <gtest/gtest.h>

#include <unistd.h>
#include <filesystem>
#include <string>
#include <vector>

#include "ImagesLoader.h"
#include "ThreadPool.h"
// implementation lives in ImagesLoader.cpp, so only the declarations here.
#include "stb_image_write.h"

namespace {

// writes a handful of small RGB PNGs of varying sizes into a fresh temp folder,
// then removes the folder when the test ends.
class ImageLoaderTest : public ::testing::Test {
protected:
    std::filesystem::path folder;
    const std::string positiveLabel = "cat";

    void SetUp() override {
        folder = std::filesystem::temp_directory_path() /
                 ("nnf_loader_test_" + std::to_string(::getpid()));
        std::filesystem::create_directories(folder);

        struct Spec { const char* name; int width; int height; };
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
    }

    void TearDown() override {
        std::filesystem::remove_all(folder);
    }
};

TEST_F(ImageLoaderTest, ParallelLoadMatchesSerial) {
    ThreadPool pool(4);
    ImageLoader serial(positiveLabel, folder.string(), pool, false);
    ImageLoader parallel(positiveLabel, folder.string(), pool, true);

    ASSERT_EQ(serial.GetFileNames().size(), 6u);
    EXPECT_EQ(serial.GetFileNames(), parallel.GetFileNames());
    EXPECT_EQ(serial.GetYValues(), parallel.GetYValues());
    EXPECT_EQ(serial.GetData(), parallel.GetData());
}

TEST_F(ImageLoaderTest, ParallelNormalizeMatchesSerial) {
    ThreadPool pool(4);
    ImageLoader serial(positiveLabel, folder.string(), pool, false);
    ImageLoader parallel(positiveLabel, folder.string(), pool, false);
    ASSERT_EQ(serial.GetData(), parallel.GetData());

    serial.NormalizeData(pool, false);
    parallel.NormalizeData(pool, true);

    EXPECT_EQ(serial.GetData(), parallel.GetData());
    // sanity: normalization actually changed the values into [0, 1]
    for (float v : serial.GetData()) {
        ASSERT_GE(v, 0.0f);
        ASSERT_LE(v, 1.0f);
    }
}

} // namespace
