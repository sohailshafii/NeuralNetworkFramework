#include "ImagesLoader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <filesystem>
#include <iostream>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <cassert>
#include <format>

#define TARGET_IMAGE_SIZE 64 // down from 256
#define EXPECTED_CHANNELS 3

ImageLoader::ImageLoader(const std::string& positiveLabel,
                         const std::string& parentFolder,
                         bool runInParallel,
                         bool outputTestImages) {
    std::vector<std::string> filePaths = GetFilesInPath(parentFolder);
    std::cout << "Found " << filePaths.size() << " images in path " << parentFolder << ".\n";
    
    if (runInParallel) {
        LoadDataParallel(filePaths, positiveLabel, parentFolder, outputTestImages);
    }
    else {
        LoadDataSerial(filePaths, positiveLabel, parentFolder, outputTestImages);
    }
}

void ImageLoader::LoadDataSerial(const std::vector<std::string>& filePaths,
                                 const std::string& positiveLabel,
                                 const std::string& parentFolder,
                                 bool outputTestImages) {
    unsigned int numFeatures = GetNumberOfFeatures();
    // outputted to file in case people want to debug what is stored in memory.
    std::vector<unsigned char> testImage(numFeatures, 0);
    std::vector<float> output(numFeatures, 0.0f);
    for (auto path : filePaths) {
        int width, height, channels;
        // positive means Y = 1, otherwise it is 0.
        size_t pos = path.find_last_of("/\\");
        std::string filename = (pos == std::string::npos) ? path : path.substr(pos + 1);
        fileNames.emplace_back(filename);
        float yValue = filename.find(positiveLabel) != std::string::npos ? 1.0f : 0.0f;
        unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 0);
        assert(img != nullptr && (std::format("Could not load image at path {}.", path).c_str()));
        assert(channels == EXPECTED_CHANNELS);
        // force a resize
        ResizeNearest(img, width, height, channels, output.data(), TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE);
        
        if (outputTestImages) {
            std::filesystem::path p(path);
            // always ".jpg": stbi_write_jpg writes JPEG regardless of the source format.
            std::string newName = p.stem().string() + "-small.jpg";
            std::filesystem::path newPath = p.parent_path() / newName;
            // Quality only applies to JPG (1–100)
            for (unsigned int i = 0; i < numFeatures; i++) {
                testImage[i] = (unsigned char)output[i];
            }
            int quality = 90;
            stbi_write_jpg(newPath.string().c_str(), TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE,
                           EXPECTED_CHANNELS, testImage.data(), quality);
        }

        data.push_back(output);
        yValues.push_back(yValue);
        stbi_image_free(img);
    }
}

void ImageLoader::LoadDataParallel(const std::vector<std::string>& filePaths,
                      const std::string& positiveLabel,
                      const std::string& parentFolder,
                      bool outputTestImages) {
    std::mutex queueMutex, dataMutex;
    std::condition_variable cv;
    std::queue<std::string> taskQueue;
    std::atomic<bool> done{false};
    std::vector<std::thread> workers;
    int maxThreads = std::thread::hardware_concurrency();
    
    for(const auto& path : filePaths) {
        taskQueue.push(path);
    }
}

std::vector<std::string> ImageLoader::GetFilesInPath(const std::string& path) {
    std::vector<std::string> allPaths;
    int numFiles;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        allPaths.push_back(entry.path());
    }
    return allPaths;
}

void ImageLoader::ResizeNearest(unsigned char* src, int oldWidth, int oldHeight, int channels,
                   float* dst, int newWidth, int newHeight) {
    float x_ratio = static_cast<float>(oldWidth) / newWidth;
    float y_ratio = static_cast<float>(oldHeight) / newHeight;
    
    for (int y = 0; y < newHeight; ++y) {
        int src_y = static_cast<int>(y * y_ratio);
        if (src_y >= oldHeight) {
            src_y = oldHeight - 1; // clamp
        }

        for (int x = 0; x < newWidth; ++x) {
            int src_x = static_cast<int>(x * x_ratio);
            if (src_x >= oldWidth) {
                src_x = oldWidth - 1; // clamp
            }

            for (int c = 0; c < channels; ++c) {
                dst[(y * newWidth + x) * channels + c] =
                    src[(src_y * oldWidth + src_x) * channels + c];
            }
        }
    }
}
