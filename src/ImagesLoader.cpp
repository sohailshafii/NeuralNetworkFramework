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
#include <stdexcept>

#define TARGET_IMAGE_SIZE 64 // down from 256
#define EXPECTED_CHANNELS 3

ImageLoader::ImageLoader(const std::string& positiveLabel,
                         const std::string& parentFolder,
                         bool runInParallel,
                         bool outputTestImages) {
    std::vector<std::string> filePaths = GetFilesInPath(parentFolder);
    std::cout << "Found " << filePaths.size() << " images in path " << parentFolder << ".\n";
    normalized = false;
    if (runInParallel) {
        LoadDataParallel(filePaths, positiveLabel, parentFolder, outputTestImages);
    }
    else {
        LoadDataSerial(filePaths, positiveLabel, parentFolder, outputTestImages);
    }
}

void ImageLoader::PrintMetadata() {
    std::cout << "Number of samples: " << data.size()
        << ", width and height of each image: " << TARGET_IMAGE_SIZE << " x " << TARGET_IMAGE_SIZE
        << ", num channels: " << EXPECTED_CHANNELS << ", number of features: " << GetNumberOfFeatures() << "\n";
}

void ImageLoader::NormalizeData(bool runInParallel) {
    if (normalized) {
        std::cerr << "Data ormalized already.\n";
        return;
    }
    if (runInParallel) {
        NormalizeDataParallel();
    }
    else {
        NormalizeDataSerial();
    }
    normalized = true;
}

void ImageLoader::LoadDataSerial(const std::vector<std::string>& filePaths,
                                 const std::string& positiveLabel,
                                 const std::string& parentFolder,
                                 bool outputTestImages) {
    unsigned int numFeatures = GetNumberOfFeatures();
    // outputted to file in case people want to debug what is stored in memory.
    std::vector<unsigned char> testImage(numFeatures, 0);
    for (auto path : filePaths) {
        int width, height, channels;
        // positive means Y = 1, otherwise it is 0.
        size_t pos = path.find_last_of("/\\");
        std::string filename = (pos == std::string::npos) ? path : path.substr(pos + 1);
        float yValue = filename.find(positiveLabel) != std::string::npos ? 1.0f : 0.0f;
        unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 0);
        if (img == nullptr) {
            throw std::runtime_error(std::format("Could not load image at path {}.", path));
        }
        if (channels != EXPECTED_CHANNELS) {
            stbi_image_free(img);
            throw std::runtime_error(std::format("Channels {} vs expected {}.", channels, EXPECTED_CHANNELS));
        }
        assert(channels == EXPECTED_CHANNELS);
        // force a resize
        data.emplace_back(numFeatures);
        ResizeNearest(img, width, height, channels, data.back().data(), TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE);
        
        if (outputTestImages) {
            std::filesystem::path p(path);
            // always ".jpg": stbi_write_jpg writes JPEG regardless of the source format.
            std::string newName = p.stem().string() + "-small.jpg";
            std::filesystem::path newPath = p.parent_path() / newName;
            // Quality only applies to JPG (1–100)
            float* currImage = data.back().data();
            for (unsigned int i = 0; i < numFeatures; i++) {
                testImage[i] = (unsigned char)currImage[i];
            }
            int quality = 90;
            stbi_write_jpg(newPath.string().c_str(), TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE,
                           EXPECTED_CHANNELS, testImage.data(), quality);
        }
        
        fileNames.emplace_back(filename);
        yValues.push_back(yValue);
        stbi_image_free(img);
    }
}

void ImageLoader::LoadDataParallel(const std::vector<std::string>& filePaths,
                      const std::string& positiveLabel,
                      const std::string& parentFolder,
                      bool outputTestImages) {
    unsigned int numFeatures = GetNumberOfFeatures();
    size_t numImages = filePaths.size();

    // every worker writes to its own slot, so no lock is needed on the results
    // and the row order matches filePaths -- same as LoadDataSerial.
    data.assign(numImages, std::vector<float>(numFeatures, 0.0f));
    yValues.assign(numImages, 0.0f);
    fileNames.assign(numImages, std::string());

    std::vector<std::thread> workers;
    std::atomic<size_t> nextIndex{0};
    // an exception thrown in a worker cannot escape its thread -- that calls
    // std::terminate. Stash the first one and rethrow it after the join.
    std::mutex errorMutex;
    std::exception_ptr firstError;
    std::atomic<bool> aborted{false};

    auto worker = [&]() {
      // outputted to file in case people want to debug what is stored in memory.
      // one scratch buffer per thread; the serial path reuses a single one.
      std::vector<unsigned char> testImage;
      if (outputTestImages) {
          testImage.assign(numFeatures, 0);
      }
      try {
        for (size_t i = nextIndex.fetch_add(1); i < numImages && !aborted;
             i = nextIndex.fetch_add(1)) {
            const std::string& path = filePaths[i];

            int width, height, channels;
            size_t pos = path.find_last_of("/\\");
            std::string filename = (pos == std::string::npos) ? path : path.substr(pos + 1);
            // positive means Y = 1, otherwise it is 0.
            float yValue = filename.find(positiveLabel) != std::string::npos ? 1.0f : 0.0f;

            // NOTE: libjpeg-turbo is faster, consider for future.
            unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 0);

            if (img == nullptr) {
                throw std::runtime_error(std::format("Could not load image at path {}.", path));
            }
            if (channels != EXPECTED_CHANNELS) {
                stbi_image_free(img);
                throw std::runtime_error(std::format("Channels {} vs expected {}.", channels, EXPECTED_CHANNELS));
            }

            // force a resize, straight into this image's row
            ResizeNearest(img, width, height, channels, data[i].data(), TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE);
            stbi_image_free(img);

            if (outputTestImages) {
                std::filesystem::path p(path);
                // always ".jpg": stbi_write_jpg writes JPEG regardless of the source format.
                std::string newName = p.stem().string() + "-small.jpg";
                std::filesystem::path newPath = p.parent_path() / newName;
                // Quality only applies to JPG (1-100)
                float* currImage = data[i].data();
                for (unsigned int px = 0; px < numFeatures; px++) {
                    testImage[px] = (unsigned char)currImage[px];
                }
                int quality = 90;
                stbi_write_jpg(newPath.string().c_str(), TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE,
                               EXPECTED_CHANNELS, testImage.data(), quality);
            }

            fileNames[i] = std::move(filename);
            yValues[i] = yValue;
        }
      }
      catch (...) {
          // first failure wins; the rest of the pool stops at the loop check.
          std::lock_guard<std::mutex> lock(errorMutex);
          if (!firstError) {
              firstError = std::current_exception();
          }
          aborted = true;
      }
    };

    // hardware_concurrency() is allowed to return 0, and there is no point
    // spawning more threads than there are images.
    unsigned int hwThreads = std::thread::hardware_concurrency();
    size_t maxThreads = hwThreads == 0 ? 1 : hwThreads;
    if (maxThreads > numImages) {
        maxThreads = numImages;
    }

    for (size_t i = 0; i < maxThreads; i++) {
        workers.emplace_back(worker);
    }

    for (auto& thread : workers) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    if (firstError) {
        std::rethrow_exception(firstError);
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

void ImageLoader::NormalizeDataSerial() {
    unsigned int numItems = GetNumberOfFeatures();
    float normalFactor = 1.0f/255.0f;
    for (size_t i = 0; i < data.size(); i++) {
        auto& currentImage = data[i];
        if (currentImage.size() != numItems) {
            throw std::runtime_error(std::format("Empty image # {}.", i));
        }
        for (unsigned int px = 0; px < numItems; px++) {
            float oldValue = currentImage[px];
            // consider subtracting around 0 to normalize around -0.5-0.5
            currentImage[px] = oldValue * normalFactor;
        }
    }
}

void ImageLoader::NormalizeDataParallel() {
    size_t numImages = data.size();
    unsigned int numItems = GetNumberOfFeatures();
    std::mutex errorMutex;
    std::atomic<size_t> nextIndex{0};
    std::atomic<bool> aborted{false};
    std::exception_ptr firstError;
    float normalFactor = 1.0f/255.0f;
    
    auto worker = [&]() {
        try {
            for (size_t i = nextIndex.fetch_add(1);
                 i < numImages && !aborted; i = nextIndex.fetch_add(1)) {
                auto& currentImage = data[i];
                if (currentImage.size() != numItems) {
                    throw std::runtime_error(std::format("Empty image # {}.", i));
                }
                for (unsigned int px = 0; px < numItems; px++) {
                    float oldValue = currentImage[px];
                    // consider subtracting around 0 to normalize around -0.5-0.5
                    currentImage[px] = oldValue * normalFactor;
                }
            }
        }
        catch (...) {
            // first failure wins; the rest of the pool stops at the loop check.
            std::lock_guard<std::mutex> lock(errorMutex);
            if (!firstError) {
                firstError = std::current_exception();
            }
            aborted = true;
        }
    };
    
    unsigned int numHwThreads = std::thread::hardware_concurrency();
    size_t maxThreads = numHwThreads == 0 ? 1 : numHwThreads;
    if (maxThreads > numImages) {
        maxThreads = numImages;
    }
    std::vector<std::thread> workers;
    for (size_t i = 0; i < maxThreads; i++) {
        workers.emplace_back(worker);
    }
    
    for (auto& thread : workers) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    if (firstError) {
        std::rethrow_exception(firstError);
    }
}
