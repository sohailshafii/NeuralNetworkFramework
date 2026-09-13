#pragma once

#include <vector>
#include <string>

#define TARGET_IMAGE_SIZE 64 // down from 256
#define EXPECTED_CHANNELS 3

class ImageLoader {
public:
    // positive label indicates if an image is a positive result
    // this is used for supervised learning. positive relates to
    // a probably of 1, otherwise it is a 0
    ImageLoader(const std::string& positiveLabel,
                const std::string& parentFolder,
                bool runInParallel = false,
                bool outputTestImages = false);
    
    unsigned int GetNumberOfFeatures()
    {
        return TARGET_IMAGE_SIZE*TARGET_IMAGE_SIZE*EXPECTED_CHANNELS;
    }
    
    void PrintMetadata();
    
    void NormalizeData(bool runInParallel = false);
private:
    std::vector<std::vector<float>> data;
    std::vector<float> yValues;
    std::vector<std::string> fileNames;
    bool normalized;
    
    void LoadDataSerial(const std::vector<std::string>& filePaths,
                        const std::string& positiveLabel,
                        const std::string& parentFolder,
                        bool outputTestImages);
    
    void LoadDataParallel(const std::vector<std::string>& filePaths,
                          const std::string& positiveLabel,
                          const std::string& parentFolder,
                          bool outputTestImages);
    
    std::vector<std::string> GetFilesInPath(const std::string& path);
    void ResizeNearest(unsigned char* src, int oldWidth, int oldHeight, int channels,
                  float* dst, int newWidth, int newHeight);
    
    void NormalizeDataSerial();
    void NormalizeDataParallel();
};
