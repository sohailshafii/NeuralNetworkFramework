#include <iostream>
#include <string>
#include <chrono>
#include "ImagesLoader.h"

typedef struct TrainingArguments {
    std::string trainingPath = "";
    std::string testPath = "";
    std::string positiveLabel = "";
    unsigned int numIterations = 1000;
    float learningRate = 0.001f;
    bool runParallel = false;
} TrainingArguments;

void PrintUsage();
char* TryGetArgument(int argIndex, int argc, char **argv);
unsigned int parseArguments(int argc, char** argv, TrainingArguments& trainingArgs);

int main(int argc, char** argv) {
    TrainingArguments trainingArgs;
    unsigned int parseReturnVal = parseArguments(argc, argv, trainingArgs);
    if (parseReturnVal != 0) {
        PrintUsage();
        return parseReturnVal;
    }
    
    if (trainingArgs.testPath == "") {
        std::cerr << "No test path given!\n";
        PrintUsage();
        return 1;
    }
    
    std::cout << "Loading all images...\n";
    auto loadStart = std::chrono::steady_clock::now();

    ImageLoader testData(trainingArgs.positiveLabel, trainingArgs.testPath,
                         trainingArgs.runParallel, true);
    ImageLoader trainingData(trainingArgs.positiveLabel, trainingArgs.trainingPath,
                             trainingArgs.runParallel, true);
    
    std::cout << "Training metadata: \n";
    trainingData.PrintMetadata();
    std::cout << "Testing metadata: \n";
    testData.PrintMetadata();
    
    auto loadEnd = std::chrono::steady_clock::now();
    auto loadMs = std::chrono::duration_cast<std::chrono::milliseconds>(loadEnd - loadStart).count();
    std::cout << "Loaded images in " << loadMs << " ms ("
              << (trainingArgs.runParallel ? "parallel" : "serial") << ").\n";
    
    std::cout << "Normalizing...\n";
    loadStart = std::chrono::steady_clock::now();
    testData.NormalizeData();
    trainingData.NormalizeData();
    loadEnd = std::chrono::steady_clock::now();
    loadMs = std::chrono::duration_cast<std::chrono::milliseconds>(loadEnd - loadStart).count();
    std::cout << "Done normalizing, it took: " << loadMs << "ms.\n";

    return 0;
}

void PrintUsage() {
    std::cout << "Arguments expected: -train <training_path> -test <test_path> -posLabel <positive_label> -parallel <1_or_0>\n";
}

char* TryGetArgument(int argIndex, int argc, char **argv) {
    if (argIndex == argc) {
        return nullptr;
    }
    return argv[argIndex];
}

unsigned int parseArguments(int argc, char** argv, TrainingArguments& trainingArgs) {
    for (int i = 1; i < argc; i++) {
        auto followingArgument = TryGetArgument(i + 1, argc, argv);
        if (!strcmp(argv[i], "-train")) {
            if (followingArgument == nullptr) {
                std::cerr << "No training path argument given.\n";
                return 1;
            }
            trainingArgs.trainingPath = followingArgument;
        }
        else if (!strcmp(argv[i], "-test")) {
            if (followingArgument == nullptr) {
                std::cerr << "No test path given.\n";
                return 1;
            }
            trainingArgs.testPath = followingArgument;
        }
        else if (!strcmp(argv[i], "-posLabel")) {
            if (followingArgument == nullptr) {
                std::cerr << "No positive label value given.\n";
                return 1;
            }
            trainingArgs.positiveLabel = followingArgument;
        }
        else if (!strcmp(argv[i], "-numItr")) {
            if (followingArgument == nullptr) {
                std::cerr << "No iteration value given.\n";
                return 1;
            }
            trainingArgs.positiveLabel = static_cast<unsigned int>(std::stoul(followingArgument));
        }
        else if (!strcmp(argv[i], "-learningRate")) {
            if (followingArgument == nullptr) {
                std::cerr << "No learning rate value given.\n";
                return 1;
            }
            try {
                trainingArgs.learningRate = (float)std::stod(followingArgument);
            }
            catch (const std::invalid_argument& e) {
                std::cerr << "Error: \"" << followingArgument << "\" contains no valid digits.\n";
                return 1;
            }
            catch (const std::out_of_range& e) {
                std::cerr << "Error: \"" << followingArgument << "\" is too large or small for a double.\n";
                return 1;
            }
        }
        else if (!strcmp(argv[i], "-parallel")) {
            if (followingArgument == nullptr) {
                std::cerr << "No parallel arg given.\n";
                return 1;
            }
            auto runParallelInt = static_cast<unsigned int>(std::stoul(followingArgument));
            trainingArgs.runParallel = (runParallelInt == 1) ? true : false;
        }
    }
    
    return 0;
}
