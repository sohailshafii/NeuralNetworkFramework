#pragma once

#include <cstdlib>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cstdint>

class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads);
    ~ThreadPool();
    
    // Calls work(begin, end) on each worker with a contiguous slice of [0, count).
    void ParallelFor(size_t count, const std::function<void(size_t begin, size_t end)>& work);

private:
    void WorkerLoop(size_t workerIndex);
    
    std::vector<std::thread> workers;
    
    std::mutex mutex;
    std::condition_variable jobAvailable;
    std::condition_variable jobFinished;
    
    // Current job. Workers read these under the mutex, then run outside it.
    const std::function<void(size_t begin, size_t end)>* currentWork = nullptr;
    size_t currentCount = 0;
    // incremented per ParallelFor so workers can tell a new job from an old one
    uint64_t jobGeneration = 0;
    // how many workers have not finished the current job.
    size_t workersRemaining = 0;
    
    bool stopping = false;
};
