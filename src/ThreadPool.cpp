
#include "ThreadPool.h"
#include <algorithm>
#include <stdexcept>

ThreadPool::ThreadPool(size_t numThreads) {
    if (numThreads == 0) {
        throw std::invalid_argument("ThreadPool needs at least one thread.");
    }
    workers.reserve(numThreads);
    for (size_t i = 0; i < numThreads; i++) {
        workers.emplace_back(&ThreadPool::WorkerLoop, this, i);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
    }
    jobAvailable.notify_all();
    for (std::thread& worker : workers) worker.join();
}

// Calls work(begin, end) on each worker with a contiguous slice of [0, count).
void ThreadPool::ParallelFor(size_t count, const std::function<void(size_t begin, size_t end)>& work) {
    std::unique_lock<std::mutex> lock(mutex);
    
    currentWork = &work;
    currentCount = count;
    workersRemaining = workers.size();
    jobGeneration++;
    jobAvailable.notify_all();
    
    jobFinished.wait(lock, [&] {
        return workersRemaining == 0;
    });
    currentWork = nullptr;
}

void ThreadPool::WorkerLoop(size_t workerIndex) {
    uint64_t lastGeneration = 0;
    while(true) {
        const std::function<void(size_t, size_t)>* work;
        size_t count;
        {
            std::unique_lock<std::mutex> lock(mutex);
            jobAvailable.wait(lock, [&] { return stopping || jobGeneration != lastGeneration; });
            if (stopping) {
                return;
            }
            lastGeneration = jobGeneration;
            work = currentWork;
            count = currentCount;
        }
        
        // Compute this worker's slice of [0, count). Split as evenly as possible.
        size_t numWorkers = workers.size();
        size_t workerChunk = count / numWorkers;
        size_t extra = count % numWorkers;
        // 10 items across 4 workers. 10/4 = 2 is chunk, with 2 extra.
        // So we have 0, 2, 4, 6 with just the initial part. for the extra:
        // 1. min(0, 2) => 0 -> 0. so 0-2
        // 2. min(1, 2) => 1 -> 3. so 3-5
        // 3. min(2, 2) => 2 -> 6. so 6-8
        // 4. min(3, 2) => 2 -> 8. so 8-9
        // this indicates how many extra items landed before the current one.
        size_t begin = workerIndex * workerChunk + std::min(workerIndex, extra);
        size_t end = begin + workerChunk + (workerIndex < extra ? 1 : 0);
        // say you have 1 chunk, and 8 workers
        // so chunk is 1/8 = 0. extra is 1. For thread 0, begin is 0, and end is
        // end is 1.
        // next one is 1 to 1 or start=end, so no work to do.
        if (begin < end) (*work)(begin, end);
        {
            std::lock_guard<std::mutex> lock(mutex);
            // notify one wakes the wait above in ParallelFor to check if done
            if (--workersRemaining == 0) {
                jobFinished.notify_one();
            }
        }
    }
}
