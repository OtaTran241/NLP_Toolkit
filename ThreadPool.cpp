#include "ThreadPool.h"

ThreadPool::ThreadPool(size_t threads) : stop(false) {
    /*
    Input:
        - numThreads: The number of worker threads to be created and added to the pool.

    Output:
        - N/A (Constructor does not return a value).

    Functionality:
        - Creates `numThreads` worker threads and stores them in the `workers` vector.
        - Each worker thread enters an infinite loop to:
            - Wait for tasks to be added to the task queue.
            - Fetch a task from the queue in a thread-safe manner.
            - Execute the fetched task.
            - Exit if the pool is signaled to stop and there are no more tasks.
        - Synchronization between threads is managed using a mutex (`queueMutex`) and
          a condition variable (`condition`).
        - The pool remains operational until explicitly stopped or destroyed.
    */

    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this]() {
            while (true) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(this->queueMutex);

                    this->condition.wait(lock, [this]() {
                        return this->stop || !this->tasks.empty();
                        });

                    if (this->stop && this->tasks.empty())
                        return;

                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }

                task();
            }
            });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();
    for (auto& worker : workers) worker.join();
}
