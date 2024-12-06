#pragma once
#include <thread>
#include <queue>
#include <functional>
#include <vector>
#include <future>
#include <condition_variable>
#include <stdexcept>

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop = false;

public:
    explicit ThreadPool(size_t numThreads);
    ~ThreadPool();

    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        /*
        Input:
            - f: A callable object (e.g., a function or lambda) representing the task to be executed.
            - args: (Optional) Arguments to be passed to the callable object.

        Output:
            - A `std::future` object representing the result of the task.
              - The `std::future` can be used to retrieve the return value of the task after its execution.

        Functionality:
            - Wraps the provided callable (`f`) and arguments (`args`) into a `std::packaged_task`.
            - The `std::packaged_task` is added to the task queue in a thread-safe manner using a mutex (`queueMutex`).
            - Notifies one of the worker threads via the condition variable (`condition.notify_one`) that a new task is available.
            - Returns the `std::future` associated with the `std::packaged_task`, allowing the caller to access the task's result when ready.

        Exceptions:
            - Throws `std::runtime_error` if the thread pool has been stopped (`stop` is true), indicating tasks can no longer be enqueued.
        */

        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }
};
