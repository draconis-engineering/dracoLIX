#pragma once
// Thread pool — Phase 3: Multithreading / Thread pool
// Licensed under GPL-3.0-only
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>

namespace dracolix {

class ThreadPool {
public:
    explicit ThreadPool(size_t n = std::thread::hardware_concurrency()) {
        if (n == 0) n = 2;
        n_threads_ = n;
        for (size_t i = 0; i < n; ++i) {
            workers_.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lk(m_);
                        cv_.wait(lk, [this]{ return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }
    ~ThreadPool() {
        { std::lock_guard<std::mutex> lk(m_); stop_ = true; }
        cv_.notify_all();
        for (auto& t : workers_) if (t.joinable()) t.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    template <typename F>
    auto enqueue(F&& f) -> std::future<decltype(f())> {
        using R = decltype(f());
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        std::future<R> fut = task->get_future();
        {
            std::lock_guard<std::mutex> lk(m_);
            tasks_.emplace([task]{ (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

    // parallel_for: split [0, n) into blocks
    template <typename F>
    void parallel_for(size_t n, F&& fn, size_t grain = 0) {
        if (n == 0) return;
        if (n < 8192 || n_threads_ == 1) { for (size_t i = 0; i < n; ++i) fn(i); return; }
        if (grain == 0) grain = std::max<size_t>(1, n / (n_threads_ * 4));
        std::atomic<size_t> next{0};
        std::vector<std::future<void>> futs;
        futs.reserve(n_threads_);
        for (size_t t = 0; t < n_threads_; ++t) {
            futs.emplace_back(enqueue([&, grain]{
                for (;;) {
                    size_t start = next.fetch_add(grain);
                    if (start >= n) break;
                    size_t end = std::min(n, start + grain);
                    for (size_t i = start; i < end; ++i) fn(i);
                }
            }));
        }
        for (auto& f : futs) f.get();
    }

    size_t size() const noexcept { return n_threads_; }

    static ThreadPool& global() {
        static ThreadPool g;
        return g;
    }

private:
    size_t n_threads_ = 0;
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex m_;
    std::condition_variable cv_;
    bool stop_ = false;
};

} // namespace dracolix
