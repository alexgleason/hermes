/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HERMES_SERIALEXECUTOR_SERIALEXECUTOR_H
#define HERMES_SERIALEXECUTOR_SERIALEXECUTOR_H

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#if !defined(_WINDOWS) && !defined(__EMSCRIPTEN__)
#include <pthread.h>
#else
#include <thread>
#endif

namespace hermes {

/// Simple executor that guarantees serial execution of tasks.
class SerialExecutor {
 private:
  /// The state of the background executor thread. Protected by mutex_.
  enum class ThreadState {
    /// No thread has been created yet.
    Uninitialized,
    /// The thread is ready to run tasks.
    Initialized,
    /// The thread has exited because it timed out.
    TimedOut,
    /// The thread is draining tasks and exiting during teardown.
    Terminating
  } threadState_{ThreadState::Uninitialized};

  /// The thread on which all work is done.
#if !defined(_WINDOWS) && !defined(__EMSCRIPTEN__)
  pthread_t tid_;
#else
  std::thread workerThread_;
#endif

  /// A list of functions to execute on the worker thread.
  std::deque<std::function<void()>> tasks_;

  /// Mutex guarding state shared with the worker thread.
  std::mutex mutex_;

  /// This is used to put the executor thread to sleep when there is nothing to
  /// do, and wake it up when that changes.
  std::condition_variable wakeUpSig_;

  /// The configured stack size for the worker thread.
  size_t stackSize_;

  /// The configured timeout after which the worker thread will exit if no
  /// additional work is enqueued.
  std::chrono::nanoseconds timeout_;

  /// This is executed on a new thread. It will run forever, executing tasks as
  /// they are posted. This stops running when shouldStop_ is set to true.
  void run();

  /// Main function of the new thread.
  static void *threadMain(void *p);

 public:
  /// Idle timeout used when the caller does not pick one. Just a reasonably
  /// large enough duration.
  static constexpr std::chrono::nanoseconds kDefaultTimeout =
      std::chrono::hours(24);

  /// Initialize a SerialExecutor with a worker thread that has a stack size of
  /// \p stackSize and will remain live for \p timeout without additional work.
  /// \p timeout can not be too large, otherwise `steady_clock::now() + timeout`
  /// may overflow.
  SerialExecutor(
      size_t stackSize = 0,
      std::chrono::nanoseconds timeout = kDefaultTimeout)
      : stackSize_(stackSize), timeout_(timeout) {
    // run() waits with wait_for, which is specified as
    // wait_until(now() + timeout_). Reject a timeout that cannot survive that
    // addition here.
    assert(timeout_.count() >= 0 && "SerialExecutor timeout is negative");
    assert(
        timeout_ <= std::chrono::steady_clock::time_point::max() -
                std::chrono::steady_clock::now() &&
        "SerialExecutor timeout overflows steady_clock::now() + timeout");
  }

  /// Make sure that the spawned thread has terminated. Will block if there is a
  /// long-running task currently being executed.
  ~SerialExecutor();

  /// Push a task to the back of the queue, lazily creating the worker thread if
  /// it does not exist.
  void add(std::function<void()> task);
};
} // namespace hermes

#endif // HERMES_SERIALEXECUTOR_SERIALEXECUTOR_H
