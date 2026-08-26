/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hermes/Support/SerialExecutor.h"
#include "llvh/ADT/ScopeExit.h"

#include "gtest/gtest.h"

#include <thread>

namespace {

/// Wait until \p predicate holds, giving up after a deadline so that a broken
/// executor fails the test instead of hanging it.
template <typename Predicate>
bool waitFor(Predicate predicate) {
  const auto giveUpAt =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!predicate()) {
    if (std::chrono::steady_clock::now() > giveUpAt)
      return false;
    // Yield rather than spin: on a single-core host a tight loop here can
    // starve the worker thread we are waiting on.
    std::this_thread::yield();
  }
  return true;
}

TEST(SerialExecutorTest, TestBasic) {
  hermes::SerialExecutor executor;
  std::atomic<int> counter{0};

  for (int i = 0; i < 100; ++i)
    executor.add([&counter] { ++counter; });

  while (counter < 100) {
  }
}

TEST(SerialExecutorTest, DestructorDrainsTasks) {
  int counter{0};
  {
    hermes::SerialExecutor executor;

    for (int i = 0; i < 100; ++i)
      executor.add([&counter] { ++counter; });
  }
  ASSERT_EQ(counter, 100);
}

TEST(SerialExecutorTest, TestTimeout) {
  // Set up an executor with a short timeout.
  constexpr std::chrono::milliseconds timeout{10};
  hermes::SerialExecutor executor{0, timeout};
  std::atomic<int> counter{0};

  for (int i = 0; i < 5; ++i) {
    auto t0 = std::chrono::steady_clock::now();

    // Add a task that sets up a thread-local destructor that will increment the
    // counter. This allows us to observe when the thread is destroyed by the
    // executor.
    executor.add([&counter] {
      thread_local auto se = llvh::make_scope_exit([&counter] { ++counter; });
    });

    // Wait for the counter to be incremented when the thread is joined.
    while (counter == i) {
    }

    auto t1 = std::chrono::steady_clock::now();
    ASSERT_GE(t1 - t0, timeout);
  }
}

TEST(SerialExecutorTest, ThreadRunnerWrapsEachWorker) {
  constexpr std::chrono::milliseconds timeout{10};
  std::atomic<int> starts{0};
  std::atomic<int> exits{0};
  std::atomic<int> tasks{0};
  hermes::SerialExecutor executor{0, timeout, [&](std::function<void()> run) {
                                    ++starts;
                                    run();
                                    ++exits;
                                  }};

  EXPECT_EQ(starts, 0);
  for (int i = 1; i <= 5; ++i) {
    executor.add([&tasks] { ++tasks; });

    ASSERT_TRUE(waitFor([&tasks, i] { return tasks >= i; }))
        << "task " << i << " never ran";
    // The worker has to time out and exit before the next add() can create a
    // new one, which is what makes starts/exits countable below.
    ASSERT_TRUE(waitFor([&exits, i] { return exits >= i; }))
        << "worker " << i << " never returned from its runner";
  }

  EXPECT_EQ(starts, 5);
  EXPECT_EQ(exits, 5);
}

} // end anonymous namespace
