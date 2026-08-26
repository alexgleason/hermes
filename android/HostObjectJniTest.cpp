/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fbjni/fbjni.h>
#include <gtest/gtest.h>
#include <hermes/hermes.h>
#include <jsi/instrumentation.h>
#include <jsi/jsi.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <utility>

namespace facebook::hermes::test {
namespace {

namespace jni = facebook::jni;

/// Verifies that a HostObject owning a JNI global reference can be finalized.
/// On Android, hermes.cpp wraps the finalizer worker in a jni::ThreadScope; if
/// that ever stops happening, releasing the reference below throws from an
/// unattached thread and aborts this test.
TEST(HostObjectJniTest, HostObjectCanOwnJniGlobalRef) {
  // Everything below needs a JavaVM, so fail here rather than aborting deep
  // inside fbjni.
  ASSERT_TRUE(jni::Environment::isGlobalJvmAvailable());

  struct ReleaseState {
    std::mutex mutex;
    std::condition_variable condition;
    bool released{false};
  };

  class GlobalRefHostObject final : public jsi::HostObject {
   public:
    GlobalRefHostObject(
        jni::global_ref<jni::JString> globalRef,
        std::shared_ptr<ReleaseState> releaseState)
        : globalRef_(std::move(globalRef)),
          releaseState_(std::move(releaseState)) {}

    ~GlobalRefHostObject() override {
      globalRef_.reset();
      {
        std::lock_guard<std::mutex> lock(releaseState_->mutex);
        releaseState_->released = true;
      }
      releaseState_->condition.notify_one();
    }

   private:
    jni::global_ref<jni::JString> globalRef_;
    std::shared_ptr<ReleaseState> releaseState_;
  };

  auto runtime = facebook::hermes::makeHermesRuntime();
  auto releaseState = std::make_shared<ReleaseState>();
  {
    auto javaObject = jni::make_jstring("host object");
    (void)jsi::Object::createFromHostObject(
        *runtime,
        std::make_shared<GlobalRefHostObject>(
            jni::make_global(javaObject), releaseState));
  }

  runtime->instrumentation().collectGarbage("test");

  std::unique_lock<std::mutex> lock(releaseState->mutex);
  EXPECT_TRUE(releaseState->condition.wait_for(
      lock, std::chrono::seconds(5), [&releaseState] {
        return releaseState->released;
      }));
}

} // namespace
} // namespace facebook::hermes::test
