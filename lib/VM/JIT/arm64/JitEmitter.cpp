/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hermes/VM/JIT/Config.h"
#if HERMESVM_JIT
#include "JitEmitter-internal.h"
#include "JitEmitter.h"
#include "JitImpl.h"

#include "JitHandlers.h"

#include "../RuntimeOffsets.h"
#include "hermes/BCGen/SerializedLiteralParser.h"
#include "hermes/Support/ErrorHandling.h"
#include "hermes/VM/ArrayStorage.h"
#include "hermes/VM/Callable.h"
#include "hermes/VM/IdentifierTable.h"
#include "hermes/VM/Interpreter.h"
#include "hermes/VM/JSObject-inline.h"
#include "hermes/VM/StaticHUtils.h"
#include "hermes/VMLayouts/StackFrameLayout.h"

#include <cstdio>
#include <limits>

#if defined(HERMESVM_COMPRESSED_POINTERS) && !defined(HERMESVM_CONTIGUOUS_HEAP)
#error JIT does not support non-contiguous heap with compressed pointers
#endif

namespace hermes::vm::arm64 {

llvh::raw_ostream &operator<<(
    llvh::raw_ostream &os,
    const hermes::vm::arm64::HWReg &hwReg) {
  if (hwReg.isValidGpX()) {
    os << "x" << (int)hwReg.indexInClass();
  } else if (hwReg.isValidVecD()) {
    os << "d" << (int)hwReg.indexInClass();
  } else {
    assert(!hwReg.isValid());
    os << "<invalid>";
  }
  return os;
}

Emitter::Emitter(
    Runtime &runtime,
    JITContext::Impl &jitImpl,
    unsigned dumpJitCode,
    bool emitAsserts,
    bool emitCounters,
    PerfJitDump *perfJitDump,
    CodeBlock *codeBlock,
    const std::function<void(std::string &&message)> &longjmpError)
    : runtime_(runtime),
      jitImpl_(jitImpl),
      dumpJitCode_(dumpJitCode),
      emitAsserts_(emitAsserts),
      emitCounters_(emitCounters),
      frameRegs_(codeBlock->getFrameSize()),
      codeBlock_(codeBlock) {
  errorHandler_ = std::unique_ptr<asmjit::ErrorHandler>(
      new OurErrorHandler(expectedError_, longjmpError));

  code.init(jitImpl.jr.environment(), jitImpl.jr.cpuFeatures());
  code.setErrorHandler(errorHandler_.get());

#ifndef ASMJIT_NO_LOGGING
  if ((dumpJitCode_ & DumpJitCode::Code) || perfJitDump) {
    logger_ = std::unique_ptr<asmjit::Logger>(
        new OurLogger(a, perfJitDump, dumpJitCode_));
    logger_->setIndentation(asmjit::FormatIndentationGroup::kCode, 4);
    logger_->addFlags(asmjit::FormatFlags::kHexImms);
    code.setLogger(logger_.get());
  }
#endif

  code.attach(&a);

  roDataLabel_ = a.newNamedLabel("RO_DATA");
  returnLabel_ = a.newNamedLabel("leave");

  // Save read/write property cache addresses.
  roOfsReadPropertyCachePtr_ = uint64Const(
      (uint64_t)codeBlock->readPropertyCache(), "readPropertyCache");
  roOfsWritePropertyCachePtr_ = uint64Const(
      (uint64_t)codeBlock->writePropertyCache(), "writePropertyCache");
  roOfsPrivateNameCachePtr_ =
      uint64Const((uint64_t)codeBlock->privateNameCache(), "privateNameCache");
}

void Emitter::enter(uint32_t numCount, uint32_t npCount) {
  unsigned nextVec = kVecSaved.first;
  unsigned nextGp = kGPSaved.first;

  // Number registers: allocate in vector hw regs first.
  for (unsigned frIndex = 0; frIndex < numCount; ++frIndex) {
    HWReg hwReg;
    if (nextVec <= kVecSaved.second) {
      hwReg = HWReg::vecD(nextVec);
      comment("    ; alloc: d%u <= r%u", nextVec, frIndex);
      ++nextVec;
    } else if (nextGp <= kGPSaved.second) {
      hwReg = HWReg::gpX(nextGp);
      comment("    ; alloc: x%u <= r%u", nextGp, frIndex);
      ++nextGp;
    } else
      break;

    frameRegs_[frIndex].globalReg = hwReg;
    frameRegs_[frIndex].globalType = FRType::Number;
  }
  // Non-pointer regs: allocate in gp regs first.
  for (unsigned frIndex = numCount; frIndex < npCount + numCount; ++frIndex) {
    HWReg hwReg;
    if (nextGp <= kGPSaved.second) {
      hwReg = HWReg::gpX(nextGp);
      comment("    ; alloc: x%u <= r%u", nextGp, frIndex);
      ++nextGp;
    } else if (nextVec <= kVecSaved.second) {
      hwReg = HWReg::vecD(nextVec);
      comment("    ; alloc: d%u <= r%u", nextVec, frIndex);
      ++nextVec;
    } else
      break;

    frameRegs_[frIndex].globalReg = hwReg;
    frameRegs_[frIndex].globalType = FRType::UnknownNonPtr;
  }

  bool hasExceptionTable = !codeBlock_->getRuntimeModule()
                                ->getBytecode()
                                ->getExceptionTable(codeBlock_->getFunctionID())
                                .empty();

  // A function with exception handlers must end up with no global registers.
  // longjmp restores callee-saved registers to their values at the setjmp,
  // so a global register would present a stale value to a catch handler;
  // correctness depends on every FR's canonical location being the memory
  // frame, which is also what makes the isInTry() sync in the throwing
  // emitters sufficient.
  //
  // Nothing here enforces that: it holds because
  // RegisterAllocator::getRegClass (lib/BCGen/RegAlloc.cpp) forces
  // RegClass::Other for any function containing a try, which leaves both
  // counts at zero. That is a contract with another module, so check it.
  assert(
      (!hasExceptionTable || (numCount == 0 && npCount == 0)) &&
      "function with exception handlers must have no global registers");

  if (hasExceptionTable)
    catchTableLabel_ = a.newNamedLabel("CATCH_TABLE");

  frameSetup(
      frameRegs_.size(), nextGp - kGPSaved.first, nextVec - kVecSaved.first);
}

void Emitter::commentV(const char *fmt, va_list args) {
  char buf[80];
  vsnprintf(buf, sizeof(buf), fmt, args);
  a.comment(buf);
}

JITCompiledFunctionPtr Emitter::addToRuntime(asmjit::JitRuntime &jr) {
  code.detach(&a);
  JITCompiledFunctionPtr fn;
  asmjit::Error err = jr.add(&fn, &code);
  if (err) {
    llvh::errs() << "AsmJit failed: " << asmjit::DebugUtils::errorAsString(err)
                 << "\n";
    hermes::hermes_fatal("AsmJit failed");
  }
  return fn;
}

#ifndef NDEBUG
void Emitter::assertPostInstructionInvariants() {
  for (const auto &frState : frameRegs_)
    assert(!frState.regIsDirty && "Frame register is dirty");

  // Check that any temps have an associated FR.
  for (unsigned i = kGPTemp.first; i <= kGPTemp.second; ++i) {
    HWReg hwReg(i, HWReg::GpX{});
    FR fr = hwRegs_[hwReg.combinedIndex()].contains;
    if (!fr.isValid()) {
      assert(gpTemp_.isAllocated(i) && "Temp register is not freed");
    }
  }

  for (unsigned i = kVecTemp1.first; i <= kVecTemp2.second; ++i) {
    if (i > kVecTemp1.second && i < kVecTemp2.first)
      continue;
    HWReg hwReg(i, HWReg::VecD{});
    FR fr = hwRegs_[hwReg.combinedIndex()].contains;
    if (!fr.isValid()) {
      assert(vecTemp_.isAllocated(i) && "Temp register is not freed");
    }
  }
}
#endif

void Emitter::newBasicBlock(const asmjit::Label &label) {
  syncAllFRTempExcept({});
  freeAllFRTempExcept({});

  // Clear all local types and regs when starting a new basic block.
  // TODO: there must be a faster way to do this when there are many regs.
  for (FRState &frState : frameRegs_) {
    frState.localType = frState.globalType;
    assert(!frState.localGpX);
    assert(!frState.localVecD);
    if (frState.globalReg)
      frState.frameUpToDate = false;
  }

  a.bind(label);
}

int32_t Emitter::getDebugFunctionName() {
  if (roOfsDebugFunctionName_ < 0) {
    std::string str;
    llvh::raw_string_ostream ss(str);
    ss << codeBlock_->getFunctionID() << "(" << codeBlock_->getNameString()
       << ")";
    ss.flush();
    int32_t size = str.size() + 1;
    roOfsDebugFunctionName_ = reserveData(size, 1, asmjit::TypeId::kInt8, size);
    memcpy(roData_.data() + roOfsDebugFunctionName_, str.data(), size);
  }
  return roOfsDebugFunctionName_;
}

void Emitter::frameSetup(
    unsigned numFrameRegs,
    unsigned gpSaveCount,
    unsigned vecSaveCount) {
  assert(
      gpSaveCount <= kGPSaved.second - kGPSaved.first + 1 &&
      "Too many callee saved GP regs");
  assert(
      vecSaveCount <= kVecSaved.second - kVecSaved.first + 1 &&
      "Too many callee saved Vec regs");

  static_assert(
      kGPSaved.first == 21, "Callee saved GP regs must start from x21");
  // Always save x21 even if it is not needed for an FR because we use it for
  // the return value.
  if (gpSaveCount == 0)
    gpSaveCount = 1;
  // We always save x19 and x20 since they are used for xRuntime and xFrame.
  gpSaveCount += 2;

  gpSaveCount_ = gpSaveCount;
  vecSaveCount_ = vecSaveCount;

  // Higher addresses are at the top.
  // +-----------------------------+<---- old sp
  // |             x30             |
  // +-----------------------------+
  // |             x29             |
  // +-----------------------------+<---- new x29
  // |             ...             |
  // +-----------------------------+
  // |             x21             |
  // +-----------------------------+
  // |             x20             |
  // +-----------------------------+
  // |             x19             |
  // +-----------------------------+
  // |  Saved SHLocals* (optional) |
  // +-----------------------------+
  // |      SHJmpBuf (optional)    |
  // +-----------------------------+<--- new sp
  a.sub(a64::sp, a64::sp, getStackSize());

  unsigned stackOfs = getSavedRegsOffset();
  for (unsigned i = 0; i < gpSaveCount; i += 2, stackOfs += 16) {
    if (i + 1 < gpSaveCount)
      a.stp(a64::GpX(19 + i), a64::GpX(20 + i), a64::Mem(a64::sp, stackOfs));
    else
      a.str(a64::GpX(19 + i), a64::Mem(a64::sp, stackOfs));
  }
  for (unsigned i = 0; i < vecSaveCount; i += 2, stackOfs += 16) {
    if (i + 1 < vecSaveCount)
      a.stp(
          a64::VecD(kVecSaved.first + i),
          a64::VecD(kVecSaved.first + 1 + i),
          a64::Mem(a64::sp, stackOfs));
    else
      a.str(a64::VecD(kVecSaved.first + i), a64::Mem(a64::sp, stackOfs));
  }
  a.stp(a64::x29, a64::x30, a64::Mem(a64::sp, stackOfs));
  a.add(a64::x29, a64::sp, stackOfs);

  comment("// xRuntime");
  a.mov(xRuntime, a64::x0);

  // Save the SHLocals pointer because we don't allocate and push a new
  // SHLocals in the JIT.
  // Used in CatchInst to restore state.
  if (catchTableLabel_.isValid()) {
    comment("// saved SHLocals *");
    a.ldr(a64::x0, a64::Mem(xRuntime, RuntimeOffsets::shLocals));
    a.str(a64::x0, a64::Mem(a64::sp, getSavedSHLocalsOffset()));
  }

#ifndef HERMES_CHECK_NATIVE_STACK
#error Only native stack checking is supported in the JIT
#endif

  comment("// _sh_check_native_stack_overflow");
  asmjit::Label nativeOverflowLab = newSlowPathLabel();
  asmjit::Label nativeOverflowContLab = newContLabel();
  // Get the stack bounds from the runtime.
  a.ldr(a64::x0, a64::Mem(xRuntime, RuntimeOffsets::nativeStackHigh));
  a.ldr(a64::x1, a64::Mem(xRuntime, RuntimeOffsets::nativeStackSize));
  // Subtract the frame pointer from nativeStackHigh and compare it against the
  // size. If the difference is less than the stack size, then we are still
  // within the current stack bounds.
  a.sub(a64::x0, a64::x0, a64::x29);
  a.cmp(a64::x0, a64::x1);
  // If the frame pointer is within bounds, we are done. Otherwise, we need to
  // check if the bounds have changed.
  a.b_hi(nativeOverflowLab);
  a.bind(nativeOverflowContLab);
  slowPaths_.push_back(
      {.slowPathLab = nativeOverflowLab,
       .contLab = nativeOverflowContLab,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment("// Slow path: _sh_check_native_stack_overflow");
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         // Do not save the IP because we have not yet set up the stack frame
         // for this function. If this throws, the exception should appear in
         // the caller.
         EMIT_RUNTIME_CALL_WITHOUT_THUNK_AND_SAVED_IP(
             em, void (*)(SHRuntime *), _sh_check_native_stack_overflow);
         em.a.b(sl.contLab);
       }});

  comment("// xFrame");
  a.ldr(xFrame, a64::Mem(xRuntime, RuntimeOffsets::stackPointer));

  // If the function has a prohibitInvoke flag, we need to check if it has been
  // called correctly.
  auto prohibitInvoke = codeBlock_->getHeaderFlags().getProhibitInvoke();
  if (prohibitInvoke != ProhibitInvoke::None) {
    // Load new.target.
    a.ldur(
        a64::x0,
        a64::Mem(
            xFrame, StackFrameLayout::NewTarget * (int)sizeof(SHLegacyValue)));
    // Compare new.target against undefined.
    emit_sh_ljs_is_undefined(a, a64::x0, a64::x0);

    void (*slowCall)(SHRuntime *);
    const char *slowCallName;
    asmjit::Label throwInvalidInvokeLab;
    if (prohibitInvoke == ProhibitInvoke::Call) {
      // If regular calls are prohibited, then we jump to throwInvalidInvoke if
      // new.target is undefined.
      throwInvalidInvokeLab = a.newNamedLabel("throwInvalidCall");
      a.b_eq(throwInvalidInvokeLab);

      slowCall = _sh_throw_invalid_call;
      slowCallName = "_sh_throw_invalid_call";
    } else {
      assert(
          prohibitInvoke == ProhibitInvoke::Construct &&
          "Unknown prohibitInvoke");
      // If construct calls are prohibited, then we jump to throwInvalidInvoke
      // if new.target is not undefined.
      throwInvalidInvokeLab = a.newNamedLabel("throwInvalidConstruct");
      a.b_ne(throwInvalidInvokeLab);

      slowCall = _sh_throw_invalid_construct;
      slowCallName = "_sh_throw_invalid_construct";
    }

    slowPaths_.push_back(
        {.slowPathLab = throwInvalidInvokeLab,
         .slowCall = (void *)slowCall,
         .slowCallName = slowCallName,
         .emit = [](Emitter &em, SlowPath &sl) {
           em.comment("// Slow path: %s", sl.slowCallName);
           em.a.bind(sl.slowPathLab);
           em.a.mov(a64::x0, xRuntime);
           // We don't register a thunk since there will only be a single call
           // to this. Note that we also don't save the IP, because this is
           // being thrown in the caller's context.
           em.callWithoutThunk(sl.slowCall, sl.slowCallName);
           // Function does not return.
         }});
  }

  // NOTE: Unlike _sh_enter, we do not push an SHLocals object.
  //  SHLegacyValue *frame = _sh_enter(shr, &locals.head, 13);
  comment("// _sh_enter");
  asmjit::Label registerOverflowLab = newSlowPathLabel();

  // Compute the remaining available stack space:
  // runtime.registerStackEnd - runtime.stackPointer
  a.ldr(a64::x0, a64::Mem(xRuntime, RuntimeOffsets::registerStackEnd));
  a.sub(a64::x0, a64::x0, xFrame);
  // Check if we need more registers than remain.
  size_t totalRegsToAlloc = numFrameRegs + hbc::StackFrameLayout::FirstLocal;
  size_t regAllocSize = totalRegsToAlloc * sizeof(SHLegacyValue);
  // NOTE: cmp has the same immediate field type as add/sub, so we can use the
  // same utility function.
  if (a64::Utils::isAddSubImm(regAllocSize)) {
    a.cmp(a64::x0, regAllocSize);
    a.b_lo(registerOverflowLab);
    a.add(a64::x0, xFrame, regAllocSize);
  } else {
    a.mov(a64::x1, regAllocSize);
    a.cmp(a64::x0, a64::x1);
    a.b_lo(registerOverflowLab);
    a.add(a64::x0, xFrame, a64::x1);
  }

  // Advance the register stack.
  a.str(a64::x0, a64::Mem(xRuntime, RuntimeOffsets::stackPointer));
  a.str(xFrame, a64::Mem(xRuntime, RuntimeOffsets::currentFrame));

  static_assert(
      HERMESVALUE_VERSION == 2, "Raw zero value must be ignored by GC");
  // Initialize the pointer to the current set of registers.
  a.mov(a64::x0, xFrame);
  size_t regsToFill = totalRegsToAlloc;
  // Fill the registers with zero in groups of 4, then 2, then 1.
  // If there are more than 32 registers, start with a loop.
  if (regsToFill > 32) {
    a.movi(a64::v0.d2(), 0);
    // We will fill 4 registers on each iteration.
    unsigned loopBytes = llvh::alignDown(regsToFill, 4) * sizeof(SHLegacyValue);
    // Initialize the loop limit in x1.
    if (a64::Utils::isAddSubImm(loopBytes)) {
      a.add(a64::x1, a64::x0, loopBytes);
    } else {
      a.mov(a64::x1, loopBytes);
      a.add(a64::x1, a64::x0, a64::x1);
    }
    asmjit::Label loop = a.newLabel();
    a.bind(loop);
    // Loop until we reach the limit.
    a.stp(a64::v0, a64::v0, a64::Mem(a64::x0).post(32));
    a.cmp(a64::x0, a64::x1);
    a.b_lo(loop);

    regsToFill %= 4;
  } else if (regsToFill >= 4) {
    a.movi(a64::v0.d2(), 0);
    // If the number of registers is small, just fill them directly.
    while (regsToFill >= 4) {
      a.stp(a64::v0, a64::v0, a64::Mem(a64::x0).post(32));
      regsToFill -= 4;
    }
  }
  // Fill any excess registers.
  if (regsToFill >= 2) {
    a.stp(a64::xzr, a64::xzr, a64::Mem(a64::x0).post(16));
    regsToFill -= 2;
  }
  if (regsToFill > 0) {
    assert(regsToFill == 1 && "All regs must be filled");
    a.str(a64::xzr, a64::Mem(a64::x0));
  }

  // Create the slow path for throwing a register stack overflow.
  slowPaths_.push_back(
      {.slowPathLab = registerOverflowLab,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment("// Slow path: _sh_throw_register_stack_overflow");
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         // Do not save the IP because we have not yet set up the stack frame
         // for this function. The exception should appear in the caller.
         EMIT_RUNTIME_CALL_WITHOUT_THUNK_AND_SAVED_IP(
             em, void (*)(SHRuntime *), _sh_throw_register_stack_overflow);
       }});

  if (catchTableLabel_.isValid()) {
    comment("// _sh_try");
    uint32_t jmpBufOffset = getJmpBufOffset();
    // buf->prev = shr->shCurJmpBuf;
    a.ldr(a64::x0, a64::Mem(xRuntime, offsetof(SHRuntime, shCurJmpBuf)));
    a.str(a64::x0, a64::Mem(a64::sp, jmpBufOffset + offsetof(SHJmpBuf, prev)));

    // shr->shCurJmpBuf = buf;
    a.add(a64::x0, a64::sp, jmpBufOffset);
    a.str(a64::x0, a64::Mem(xRuntime, offsetof(SHRuntime, shCurJmpBuf)));

    // _setjmp(buf->buf);
    a.add(a64::x0, a64::sp, jmpBufOffset + offsetof(SHJmpBuf, buf));
    // setjmp can't throw and it'll be called once, so don't use a thunk.
    EMIT_RUNTIME_CALL_WITHOUT_THUNK_AND_SAVED_IP(
        *this, int (*)(jmp_buf), _sh_setjmp);
    // If this a catch, go to the catch table to jump to either a handler BB or
    // rethrow.
    a.cbnz(a64::x0, catchTableLabel_);
  }

  if (dumpJitCode_ & DumpJitCode::EntryExit) {
    comment("// print entry");
    a.mov(a64::w0, 1);
    a.adr(a64::x1, roDataLabel_);
    a.add(a64::x1, a64::x1, getDebugFunctionName());
    EMIT_RUNTIME_CALL_WITHOUT_SAVED_IP(
        *this, void (*)(bool, const char *), _sh_print_function_entry_exit);
  }
}

void Emitter::leave(llvh::ArrayRef<const asmjit::Label *> exceptionHandlers) {
  comment("// leaveFrame");
  a.bind(returnLabel_);
  if (dumpJitCode_ & DumpJitCode::EntryExit) {
    comment("// print exit");
    a.mov(a64::w0, 0);
    a.adr(a64::x1, roDataLabel_);
    a.add(a64::x1, a64::x1, getDebugFunctionName());
    EMIT_RUNTIME_CALL_WITHOUT_SAVED_IP(
        *this, void (*)(bool, const char *), _sh_print_function_entry_exit);
  }

  if (catchTableLabel_.isValid()) {
    comment("// _sh_end_try");
    // shr->shCurJmpBuf = buf->prev
    uint32_t jmpBufOffset = getJmpBufOffset();
    a.ldr(a64::x0, a64::Mem(a64::sp, jmpBufOffset + offsetof(SHJmpBuf, prev)));
    a.str(a64::x0, a64::Mem(xRuntime, offsetof(SHRuntime, shCurJmpBuf)));
  }

  // _sh_leave(shr, &locals.head, frame);
  // Restore the previous stack frame.
  a.str(xFrame, a64::Mem(xRuntime, RuntimeOffsets::stackPointer));
  a.ldr(
      a64::x0,
      a64::Mem(
          xFrame,
          StackFrameLayout::PreviousFrame * (int)sizeof(SHLegacyValue)));
  a.str(a64::x0, a64::Mem(xRuntime, RuntimeOffsets::currentFrame));

  // The return value has been stashed in x21 by ret(). Move it to the return
  // register.
  a.mov(a64::x0, a64::x21);

  unsigned stackOfs = getSavedRegsOffset();
  for (unsigned i = 0; i < gpSaveCount_; i += 2, stackOfs += 16) {
    if (i + 1 < gpSaveCount_)
      a.ldp(a64::GpX(19 + i), a64::GpX(20 + i), a64::Mem(a64::sp, stackOfs));
    else
      a.ldr(a64::GpX(19 + i), a64::Mem(a64::sp, stackOfs));
  }
  for (unsigned i = 0; i < vecSaveCount_; i += 2, stackOfs += 16) {
    if (i + 1 < vecSaveCount_)
      a.ldp(
          a64::VecD(kVecSaved.first + i),
          a64::VecD(kVecSaved.first + 1 + i),
          a64::Mem(a64::sp, stackOfs));
    else
      a.ldr(a64::VecD(kVecSaved.first + i), a64::Mem(a64::sp, stackOfs));
  }
  a.ldp(a64::x29, a64::x30, a64::Mem(a64::sp, stackOfs));

  a.add(a64::sp, a64::sp, getStackSize());

  a.ret(a64::x30);

  emitCatchTable(exceptionHandlers);
  emitSlowPaths();
  emitThunks();
  emitROData();
}

void Emitter::callThunk(void *fn, const char *name) {
  // Using thunks leads to 0.27% more branch mispredicts and 2.8% performance
  // regression on an important React benchmark. So, disable them for now.
  if constexpr (false) {
    comment("// call %s", name);
    a.bl(registerThunk(fn, name));
  } else {
    callWithoutThunk(fn, name);
  }
}

void Emitter::callThunkWithSavedIP(void *fn, const char *name) {
  // Save the current IP in the runtime.
  getBytecodeIP(xScratch);
  a.str(xScratch, a64::Mem(xRuntime, RuntimeOffsets::currentIP));

  // Call the passed function.
  callThunk(fn, name);

  if (emitAsserts_) {
    // Invalidate the current IP to make sure it is set before the next call.
    a.mov(xScratch, Runtime::kInvalidCurrentIP);
    a.str(xScratch, a64::Mem(xRuntime, RuntimeOffsets::currentIP));
  }
}

void Emitter::callWithoutThunk(void *fn, const char *name) {
  comment("// call %s", name);
  loadBits64InGp(xScratch, (uint64_t)fn, name);
  a.blr(xScratch);
}

void Emitter::emitIncrementCounter(JitCounter counter) {
  if (!emitCounters_)
    return;
  // Push some registers onto the stack so we can use them.
  a.stp(a64::x0, a64::x1, a64::Mem(a64::sp).pre(-16));

  // Increment the counter.
  a.ldr(a64::x0, a64::Mem(xRuntime, RuntimeOffsets::runtimeJitCounters));
  a.ldr(a64::x1, a64::Mem(a64::x0, (unsigned)counter * sizeof(uint64_t)));
  a.add(a64::x1, a64::x1, 1);
  a.str(a64::x1, a64::Mem(a64::x0, (unsigned)counter * sizeof(uint64_t)));

  // Pop the saved values back off the stack.
  a.ldp(a64::x0, a64::x1, a64::Mem(a64::sp).post(16));
}

uint16_t Emitter::initHCLazyIDMayAlloc(HiddenClass *hc) {
  // Callers pass the result of WeakRoot::get(), which is null if the GC has
  // cleared the root. Since 0 already means "no id" and every caller checks
  // for it, tolerating null here keeps all present and future call sites safe
  // without each of them having to re-validate across safepoints.
  if (!hc)
    return 0;

  uint16_t id = hc->getLazyJITId();
  // Assign a new ID to the HC if we have to.
  if (id != 0)
    return id;

  // Too many IDs. Fail.
  if (jitImpl_.prevHCId >= jitImpl_.hcIdLimit)
    return 0;

  struct : Locals {
    PinnedValue<HiddenClass> hc;
  } lv;
  LocalsRAII lraii{runtime_, &lv};
  lv.hc = hc;

  if (jitImpl_.usedHCs.isUndefined()) {
    CallResult<HermesValue> cr = ArrayStorageSmall::create(runtime_, 8);
    if (LLVM_UNLIKELY(cr == ExecutionStatus::EXCEPTION)) {
      // Failing to pin is not fatal: report "no id" and let the caller fall
      // back to a non-specialized path. Swallow the pending OOM, since we are
      // in the compiler and there is nobody to propagate it to.
      runtime_.clearThrownValue();
      return 0;
    }
    // We would like to use a long-lived object, but we can't, because
    // ArrayStorage cannot be long-lived: it can be allocated that way
    // initially, but when it grows, it is allocated "normally".
    jitImpl_.usedHCs = *cr;
  }

  // Pin the class before assigning the id, so that the invariant
  // "id != 0 implies the class is in usedHCs" cannot be broken by a failed
  // allocation. usedHCs is the only strong root for these classes, and the id
  // is baked into immutable machine code, so an id on an unpinned class would
  // be a dangling reference.
  auto mh = MutableHandle<ArrayStorageSmall>::vmcast(&jitImpl_.usedHCs);
  if (LLVM_UNLIKELY(
          ArrayStorageSmall::push_back(mh, runtime_, lv.hc) ==
          ExecutionStatus::EXCEPTION)) {
    // Failing to pin is not fatal: report "no id" and let the caller fall
    // back to a non-specialized path. Swallow the pending OOM, since we are
    // in the compiler and there is nobody to propagate it to.
    runtime_.clearThrownValue();
    return 0;
  }

  id = ++jitImpl_.prevHCId;
  // Note: use the pinned handle rather than the raw argument, which the
  // allocation above may have invalidated by moving the object.
  lv.hc->setLazyJITId(id);

  return id;
}

void Emitter::loadFrameAddr(a64::GpX dst, FR frameReg) {
  auto ofs =
      (frameReg.index() + StackFrameLayout::FirstLocal) * sizeof(SHLegacyValue);
  // If the offset fits as an immediate, just emit an add.
  if (a64::Utils::isAddSubImm(ofs)) {
    a.add(dst, xFrame, ofs);
    return;
  }
  // We cannot add the offset as an immediate, so move it in first.
  a.mov(dst, ofs);
  a.add(dst, dst, xFrame);
}

void Emitter::getBytecodeIP(const a64::GpX &xOut) {
  auto ofs = codeBlock_->getOffsetOf(emittingIP);
  loadBits64InGp(xOut, (uint64_t)codeBlock_->begin(), "Bytecode start");
  // The add instruction takes a 12 bit immediate optionally shifted by 12 bits.
  // So we do the add as up to two 12 bit steps. Note that this means that it
  // will currently fail on any function that is larger than 16MB.
  auto low12Bits = ofs & llvh::maskTrailingOnes<uint32_t>(12);
  assert(a64::Utils::isAddSubImm(low12Bits) && "immediate should be 12 bits");
  a.add(xOut, xOut, low12Bits);
  if (auto restBits = ofs - low12Bits)
    a.add(xOut, xOut, restBits);
}

void Emitter::unreachable() {
  EMIT_RUNTIME_CALL(*this, void (*)(), _sh_unreachable);
}

void Emitter::profilePoint(uint16_t pointIndex) {
  comment("// ProfilePoint %u", pointIndex);
#ifdef HERMESVM_PROFILER_BB
  syncAllFRTempExcept({});
  freeAllFRTempExcept({});
  a.mov(a64::x0, xRuntime);
  a.mov(a64::w1, pointIndex);
  EMIT_RUNTIME_CALL(
      *this,
      void (*)(SHRuntime *, uint16_t),
      _interpreter_register_bb_execution);
#else
  // No-op if profiling is not enabled.
#endif
}

void Emitter::directEval(FR frRes, FR frText, bool strictCaller) {
  comment("// DirectEval r%u, r%u", frRes.index(), frText.index());
  syncAllFRTempExcept({});
  syncToFrame(frText);
  freeAllFRTempExcept({});

  a.mov(a64::x0, xRuntime);
  loadFrameAddr(a64::x1, frText);
  a.mov(a64::w2, strictCaller);
  EMIT_RUNTIME_CALL(
      *this,
      HermesValue (*)(Runtime &, PinnedHermesValue *, bool),
      _jit_direct_eval);

  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::gpX(0));
  movHWFromHW<true>(hwRes, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::catchInst(FR frRes) {
  comment("// Catch r%u", frRes.index());

  HWReg hwTemp = allocTempGpX();
  HWReg hwRes = getOrAllocFRInGpX(frRes, false);
  frUpdatedWithHW(frRes, hwRes);
  freeReg(hwTemp);

  // Catch simply returns the thrown value and clears it.

  // Read thrown value.
  a.ldr(hwRes.a64GpX(), a64::Mem(xRuntime, RuntimeOffsets::thrownValue));
  // Clear thrown value.
  loadBits64InGp(hwTemp.a64GpX(), _sh_ljs_empty().raw, "empty");
  a.str(hwTemp.a64GpX(), a64::Mem(xRuntime, RuntimeOffsets::thrownValue));
}

void Emitter::ret(FR frValue) {
  movHWFromFR(HWReg::gpX(21), frValue);
  a.b(returnLabel_);
}

void Emitter::mov(FR frRes, FR frInput, bool logComment) {
  // Sometimes mov() is used by other instructions, so logging is optional.
  if (logComment)
    comment("// %s r%u, r%u", "mov", frRes.index(), frInput.index());
  if (frRes == frInput)
    return;

  HWReg hwInput = getOrAllocFRInAnyReg(frInput, true);
  HWReg hwDest = getOrAllocFRInAnyReg(frRes, false);
  movHWFromHW<false>(hwDest, hwInput);
  frUpdatedWithHW(frRes, hwDest, frameRegs_[frInput.index()].localType);
}

void Emitter::loadParam(FR frRes, uint32_t paramIndex) {
  comment("// LoadParam r%u, %u", frRes.index(), paramIndex);

  asmjit::Error err;
  asmjit::Label slowPathLab = newSlowPathLabel();
  asmjit::Label contLab = newContLabel();

  HWReg hwTmp = allocAndLogTempGpX();
  a64::GpW wTmp(hwTmp.indexInClass());

  a.ldur(
      wTmp,
      a64::Mem(
          xFrame,
          (int)StackFrameLayout::ArgCount * (int)sizeof(SHLegacyValue)));

  EXPECT_ERROR(asmjit::kErrorInvalidImmediate, err = a.cmp(wTmp, paramIndex));
  // Does paramIndex fit in the 12-bit unsigned immediate?
  if (err) {
    HWReg hwTmp2 = allocAndLogTempGpX();
    a64::GpW wTmp2(hwTmp2.indexInClass());
    loadBits64InGp(wTmp2, paramIndex, "paramIndex");
    a.cmp(wTmp, wTmp2);
    freeReg(hwTmp2);
  }
  a.b_lo(slowPathLab);

  freeReg(hwTmp);

  HWReg hwRes = getOrAllocFRInGpX(frRes, false);

  // Compute the frame offset in 64 bits. paramIndex is a UInt32 operand of
  // LoadParamLong, and (ThisArg - paramIndex) * sizeof(SHLegacyValue)
  // overflows int32 from paramIndex 2^28 upwards, long before that range is
  // exhausted.
  int64_t ofs64 = ((int64_t)StackFrameLayout::ThisArg - (int64_t)paramIndex) *
      (int64_t)sizeof(SHLegacyValue);
  assert(ofs64 < 0 && "frame offset of a parameter must be negative");

  if (LLVM_UNLIKELY(ofs64 <= std::numeric_limits<int32_t>::min())) {
    // The parameter is so far away that no argument count can reach it, so
    // the comparison above always branches. Emit the branch unconditionally
    // rather than a fast path that cannot be reached and whose offset would
    // not encode; the slow path yields undefined, which is the right answer.
    a.b(slowPathLab);
  } else {
    int32_t ofs = (int32_t)ofs64;
    EXPECT_ERROR(
        asmjit::kErrorInvalidDisplacement,
        err = a.ldur(hwRes.a64GpX(), a64::Mem(xFrame, ofs)));
    // Does the offset fit in the 9-bit signed offset?
    if (err) {
      ofs = -ofs;
      a64::GpX xRes = hwRes.a64GpX();
      if (ofs <= 4095) {
        a.sub(xRes, xFrame, ofs);
      } else {
        loadBits64InGp(xRes, ofs, nullptr);
        a.sub(xRes, xFrame, xRes);
      }
      a.ldr(xRes, a64::Mem(xRes));
    }
  }

  a.bind(contLab);
  frUpdatedWithHW(frRes, hwRes);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .frRes = frRes,
       .hwRes = hwRes,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment("// Slow path: LoadParam r%u", sl.frRes.index());
         em.a.bind(sl.slowPathLab);
         em.loadBits64InGp(
             sl.hwRes.a64GpX(), _sh_ljs_undefined().raw, "undefined");
         em.a.b(sl.contLab);
       }});
}

void Emitter::toNumber(FR frRes, FR frInput) {
  comment("// %s r%u, r%u", "toNumber", frRes.index(), frInput.index());
  if (isFRKnownNumber(frInput))
    return mov(frRes, frInput, false);

  HWReg hwRes, hwInput;
  asmjit::Label slowPathLab = newSlowPathLabel();
  asmjit::Label contLab = newContLabel();
  syncAllFRTempExcept(frRes != frInput ? frRes : FR());
  syncToFrame(frInput);

  hwInput = getOrAllocFRInVecD(frInput, true);

  // We don't free frRes so that if it is the same as frThis, the register is
  // simply persisted and we do not need to perform a move in the fast path.
  freeAllFRTempExcept(frRes);
  hwRes = getOrAllocFRInVecD(frRes, false);
  frUpdatedWithHW(frRes, hwRes, FRType::Number);

  // Since HermesValue is NaN-boxed we know that all non-number values will be
  // NaN. So we can conveniently test for non-number values by checking for NaN
  // (which does not compare equal to itself).
  static_assert(HERMESVALUE_VERSION == 2, "Non-numbers must be NaN");
  a.fcmp(hwInput.a64VecD(), hwInput.a64VecD());
  a.b_ne(slowPathLab);
  movHWFromHW<false>(hwRes, hwInput);

  a.bind(contLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .frRes = frRes,
       .frInput1 = frInput,
       .hwRes = hwRes,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// Slow path: toNumber r%u, r%u",
             sl.frRes.index(),
             sl.frInput1.index());
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         em.loadFrameAddr(a64::x1, sl.frInput1);
         EMIT_RUNTIME_CALL(
             em,
             double (*)(SHRuntime *, const SHLegacyValue *),
             _sh_ljs_to_double_rjs);
         em.movHWFromHW<false>(sl.hwRes, HWReg::vecD(0));
         em.a.b(sl.contLab);
       }});
}

void Emitter::toNumeric(FR frRes, FR frInput) {
  comment("// %s r%u, r%u", "toNumeric", frRes.index(), frInput.index());
  if (isFRKnownNumber(frInput))
    return mov(frRes, frInput, false);

  HWReg hwRes, hwInput;
  asmjit::Label slowPathLab = newSlowPathLabel();
  asmjit::Label contLab = newContLabel();
  syncAllFRTempExcept(frRes != frInput ? frRes : FR());
  syncToFrame(frInput);

  hwInput = getOrAllocFRInVecD(frInput, true);

  // We don't free frRes so that if it is the same as frThis, the register is
  // simply persisted and we do not need to perform a move in the fast path.
  freeAllFRTempExcept(frRes);
  hwRes = getOrAllocFRInVecD(frRes, false);
  frUpdatedWithHW(frRes, hwRes, FRType::UnknownPtr);

  // Since HermesValue is NaN-boxed we know that all non-number values will be
  // NaN. So we can conveniently test for non-number values by checking for NaN
  // (which does not compare equal to itself).
  static_assert(HERMESVALUE_VERSION == 2, "Non-numbers must be NaN");
  a.fcmp(hwInput.a64VecD(), hwInput.a64VecD());
  a.b_ne(slowPathLab);
  movHWFromHW<false>(hwRes, hwInput);

  a.bind(contLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .frRes = frRes,
       .frInput1 = frInput,
       .hwRes = hwRes,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// Slow path: toNumeric r%u, r%u",
             sl.frRes.index(),
             sl.frInput1.index());
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         em.loadFrameAddr(a64::x1, sl.frInput1);
         EMIT_RUNTIME_CALL(
             em,
             SHLegacyValue (*)(SHRuntime *, const SHLegacyValue *),
             _sh_ljs_to_numeric_rjs);
         em.movHWFromHW<false>(sl.hwRes, HWReg::gpX(0));
         em.a.b(sl.contLab);
       }});
}

void Emitter::toInt32(FR frRes, FR frInput, bool isSigned) {
  comment(
      "// %s r%u, r%u",
      isSigned ? "ToInt32" : "ToUint32",
      frRes.index(),
      frInput.index());

  HWReg hwTempGpX = allocTempGpX();
  HWReg hwTempVecD = allocTempVecD();

  syncAllFRTempExcept(frRes != frInput ? frRes : FR());
  // TODO: As with binary bit ops, it should be possible to only do this in the
  // slow path.
  syncToFrame(frInput);

  asmjit::Label slowPathLab = newSlowPathLabel();
  asmjit::Label contLab = newContLabel();

  HWReg hwInput = getOrAllocFRInVecD(frInput, true);
  emit_double_is_int(
      a, hwTempGpX.a64GpX(), hwTempVecD.a64VecD(), hwInput.a64VecD());
  a.b_ne(slowPathLab);

  // Done allocating registers. Free them all and allocate the result.
  freeAllFRTempExcept({});
  freeReg(hwTempGpX);
  freeReg(hwTempVecD);
  HWReg hwRes = getOrAllocFRInVecD(frRes, false);
  frUpdatedWithHW(frRes, hwRes, FRType::Number);

  if (isSigned) {
    // Convert int32 back to double.
    a.scvtf(hwRes.a64VecD(), hwTempGpX.a64GpX().w());
  } else {
    // Convert uint32 back to double.
    a.ucvtf(hwRes.a64VecD(), hwTempGpX.a64GpX().w());
  }

  a.bind(contLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .name = isSigned ? "ToInt32" : "ToUint32",
       .frRes = frRes,
       .frInput1 = frInput,
       .hwRes = hwRes,
       .slowCall = isSigned ? (void *)_sh_ljs_to_int32_rjs
                            : (void *)_sh_ljs_to_uint32_rjs,
       .slowCallName =
           isSigned ? "_sh_ljs_to_int32_rjs" : "_sh_ljs_to_uint32_rjs",
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// Slow path: %s, r%u, r%u, r%u",
             sl.name,
             sl.frRes.index(),
             sl.frInput1.index(),
             sl.frInput2.index());
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         em.loadFrameAddr(a64::x1, sl.frInput1);
         em.callThunkWithSavedIP((void *)sl.slowCall, sl.slowCallName);
         em.movHWFromHW<false>(sl.hwRes, HWReg::vecD(0));
         em.a.b(sl.contLab);
       }});
}

void Emitter::addEmptyString(FR frRes, FR frInput) {
  comment("// AddEmptyString r%u, r%u", frRes.index(), frInput.index());

  syncAllFRTempExcept(frRes != frInput ? frRes : FR());
  // TODO: As with binary bit ops, it should be possible to only do this in the
  // slow path.
  syncToFrame(frInput);

  asmjit::Label slowPathLab = newSlowPathLabel();
  asmjit::Label contLab = newContLabel();

  HWReg hwInput = getOrAllocFRInGpX(frInput, true);
  HWReg hwTemp = allocTempGpX();
  freeReg(hwTemp);
  freeAllFRTempExcept(frRes);

  HWReg hwRes = getOrAllocFRInGpX(frRes, false);

  // Check if the input is already a string and don't do anything.
  emit_sh_ljs_is_string(a, hwTemp.a64GpX(), hwInput.a64GpX());
  a.b_ne(slowPathLab);

  // Fast path.
  movHWFromHW<false>(hwRes, hwInput);
  frUpdatedWithHW(frRes, hwRes, FRType::Pointer);

  a.bind(contLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .frRes = frRes,
       .frInput1 = frInput,
       .hwRes = hwRes,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// Slow path: AddEmptyString r%u, r%u",
             sl.frRes.index(),
             sl.frInput1.index());
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         em.loadFrameAddr(a64::x1, sl.frInput1);
         EMIT_RUNTIME_CALL(
             em,
             SHLegacyValue (*)(SHRuntime *, const SHLegacyValue *),
             _sh_ljs_add_empty_string_rjs);
         em.movHWFromHW<false>(sl.hwRes, HWReg::gpX(0));
         em.a.b(sl.contLab);
       }});
}

void Emitter::newArray(FR frRes, uint32_t size) {
  comment("// NewArray r%u, %u", frRes.index(), size);
  syncAllFRTempExcept(frRes);
  freeAllFRTempExcept({});
  a.mov(a64::x0, xRuntime);
  a.mov(a64::w1, size);
  EMIT_RUNTIME_CALL(
      *this, SHLegacyValue (*)(SHRuntime *, uint32_t), _sh_ljs_new_array);
  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::gpX(0));
  movHWFromHW<false>(hwRes, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::newArrayWithBuffer(
    FR frRes,
    uint32_t numElements,
    uint32_t numLiterals,
    uint32_t bufferIndex) {
  comment(
      "// NewArrayWithBuffer r%u, %u, %u, %u",
      frRes.index(),
      numElements,
      numLiterals,
      bufferIndex);

  syncAllFRTempExcept(frRes);
  freeAllFRTempExcept({});
  a.mov(a64::x0, xRuntime);
  loadBits64InGp(a64::x1, (uint64_t)codeBlock_, "CodeBlock");
  a.mov(a64::w2, numElements);
  a.mov(a64::w3, numLiterals);
  a.mov(a64::w4, bufferIndex);
  EMIT_RUNTIME_CALL(
      *this,
      SHLegacyValue (*)(
          SHRuntime *, SHCodeBlock *, uint32_t, uint32_t, uint32_t),
      _interpreter_create_array_from_buffer);
  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::gpX(0));
  movHWFromHW<false>(hwRes, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::newFastArray(FR frRes, FR frProto, uint32_t size) {
  comment("// NewFastArray r%u, r%u, %u", frRes.index(), frProto.index(), size);
  syncAllFRTempExcept(frRes);
  syncToFrame(frProto);
  freeAllFRTempExcept({});
  a.mov(a64::x0, xRuntime);
  loadFrameAddr(a64::x1, frProto);
  a.mov(a64::w2, size);
  EMIT_RUNTIME_CALL(
      *this,
      SHLegacyValue (*)(SHRuntime *, SHLegacyValue *, uint32_t),
      _sh_new_fastarray_with_proto);
  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::gpX(0));
  movHWFromHW<false>(hwRes, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::fastArrayLength(FR frRes, FR frArr) {
  comment("// FastArrayLength r%u, r%u", frRes.index(), frArr.index());
  // We allocate a temporary register to compute the address instead of using
  // the result register in case the result has a VecD allocated for it.
  HWReg temp = allocTempGpX();
  HWReg hwArr = getOrAllocFRInGpX(frArr, true);
  // Done allocating, free the temp so it can be reused for the result.
  freeReg(temp);
  emit_sh_ljs_get_pointer(a, temp.a64GpX(), hwArr.a64GpX());

#ifdef HERMESVM_BOXED_DOUBLES
  // If boxed doubles are enabled, load the size from the ArrayStorage, where it
  // is stored as an integer.
  emit_load_cp(
      a,
      temp.a64GpX(),
      a64::Mem(temp.a64GpX(), offsetof(SHFastArray, indexedStorage)));
  emit_sh_cp_decode_non_null(a, temp.a64GpX());
  a.ldr(
      temp.a64GpX().w(),
      a64::Mem(temp.a64GpX(), offsetof(SHArrayStorage, size)));
  HWReg hwRes = getOrAllocFRInVecD(frRes, false);
  a.ucvtf(hwRes.a64VecD(), temp.a64GpX().w());
#else
  // If boxed doubles are disabled, we can just load the size from the length
  // property of the FastArray.
  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false);
  movHWFromMem(hwRes, a64::Mem(temp.a64GpX(), offsetof(SHFastArray, length)));
#endif

  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::fastArrayLoad(FR frRes, FR frArr, FR frIdx) {
  comment(
      "// FastArrayLoad r%u, r%u, r%u",
      frRes.index(),
      frArr.index(),
      frIdx.index());
#if defined(HERMESVM_COMPRESSED_POINTERS) || defined(HERMESVM_BOXED_DOUBLES)
  syncAllFRTempExcept(frRes != frArr && frRes != frIdx ? frRes : FR());
  syncToFrame(frArr);
  freeAllFRTempExcept({});
  a.mov(a64::x0, xRuntime);
  loadFrameAddr(a64::x1, frArr);
  movHWFromFR(HWReg::vecD(0), frIdx);
  EMIT_RUNTIME_CALL(
      *this,
      SHLegacyValue (*)(SHRuntime *, SHLegacyValue *, double idx),
      _sh_fastarray_load);
  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::gpX(0));
  movHWFromHW<false>(hwRes, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes);
#else
  asmjit::Label slowPathLab = newSlowPathLabel();
  // We allocate a temporary register to compute the address instead of using
  // the result register in case the result has a VecD allocated for it.
  HWReg hwTmpStorage = allocTempGpX();
  HWReg hwTmpSize = allocTempGpX();
  HWReg hwTmpIdxGpX = allocTempGpX();
  HWReg hwTmpIdxVecD = allocTempVecD();
  HWReg hwArr = getOrAllocFRInGpX(frArr, true);
  HWReg hwIdx = getOrAllocFRInVecD(frIdx, true);
  // Done allocating, free the temps so they can be reused for the result.
  freeReg(hwTmpStorage);
  freeReg(hwTmpSize);
  freeReg(hwTmpIdxGpX);
  freeReg(hwTmpIdxVecD);

  // Retrieve the FastArray pointer and use it to load the indexed storage
  // pointer.
  emit_sh_ljs_get_pointer(a, hwTmpStorage.a64GpX(), hwArr.a64GpX());
  movHWFromMem(
      hwTmpStorage,
      a64::Mem(hwTmpStorage.a64GpX(), offsetof(SHFastArray, indexedStorage)));

  // Load the size from the indexed storage.
  a.ldr(
      hwTmpSize.a64GpX().w(),
      a64::Mem(hwTmpStorage.a64GpX(), offsetof(SHArrayStorageSmall, size)));

  // Check if the index is a uint32.
  emit_double_is_uint32(
      a, hwTmpIdxGpX.a64GpX().w(), hwTmpIdxVecD.a64VecD(), hwIdx.a64VecD());
  // If the conversion was successful, compare the size against the index.
  // Otherwise, set the flags to zero to force the subsequent b_ls to be taken.
  a.ccmp(
      hwTmpSize.a64GpX().w(), hwTmpIdxGpX.a64GpX().w(), 0, a64::CondCode::kEQ);
  // If the index is out-of-bounds jump to the failure path.
  // We will have to sync registers when the access is inside a try region
  // because we could read from the FRs again in this function.
  if (isInTry())
    syncAllFRTempExcept(frRes != frArr && frRes != frIdx ? frRes : FR());
  a.b_ls(slowPathLab);

  // Add the offset of the actual data in the ArrayStorage.
  a.add(
      hwTmpStorage.a64GpX(),
      hwTmpStorage.a64GpX(),
      offsetof(SHArrayStorageSmall, storage));

  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false);
  movHWFromMem(
      hwRes,
      a64::Mem(hwTmpStorage.a64GpX(), hwTmpIdxGpX.a64GpX(), a64::lsl(3)));
  frUpdatedWithHW(frRes, hwRes);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .frRes = frRes,
       .frInput1 = frArr,
       .frInput2 = frIdx,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// Slow path: FastArrayLoad r%u, r%u, r%u",
             sl.frRes.index(),
             sl.frInput1.index(),
             sl.frInput2.index());
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         EMIT_RUNTIME_CALL(em, void (*)(SHRuntime *), _sh_throw_array_oob);
         // Call does not return.
       }});
#endif
}

void Emitter::fastArrayStore(FR frArr, FR frIdx, FR frVal) {
  comment(
      "// FastArrayStore r%u, r%u, r%u",
      frArr.index(),
      frIdx.index(),
      frVal.index());
  syncAllFRTempExcept({});
  syncToFrame(frArr);
  syncToFrame(frVal);
  freeAllFRTempExcept({});
  a.mov(a64::x0, xRuntime);
  loadFrameAddr(a64::x1, frVal);
  loadFrameAddr(a64::x2, frArr);
  movHWFromFR(HWReg{a64::d0}, frIdx);
  EMIT_RUNTIME_CALL(
      *this,
      void (*)(SHRuntime *, const SHLegacyValue *, SHLegacyValue *, double idx),
      _sh_fastarray_store);
}

void Emitter::fastArrayPush(FR frArr, FR frVal) {
  comment("// FastArrayPush r%u, r%u", frArr.index(), frVal.index());
  syncAllFRTempExcept({});
  syncToFrame(frArr);
  syncToFrame(frVal);
  freeAllFRTempExcept({});
  a.mov(a64::x0, xRuntime);
  loadFrameAddr(a64::x1, frVal);
  loadFrameAddr(a64::x2, frArr);
  EMIT_RUNTIME_CALL(
      *this,
      void (*)(SHRuntime *, SHLegacyValue *, SHLegacyValue *),
      _sh_fastarray_push);
}

void Emitter::fastArrayAppend(FR frArr, FR frOther) {
  comment("// FastArrayAppend r%u, r%u", frArr.index(), frOther.index());
  syncAllFRTempExcept({});
  syncToFrame(frArr);
  syncToFrame(frOther);
  freeAllFRTempExcept({});
  a.mov(a64::x0, xRuntime);
  loadFrameAddr(a64::x1, frOther);
  loadFrameAddr(a64::x2, frArr);
  EMIT_RUNTIME_CALL(
      *this,
      void (*)(SHRuntime *, SHLegacyValue *, SHLegacyValue *),
      _sh_fastarray_append);
}

void Emitter::getGlobalObject(FR frRes) {
  comment("// GetGlobalObject r%u", frRes.index());
  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false);
  movHWFromMem(hwRes, a64::Mem(xRuntime, RuntimeOffsets::globalObject));
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::declareGlobalVar(SHSymbolID symID) {
  comment("// DeclareGlobalVar %u", symID);

  syncAllFRTempExcept({});
  freeAllFRTempExcept({});

  a.mov(a64::x0, xRuntime);
  a.mov(a64::w1, symID);
  EMIT_RUNTIME_CALL(
      *this, void (*)(SHRuntime *, SHSymbolID), _sh_ljs_declare_global_var);
}

void Emitter::createTopLevelEnvironment(FR frRes, uint32_t size) {
  comment("// CreateTopLevelEnvironment r%u, %u", frRes.index(), size);

  syncAllFRTempExcept(frRes);
  freeAllFRTempExcept({});

  a.mov(a64::x0, xRuntime);
  a.mov(a64::x1, 0);
  a.mov(a64::w2, size);

  EMIT_RUNTIME_CALL(
      *this,
      SHLegacyValue (*)(SHRuntime *, const SHLegacyValue *, uint32_t),
      _sh_ljs_create_environment);

  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::gpX(0));
  movHWFromHW<false>(hwRes, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::createFunctionEnvironment(FR frRes, uint32_t size) {
  comment("// CreateFunctionEnvironment r%u, %u", frRes.index(), size);

  syncAllFRTempExcept({});
  freeAllFRTempExcept({});

  // Allocate the result register.
  HWReg hwRes = getOrAllocFRInGpX(frRes, false, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes, FRType::Pointer);
  auto xRes = hwRes.a64GpX();

  // Allocate some temporaries.
  HWReg hwTemp1 = allocTempGpX();
  HWReg hwTemp2 = allocTempGpX();
  auto hwTempV = allocTempVecD();
  auto xTemp1 = hwTemp1.a64GpX();
  auto xTemp2 = hwTemp2.a64GpX();
  auto vTemp = hwTempV.a64VecD().v();
  freeReg(hwTemp1);
  freeReg(hwTemp2);
  freeReg(hwTempV);

  asmjit::Label slowPathLab = newSlowPathLabel();
  asmjit::Label contLab = newContLabel();

  // Try to allocate the new environment cell.
  allocInYoung(
      CellKind::EnvironmentKind,
      Environment::allocationSize(size),
      xRes,
      xTemp1,
      xTemp2,
      slowPathLab);

  // Get the current closure pointer.
  a.ldur(
      xTemp1,
      a64::Mem(
          xFrame,
          (int)StackFrameLayout::CalleeClosureOrCB *
              (int)sizeof(SHLegacyValue)));
  emit_sh_ljs_get_pointer(a, xTemp1, xTemp1);

  // xTemp1 = closure->environment
  emit_load_cp(a, xTemp1, a64::Mem(xTemp1, offsetof(SHCallable, environment)));

  // Initialize the environment cell.
  emit_environment_init(a, xRes, /* xParentEnv */ xTemp1, xTemp2, vTemp, size);

  // Encode the cell as a HermesValue.
  emit_sh_ljs_object(a, xRes);

  a.bind(contLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .frRes = frRes,
       .hwRes = hwRes,
       .sizeOrIdx = size,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// Slow path: CreateFunctionEnvironment r%u", sl.frRes.index());
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         em.a.mov(a64::x1, xFrame);
         em.a.mov(a64::w2, sl.sizeOrIdx);

         EMIT_RUNTIME_CALL(
             em,
             SHLegacyValue (*)(SHRuntime *, SHLegacyValue *, uint32_t),
             _sh_ljs_create_function_environment);
         em.movHWFromHW<false>(sl.hwRes, HWReg::gpX(0));
         em.a.b(sl.contLab);
       }});
}

void Emitter::createEnvironment(FR frRes, FR frParent, uint32_t size) {
  comment(
      "// CreateEnvironment r%u, r%u, %u",
      frRes.index(),
      frParent.index(),
      size);

  syncAllFRTempExcept(frRes != frParent ? frRes : FR{});
  syncToFrame(frParent);
  auto hwParent = getOrAllocFRInGpX(frParent, true);
  auto hwTemp1 = allocTempGpX();
  auto hwTemp2 = allocTempGpX();
  auto hwNewEnvPtr = allocTempGpX();
  auto hwTempV = allocTempVecD();

  auto xTemp1 = hwTemp1.a64GpX();
  auto xTemp2 = hwTemp2.a64GpX();
  auto xNewEnvPtr = hwNewEnvPtr.a64GpX();
  auto vTemp = hwTempV.a64VecD().v();

  freeReg(hwTemp1);
  freeReg(hwTemp2);
  freeReg(hwNewEnvPtr);
  freeReg(hwTempV);

  freeAllFRTempExcept({});

  asmjit::Label slowPathLab = newSlowPathLabel();
  asmjit::Label contLab = newContLabel();

  allocInYoung(
      CellKind::EnvironmentKind,
      Environment::allocationSize(size),
      xNewEnvPtr,
      xTemp1,
      xTemp2,
      slowPathLab);

  // Load a compressed pointer to the parent environment in xTemp1.
  emit_sh_ljs_get_pointer(a, xTemp1, hwParent.a64GpX());
  emit_sh_cp_encode_non_null(a, xTemp1);

  emit_environment_init(
      a, xNewEnvPtr, /* xParentEnv */ xTemp1, xTemp2, vTemp, size);

  // Finally, allocate the result register.
  HWReg hwRes = getOrAllocFRInGpX(frRes, false, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes, FRType::Pointer);

  emit_sh_ljs_object2(a, hwRes.a64GpX(), xNewEnvPtr);

  a.bind(contLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .frRes = frRes,
       .frInput1 = frParent,
       .hwRes = hwRes,
       .sizeOrIdx = size,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// Slow path: CreateEnvironment r%u, r%u",
             sl.frRes.index(),
             sl.frInput1.index());
         em.a.bind(sl.slowPathLab);

         em.a.mov(a64::x0, xRuntime);
         em.loadFrameAddr(a64::x1, sl.frInput1);
         em.a.mov(a64::w2, sl.sizeOrIdx);

         EMIT_RUNTIME_CALL(
             em,
             SHLegacyValue (*)(SHRuntime *, const SHLegacyValue *, uint32_t),
             _sh_ljs_create_environment);

         em.movHWFromHW<false>(sl.hwRes, HWReg::gpX(0));
         em.a.b(sl.contLab);
       }});
}

void Emitter::getParentEnvironment(FR frRes, uint32_t level) {
  comment("// GetParentEnvironment r%u, %u", frRes.index(), level);

  HWReg hwRes = getOrAllocFRInGpX(frRes, false);
  a64::GpX xRes = hwRes.a64GpX();
  frUpdatedWithHW(frRes, hwRes);

  // Get current closure.
  a.ldur(
      xRes,
      a64::Mem(
          xFrame,
          (int)StackFrameLayout::CalleeClosureOrCB *
              (int)sizeof(SHLegacyValue)));
  // get pointer.
  emit_sh_ljs_get_pointer(a, xRes, xRes);
  // xRes = closure->environment
  emit_load_cp(a, xRes, a64::Mem(xRes, offsetof(SHCallable, environment)));
  emit_sh_cp_decode_non_null(a, xRes);
  for (; level; --level) {
    // xRes = env->parent.
    emit_load_cp(
        a, xRes, a64::Mem(xRes, offsetof(SHEnvironment, parentEnvironment)));
    emit_sh_cp_decode_non_null(a, xRes);
  }
  // encode object.
  emit_sh_ljs_object(a, xRes);
}

void Emitter::getEnvironment(FR frRes, FR frSource, uint32_t level) {
  comment(
      "// GetEnvironment r%u, r%u, %u", frRes.index(), frSource.index(), level);

  HWReg hwSource = getOrAllocFRInGpX(frSource, true);
  HWReg hwRes = getOrAllocFRInGpX(frRes, false);
  frUpdatedWithHW(frRes, hwRes);
  a64::GpX xRes = hwRes.a64GpX();

  emit_sh_ljs_get_pointer(a, xRes, hwSource.a64GpX());
  for (; level; --level) {
    // xRes = env->parent.
    emit_load_cp(
        a, xRes, a64::Mem(xRes, offsetof(SHEnvironment, parentEnvironment)));
    emit_sh_cp_decode_non_null(a, xRes);
  }
  // encode object.
  emit_sh_ljs_object(a, xRes);
}

void Emitter::getClosureEnvironment(FR frRes, FR frClosure) {
  comment(
      "// GetClosureEnvironment r%u, r%u", frRes.index(), frClosure.index());
  // We know the layout of the closure, so we can load directly.
  auto ofs = offsetof(SHCallable, environment);
  auto hwClosure = getOrAllocFRInGpX(frClosure, true);
  auto hwRes = getOrAllocFRInGpX(frRes, false);
  // Use the result register as a scratch register for computing the address.
  emit_sh_ljs_get_pointer(a, hwRes.a64GpX(), hwClosure.a64GpX());
  emit_load_cp(a, hwRes.a64GpX(), a64::Mem(hwRes.a64GpX(), ofs));
  emit_sh_cp_decode_non_null(a, hwRes.a64GpX());
  // The result is a pointer, so add the object tag.
  emit_sh_ljs_object(a, hwRes.a64GpX());
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::loadFromEnvironment(FR frRes, FR frEnv, uint32_t slot) {
  comment(
      "// LoadFromEnvironment r%u, r%u, %u",
      frRes.index(),
      frEnv.index(),
      slot);

  // TODO: register allocation could be smarter if frRes !=- frEnv.

  HWReg hwTmp1 = allocTempGpX();
  a64::GpX xTmp1 = hwTmp1.a64GpX();

  movHWFromFR(hwTmp1, frEnv);
  // get pointer.
  emit_sh_ljs_get_pointer(a, xTmp1, xTmp1);

  // xScratch is touched only if the offset is too large for the scaled
  // immediate: LoadFromEnvironmentL carries a UInt16 slot, and past slot
  // ~4092 the displacement no longer encodes.
  emit_load_from_base_offset<sizeof(SHLegacyValue)>(
      a,
      xTmp1,
      xTmp1,
      xScratch,
      offsetof(SHEnvironment, slots) + sizeof(SHLegacyValue) * slot);

  freeReg(hwTmp1);
  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, hwTmp1);
  movHWFromHW<false>(hwRes, hwTmp1);
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::storeToEnvironment(bool np, FR frEnv, uint32_t slot, FR frValue) {
  // TODO: this should really be inlined!
  comment(
      np ? "// StoreNPToEnvironment r%u, %u, r%u"
         : "// StoreToEnvironment r%u, %u, r%u",
      frEnv.index(),
      slot,
      frValue.index());

  // Here we apply a technique that may be subtle. We have various FRs that we
  // want to load into parameter registers (x0, x1, etc) by value. Some of these
  // FRs may live in the parameter registers we want to use, but some may not.
  // So, first we make sure that the FRs that live in x0, x1, etc., are synced
  // to their primary location and the temps x0, x1, etc., are freed.
  // As we do this, we immediately move the corresponding parameter from its
  // corresponding FR, to maximize the chance that it can be moved from a
  // register.

  // Make sure x0, x1, x2, x3 are unused.
  syncAndFreeTempReg(HWReg::gpX(0));
  a.mov(a64::x0, xRuntime);

  syncAndFreeTempReg(HWReg::gpX(1));
  movHWFromFR(HWReg::gpX(1), frEnv);

  syncAndFreeTempReg(HWReg::gpX(2));
  movHWFromFR(HWReg::gpX(2), frValue);

  syncAndFreeTempReg(HWReg::gpX(3));
  a.mov(a64::w3, slot);

  // Make sure all FRs can be accessed. Some of them might be in temp regs.
  syncAllFRTempExcept({});
  freeAllFRTempExcept({});

  if (np) {
    EMIT_RUNTIME_CALL(
        *this,
        void (*)(SHRuntime *, SHLegacyValue, SHLegacyValue, uint32_t),
        _sh_ljs_store_np_to_env);
  } else {
    EMIT_RUNTIME_CALL(
        *this,
        void (*)(SHRuntime *, SHLegacyValue, SHLegacyValue, uint32_t),
        _sh_ljs_store_to_env);
  }
}

void Emitter::createClosure(
    FR frRes,
    FR frEnv,
    RuntimeModule *runtimeModule,
    uint32_t functionID) {
  comment(
      "// CreateClosure r%u, r%u, %u",
      frRes.index(),
      frEnv.index(),
      functionID);
  syncAllFRTempExcept(frRes != frEnv ? frRes : FR());
  syncToFrame(frEnv);
  freeAllFRTempExcept({});

  a.mov(a64::x0, xRuntime);
  loadFrameAddr(a64::x1, frEnv);
  loadBits64InGp(a64::x2, (uint64_t)runtimeModule, "RuntimeModule");
  loadBits64InGp(a64::w3, functionID, nullptr);
  EMIT_RUNTIME_CALL(
      *this,
      SHLegacyValue (*)(
          SHRuntime *, const SHLegacyValue *, SHRuntimeModule *, uint32_t),
      _sh_ljs_create_bytecode_closure);

  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::gpX(0));
  movHWFromHW<false>(hwRes, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::createBaseClass(FR frRes, FR frPrototypeOut, FR frEnv) {
  comment(
      "// CreateBaseClass r%u, r%u, r%u",
      frRes.index(),
      frPrototypeOut.index(),
      frEnv.index());
  // TODO: we should also not be syncing frPrototypeOut when possible.
  syncAllFRTempExcept(frRes != frEnv ? frRes : FR());
  syncToFrame(frEnv);
  freeAllFRTempExcept({});

  a.mov(a64::x0, xRuntime);
  // The interpreter expects that the frameRegs it receives starts on the first
  // local register.
  auto ofs = hbc::StackFrameLayout::FirstLocal * sizeof(SHLegacyValue);
  a.add(a64::x1, xFrame, ofs);
  EMIT_RUNTIME_CALL(
      *this, void (*)(SHRuntime *, SHLegacyValue *), _interpreter_create_class);

  // Ensure that the out params have their frame location marked as up-to-date,
  // and any global register is updated.
  syncFrameOutParam(frRes);
  syncFrameOutParam(frPrototypeOut);
}

void Emitter::createDerivedClass(
    FR frRes,
    FR frPrototypeOut,
    FR frEnv,
    FR frSuperClass) {
  comment(
      "// CreateDerivedClass r%u, r%u, r%u r%u",
      frRes.index(),
      frPrototypeOut.index(),
      frEnv.index(),
      frSuperClass.index());
  // TODO: we should also not be syncing frPrototypeOut when possible.
  syncAllFRTempExcept(frRes != frEnv && frRes != frSuperClass ? frRes : FR());
  syncToFrame(frEnv);
  syncToFrame(frSuperClass);
  freeAllFRTempExcept({});

  a.mov(a64::x0, xRuntime);
  // The interpreter expects that the frameRegs it receives starts on the first
  // local register.
  auto ofs = hbc::StackFrameLayout::FirstLocal * sizeof(SHLegacyValue);
  a.add(a64::x1, xFrame, ofs);
  EMIT_RUNTIME_CALL(
      *this, void (*)(SHRuntime *, SHLegacyValue *), _interpreter_create_class);

  // Ensure that the updated frame location is sync'd back.
  syncFrameOutParam(frRes);
  syncFrameOutParam(frPrototypeOut);
}

void Emitter::createGenerator(
    FR frRes,
    FR frEnv,
    RuntimeModule *runtimeModule,
    uint32_t functionID) {
  comment(
      "// CreateGenerator r%u, r%u, %u",
      frRes.index(),
      frEnv.index(),
      functionID);
  syncAllFRTempExcept(frRes != frEnv ? frRes : FR());
  syncToFrame(frEnv);
  freeAllFRTempExcept({});

  a.mov(a64::x0, xRuntime);
  a.mov(a64::x1, xFrame);
  loadFrameAddr(a64::x2, frEnv);
  loadBits64InGp(a64::x3, (uint64_t)runtimeModule, "RuntimeModule");
  a.mov(a64::w4, functionID);
  EMIT_RUNTIME_CALL(
      *this,
      SHLegacyValue (*)(
          SHRuntime *,
          SHLegacyValue *,
          const SHLegacyValue *,
          SHRuntimeModule *,
          uint32_t),
      _interpreter_create_generator);

  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::gpX(0));
  movHWFromHW<false>(hwRes, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::getArgumentsLength(FR frRes, FR frLazyReg) {
  comment("// GetArgumentsLength r%u, r%u", frRes.index(), frLazyReg.index());

  asmjit::Label slowPathLab = newSlowPathLabel();
  asmjit::Label contLab = newContLabel();

  syncAllFRTempExcept(frRes != frLazyReg ? frRes : FR());
  syncToFrame(frLazyReg);

  HWReg hwLazyReg = getOrAllocFRInGpX(frLazyReg, true);
  HWReg hwTemp = allocTempGpX();
  freeAllFRTempExcept({});
  freeReg(hwTemp);
  // Avoid an extra mov by using the temp register for the result if possible.
  HWReg hwRes = getOrAllocFRInVecD(frRes, false);
  frUpdatedWithHW(frRes, hwRes);

  emit_sh_ljs_is_object(a, hwTemp.a64GpX(), hwLazyReg.a64GpX());
  a.b_eq(slowPathLab);

  // Fast path: if it's not an object, read from the frame.
  static_assert(
      HERMESVALUE_VERSION == 2,
      "NativeUint32 is stored as the lower 32 bits of the raw HermesValue");
  a.ldur(
      hwTemp.a64GpX().w(),
      a64::Mem(
          xFrame,
          (int)StackFrameLayout::ArgCount * (int)sizeof(SHLegacyValue)));

  // Encode the uint32_t as a double (making it a HermesValue).
  a.ucvtf(hwRes.a64VecD(), hwTemp.a64GpX().w());

  a.bind(contLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .frRes = frRes,
       .frInput1 = frLazyReg,
       .hwRes = hwRes,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// Slow path: GetArgumentsLength r%u, r%u",
             sl.frRes.index(),
             sl.frInput1.index());
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         em.a.mov(a64::x1, xFrame);
         em.loadFrameAddr(a64::x2, sl.frInput1);
         EMIT_RUNTIME_CALL(
             em,
             SHLegacyValue (*)(SHRuntime *, SHLegacyValue *, SHLegacyValue *),
             _sh_ljs_get_arguments_length);
         em.movHWFromHW<false>(sl.hwRes, HWReg::gpX(0));
         em.a.b(sl.contLab);
       }});
}

void Emitter::debugger() {
  comment("// Debugger");
  if (dumpJitCode_ & DumpJitCode::BRK)
    a.brk(0);
}

void Emitter::iteratorBegin(FR frRes, FR frSource) {
  comment("// IteratorBegin r%u, r%u", frRes.index(), frSource.index());

  syncAllFRTempExcept(frRes != frSource ? frRes : FR());
  syncToFrame(frSource);
  freeAllFRTempExcept({});

  a.mov(a64::x0, xRuntime);
  loadFrameAddr(a64::x1, frSource);
  EMIT_RUNTIME_CALL(
      *this,
      SHLegacyValue (*)(SHRuntime *, SHLegacyValue *),
      _sh_ljs_iterator_begin_rjs);

  syncFrameOutParam(frSource);

  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::gpX(0));
  movHWFromHW<false>(hwRes, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::iteratorNext(FR frRes, FR frIteratorOrIdx, FR frSourceOrNext) {
  comment(
      "// IteratorNext r%u, r%u, r%u",
      frRes.index(),
      frIteratorOrIdx.index(),
      frSourceOrNext.index());

  syncAllFRTempExcept(
      frRes != frIteratorOrIdx && frRes != frSourceOrNext ? frRes : FR());
  syncToFrame(frIteratorOrIdx);
  syncToFrame(frSourceOrNext);
  freeAllFRTempExcept({});

  a.mov(a64::x0, xRuntime);
  loadFrameAddr(a64::x1, frIteratorOrIdx);
  loadFrameAddr(a64::x2, frSourceOrNext);
  EMIT_RUNTIME_CALL(
      *this,
      SHLegacyValue (*)(SHRuntime *, SHLegacyValue *, const SHLegacyValue *),
      _sh_ljs_iterator_next_rjs);

  syncFrameOutParam(frIteratorOrIdx);

  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::gpX(0));
  movHWFromHW<false>(hwRes, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::iteratorClose(FR frIteratorOrIdx, bool ignoreExceptions) {
  comment(
      "// IteratorClose r%u, %u", frIteratorOrIdx.index(), ignoreExceptions);

  syncAllFRTempExcept({});
  syncToFrame(frIteratorOrIdx);
  freeAllFRTempExcept({});

  a.mov(a64::x0, xRuntime);
  loadFrameAddr(a64::x1, frIteratorOrIdx);
  a.mov(a64::w2, ignoreExceptions);
  EMIT_RUNTIME_CALL(
      *this,
      void (*)(SHRuntime *, const SHLegacyValue *, bool),
      _sh_ljs_iterator_close_rjs);
}

void Emitter::throwInst(FR frInput) {
  comment("// Throw r%u", frInput.index());

  // We have to sync registers when the throw is inside a try region
  // because we could read from the FRs again in this function.
  if (isInTry())
    syncAllFRTempExcept({});
  movHWFromFR(HWReg::gpX(1), frInput);
  freeAllFRTempExcept({});

  a.mov(a64::x0, xRuntime);
  EMIT_RUNTIME_CALL(*this, void (*)(SHRuntime *, SHLegacyValue), _sh_throw);
}

void Emitter::throwIfEmptyUndefinedImpl(FR frRes, FR frInput, bool empty) {
  comment(
      "// %s r%u, r%u",
      empty ? "ThrowIfEmpty" : "ThrowIfUndefined",
      frRes.index(),
      frInput.index());

  asmjit::Label slowPathLab = newSlowPathLabel();

  // We have to sync registers when the throw is inside a try region
  // because we could read from the FRs again in this function.
  if (isInTry())
    syncAllFRTempExcept(frRes != frInput ? frRes : FR());
  HWReg hwInput = getOrAllocFRInGpX(frInput, true);
  HWReg hwTemp = allocTempGpX();
  if (isInTry())
    freeAllFRTempExcept({});
  freeReg(hwTemp);

  if (empty)
    emit_sh_ljs_is_empty(a, hwTemp.a64GpX(), hwInput.a64GpX());
  else
    emit_sh_ljs_is_undefined(a, hwTemp.a64GpX(), hwInput.a64GpX());
  a.b_eq(slowPathLab);

  HWReg hwRes = getOrAllocFRInGpX(frRes, false);
  movHWFromHW<false>(hwRes, hwInput);
  frUpdatedWithHW(frRes, hwRes);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .name = empty ? "ThrowIfEmpty" : "ThrowIfUndefined",
       .frRes = frRes,
       .frInput1 = frInput,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// Slow path: %s r%u, r%u",
             sl.name,
             sl.frRes.index(),
             sl.frInput1.index());
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         EMIT_RUNTIME_CALL(em, void (*)(SHRuntime *), _sh_throw_empty);
         // Call does not return.
       }});
}

void Emitter::throwIfThisInitialized(FR frInput) {
  comment("// ThrowIfThisInitialized r%u", frInput.index());

  asmjit::Label slowPathLab = newSlowPathLabel();

  // We have to sync registers when the throw is inside a try region
  // because we could read from the FRs again in this function.
  // Outside a try it's not observable behavior.
  // Note that only the sync is needed, not a free. A free is required when a
  // call is emitted on a path that continues in this basic block, since temps
  // are caller-saved. Here the only call is the non-returning throw in the
  // out-of-line slow path; the fall-through path calls nothing, and the catch
  // handler begins a new basic block, which re-normalizes temp state anyway.
  // Freeing would only force needless reloads for the rest of the block.
  //
  // Cf. fastArrayLoad's #else (non-compressed-pointer) path, which syncs
  // without freeing for the same reason. Its
  // HERMESVM_COMPRESSED_POINTERS/HERMESVM_BOXED_DOUBLES path is not
  // comparable: it emits an unconditional runtime call, so it must sync and
  // free regardless of isInTry(). Cf. also throwInst, which syncs only under
  // isInTry() but frees unconditionally, because it emits its call inline.
  if (isInTry())
    syncAllFRTempExcept({});
  HWReg hwInput = getOrAllocFRInGpX(frInput, true);
  HWReg hwTemp = allocTempGpX();
  freeReg(hwTemp);

  emit_sh_ljs_is_empty(a, hwTemp.a64GpX(), hwInput.a64GpX());
  a.b_ne(slowPathLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .frInput1 = frInput,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// Slow path: ThrowIfThisInitialized r%u", sl.frInput1.index());
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         EMIT_RUNTIME_CALL(
             em, void (*)(SHRuntime *), _sh_throw_this_already_initialized);
         // Call does not return.
       }});
}

void Emitter::createRegExp(
    FR frRes,
    SHSymbolID patternID,
    SHSymbolID flagsID,
    uint32_t regexpID) {
  comment("// CreateRegExp r%u, %u, %u", frRes.index(), patternID, flagsID);

  syncAllFRTempExcept(frRes);
  freeAllFRTempExcept({});

  a.mov(a64::x0, xRuntime);
  loadBits64InGp(a64::x1, (uint64_t)codeBlock_, "CodeBlock");
  a.mov(a64::w2, patternID);
  a.mov(a64::w3, flagsID);
  a.mov(a64::w4, regexpID);
  EMIT_RUNTIME_CALL(
      *this,
      SHLegacyValue (*)(
          SHRuntime *, SHCodeBlock *, uint32_t, uint32_t, uint32_t),
      _interpreter_create_regexp);

  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::gpX(0));
  movHWFromHW<false>(hwRes, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::jmpTypeOfIs(
    const asmjit::Label &target,
    FR frInput,
    TypeOfIsTypes origTypes) {
  comment("// jTypeOfIs r%u, %u", frInput.index(), origTypes.getRaw());

  TypeOfIsTypes invertedTypes = origTypes.invert();

  // Do this always because it's the end of a basic block.
  // The freeAllFRTempExcept calls are within fast paths because we may want to
  // use FR temps to syncToFrame(frInput) in the call path, and we know at JIT
  // time whether we'll emit the call path.
  syncAllFRTempExcept({});

  HWReg hwInput = getOrAllocFRInGpX(frInput, true);
  HWReg hwTemp = allocTempGpX();
  freeReg(hwTemp);
  freeAllFRTempExcept({});

  auto xInput = hwInput.a64GpX();
  auto xTemp = hwTemp.a64GpX();
  auto wTemp = xTemp.w();

  // Try and see if inverting will result in fewer checks.
  // If so, flip it and set invert=true.
  bool invert = false;
  TypeOfIsTypes typesToCheck = origTypes;
  if (invertedTypes.count() < origTypes.count()) {
    invert = true;
    typesToCheck = invertedTypes;
  }

  // Nothing left to check means the answer does not depend on the input: an
  // empty origTypes matches nothing, so falling through is already right,
  // while an all-bits origTypes inverts to empty and matches everything, so
  // the branch is unconditional. None of the checks below are emitted.
  if (typesToCheck.count() == 0 && invert)
    a.b(target);

  // doneLab goes at the end of the instruction if there's multiple bits to
  // check, allowing short-circuiting the remaining checks if one of the
  // TypeOfIsTypes bits matches the kind of the input.
  // Use numRemainingTypes to track how many bits are left to check.
  asmjit::Label doneLab = a.newLabel();
  size_t numRemainingTypes = typesToCheck.count();

  // Checks are done as follows:
  // * If not inverted, just go to the target if the tag matches the bit,
  //   else fallthrough to the next case (if any).
  // * If inverted and there's multiple bits remaining,
  //   if the tag matches the bit, short circuit to doneLab and we've
  //   finished executing the instruction (no need to check the other bits).
  // * If inverted and there's only one bit remaining,
  //   then if the tag does NOT match the bit, go to the target
  //   immediately.
  //
  // In this way, single-bit checks (both inverted and not) are fast,
  // and multiple-bit checks are correct.
  // It's possible more complexity can optimize this further if needed, but this
  // is not a bad start.

  /// Emit the simple check for a match.
  /// If we're not inverted, branch to the target based on cond.
  /// If we're inverted:
  ///   If there's bits remaining to check, branch to doneLab if the tag matches
  ///   because we can short circuit the rest of the checks.
  ///   If there's no bits remaining to check, branch to the target if the tag
  ///   does NOT match the bit.
  /// \param cond the condition code, which if true, indicates a tag match.
  auto emitCondCheck = [this, invert, &numRemainingTypes, &target, &doneLab](
                           a64::CondCode cond) {
    if (!invert)
      a.b(cond, target);
    else if (numRemainingTypes > 0)
      a.b(cond, doneLab);
    else
      a.b(a64::negateCond(cond), target);
  };

  if (typesToCheck.hasUndefined()) {
    --numRemainingTypes;
    emit_sh_ljs_is_undefined(a, xTemp, xInput);
    emitCondCheck(a64::CondCode::kEQ);
  }
  if (typesToCheck.hasSymbol()) {
    --numRemainingTypes;
    emit_sh_ljs_is_symbol(a, xTemp, xInput);
    emitCondCheck(a64::CondCode::kEQ);
  }
  if (typesToCheck.hasString()) {
    --numRemainingTypes;
    emit_sh_ljs_is_string(a, xTemp, xInput);
    emitCondCheck(a64::CondCode::kEQ);
  }
  if (typesToCheck.hasBoolean()) {
    --numRemainingTypes;
    emit_sh_ljs_is_bool(a, xTemp, xInput);
    emitCondCheck(a64::CondCode::kEQ);
  }
  if (typesToCheck.hasNull()) {
    --numRemainingTypes;
    emit_sh_ljs_is_null(a, xTemp, xInput);
    emitCondCheck(a64::CondCode::kEQ);
  }
  if (typesToCheck.hasBigint()) {
    --numRemainingTypes;
    emit_sh_ljs_is_bigint(a, xTemp, xInput);
    emitCondCheck(a64::CondCode::kEQ);
  }
  if (typesToCheck.hasNumber()) {
    --numRemainingTypes;
    static_assert(
        HERMESVALUE_VERSION == 2,
        "HVTag_First must be the first after double limit");
    loadBits64InGp(
        xTemp, ((uint64_t)HVTag_First << kHV_NumDataBits), "doubleLim");
    a.cmp(xInput, xTemp);
    emitCondCheck(a64::CondCode::kLO);
  }
  // TODO: Special-case if both hasObject() and hasFunction() are set,
  // because we no longer would need to check the CellKind.
  if (typesToCheck.hasObject()) {
    --numRemainingTypes;
    asmjit::Label objectDoneLab = a.newLabel();
    emit_sh_ljs_is_object(a, xTemp, xInput);
    if (!invert)
      a.b_ne(objectDoneLab);
    else if (numRemainingTypes > 0)
      a.b_ne(objectDoneLab);
    else
      a.b_ne(target);
    emit_sh_ljs_get_pointer(a, hwTemp.a64GpX(), hwInput.a64GpX());
    emit_gccell_get_kind(a, xTemp, xTemp);
    emit_cellkind_in_range(
        a,
        wTemp,
        wTemp,
        CellKind::CallableKind_first,
        CellKind::CallableKind_last);
    emitCondCheck(a64::CondCode::kHI);
    a.bind(objectDoneLab);
  }
  if (typesToCheck.hasFunction()) {
    --numRemainingTypes;
    asmjit::Label functionDoneLab = a.newLabel();
    emit_sh_ljs_is_object(a, xTemp, xInput);
    if (!invert)
      a.b_ne(functionDoneLab);
    else if (numRemainingTypes > 0)
      a.b_ne(functionDoneLab);
    else
      a.b_ne(target);
    emit_sh_ljs_get_pointer(a, xTemp, xInput);
    emit_gccell_get_kind(a, xTemp, xTemp);
    emit_cellkind_in_range(
        a,
        wTemp,
        wTemp,
        CellKind::CallableKind_first,
        CellKind::CallableKind_last);
    emitCondCheck(a64::CondCode::kLS);
    a.bind(functionDoneLab);
  }

  assert(numRemainingTypes == 0 && "missed a type");

  // Put doneLab after, so we skip the branch if we directly branch to doneLab
  // from above.
  a.bind(doneLab);
}

void Emitter::typeOfIs(FR frRes, FR frInput, TypeOfIsTypes origTypes) {
  comment(
      "// typeOfIs r%u, r%u, %u",
      frRes.index(),
      frInput.index(),
      origTypes.getRaw());

  // Store the input in hwInputTemp for the duration of the instruction.
  // Needed because it's possible frRes == frInput, and we want to write to
  // frRes at the top of the instruction.
  HWReg hwInputTemp;
  if (frRes == frInput) {
    hwInputTemp = allocTempGpX();
    movHWFromFR(hwInputTemp, frInput);
  } else {
    hwInputTemp = getOrAllocFRInGpX(frInput, true);
  }
  HWReg hwTemp = allocTempGpX();
  HWReg hwRes = getOrAllocFRInGpX(frRes, false);
  frUpdatedWithHW(frRes, hwRes);
  freeReg(hwTemp);
  if (frRes == frInput) {
    freeReg(hwInputTemp);
  }

  auto xInputTemp = hwInputTemp.a64GpX();
  auto xTemp = hwTemp.a64GpX();
  auto wTemp = xTemp.w();
  auto xRes = hwRes.a64GpX();

  TypeOfIsTypes invertedTypes = origTypes.invert();

  // Try and see if inverting will result in fewer checks.
  // If so, flip it and set invert=true.
  bool invert = false;
  TypeOfIsTypes typesToCheck = origTypes;
  if (invertedTypes.count() < origTypes.count()) {
    invert = true;
    typesToCheck = invertedTypes;
  }

  // Nothing left to check means the answer does not depend on the input: an
  // empty origTypes matches nothing, and an all-bits origTypes inverts to
  // empty and matches everything. Either way the result is the same constant
  // the individual cases produce when no tag can match, and none of the checks
  // below are emitted.
  if (typesToCheck.count() == 0)
    a.mov(xRes, invert ? 1 : 0);

  // matchLab goes directly to the end of the instruction if there are multiple
  // bits to check, allowing short-circuiting the remaining checks if one of the
  // TypeOfIsTypes bits matches the kind of the input.
  // If there's only one bit to check, we don't put extra code the end - none of
  // the other cases will be emitted.
  asmjit::Label matchLab{};
  if (typesToCheck.count() > 1)
    matchLab = a.newLabel();

  // First, initialize xRes if necessary:
  // * If there are multiple bits set, initialize it to the value we would
  //   produce on a match. This is false if inverted and true otherwise.
  // * If there's only one bit set, leave it uninitialized, since we will
  //   overwrite the value in the individual cases with cset.
  //
  // Checks are done as follows:
  // * If there are multiple bits set, then matchLab is valid,
  //   so if the tag matches the bit, branch to matchLab.
  //   If the tag doesn't match, then fall through to the next check.
  // * If there's only one bit set, then matchLab is NOT valid,
  //   so emit cset with the appropriate condition code and we're done.
  //
  // In this way, single-bit checks (both inverted and not) are fast,
  // and multiple-bit checks are correct.

  /// Emit the simple check for a match.
  /// If there's multiple bits to check, this will branch based on \p cond
  /// to matchLab if the tag matches.
  /// If there's only one bit to check, this will emit a cinc with the
  /// appropriate condition code (and we're done).
  /// \param cond the condition code, which if true, indicates a tag match.
  auto emitCondCheck = [this, invert, &xRes, &matchLab](a64::CondCode cond) {
    if (matchLab.isValid())
      a.b(cond, matchLab);
    else
      a.cset(xRes, !invert ? cond : a64::negateCond(cond));
  };

  // As described above, if there are multiple cases, initialize it to the value
  // it should have on a successful match.
  if (matchLab.isValid())
    a.mov(xRes, invert ? 0 : 1);

  if (typesToCheck.hasUndefined()) {
    emit_sh_ljs_is_undefined(a, xTemp, xInputTemp);
    emitCondCheck(a64::CondCode::kEQ);
  }
  if (typesToCheck.hasSymbol()) {
    emit_sh_ljs_is_symbol(a, xTemp, xInputTemp);
    emitCondCheck(a64::CondCode::kEQ);
  }
  if (typesToCheck.hasString()) {
    emit_sh_ljs_is_string(a, xTemp, xInputTemp);
    emitCondCheck(a64::CondCode::kEQ);
  }
  if (typesToCheck.hasBoolean()) {
    emit_sh_ljs_is_bool(a, xTemp, xInputTemp);
    emitCondCheck(a64::CondCode::kEQ);
  }
  if (typesToCheck.hasBigint()) {
    emit_sh_ljs_is_bigint(a, xTemp, xInputTemp);
    emitCondCheck(a64::CondCode::kEQ);
  }
  if (typesToCheck.hasNull()) {
    emit_sh_ljs_is_null(a, xTemp, xInputTemp);
    emitCondCheck(a64::CondCode::kEQ);
  }
  if (typesToCheck.hasNumber()) {
    static_assert(
        HERMESVALUE_VERSION == 2,
        "HVTag_First must be the first after double limit");
    loadBits64InGp(
        xTemp, ((uint64_t)HVTag_First << kHV_NumDataBits), "doubleLim");
    a.cmp(xInputTemp, xTemp);
    emitCondCheck(a64::CondCode::kLO);
  }
  if (typesToCheck.hasObject()) {
    asmjit::Label objectDoneLab = a.newLabel();
    emit_sh_ljs_is_object(a, xTemp, xInputTemp);
    if (matchLab.isValid()) {
      // If the tag did NOT match, we can't run anything else in this case.
      // We must branch, b_ne and proceed to try matching any other cases.
      a.b_ne(objectDoneLab);
    } else {
      // No more tags to check. Decide the result here and go to the end.
      a.mov(xRes, invert ? 1 : 0);
      a.b_ne(objectDoneLab);
    }
    emit_sh_ljs_get_pointer(a, xTemp, xInputTemp);
    emit_gccell_get_kind(a, xTemp, xTemp);
    emit_cellkind_in_range(
        a,
        wTemp,
        wTemp,
        CellKind::CallableKind_first,
        CellKind::CallableKind_last);
    emitCondCheck(a64::CondCode::kHI);
    a.bind(objectDoneLab);
  }
  if (typesToCheck.hasFunction()) {
    asmjit::Label functionDoneLab = a.newLabel();
    emit_sh_ljs_is_object(a, xTemp, xInputTemp);
    if (matchLab.isValid()) {
      // If the tag did NOT match, we can't run anything else in this case.
      // We must branch, b_ne and proceed to try matching any other cases.
      a.b_ne(functionDoneLab);
    } else {
      // No more tags to check. Decide the result here and go to the end.
      a.mov(xRes, invert ? 1 : 0);
      a.b_ne(functionDoneLab);
    }
    emit_sh_ljs_get_pointer(a, xTemp, xInputTemp);
    emit_gccell_get_kind(a, xTemp, xTemp);
    emit_cellkind_in_range(
        a,
        wTemp,
        wTemp,
        CellKind::CallableKind_first,
        CellKind::CallableKind_last);
    emitCondCheck(a64::CondCode::kLS);
    a.bind(functionDoneLab);
  }

  if (matchLab.isValid()) {
    // We failed to match, so flip the result
    a.eor(xRes, xRes, 1);
    // We initialize xRes to the "match value", so there is nothing to do on a
    // match.
    a.bind(matchLab);
  }

  // xRes contains either 0 or 1 at this point, turn it into a bool HermesValue.
  emit_sh_ljs_bool(a, xRes);
}

void Emitter::uintSwitchImm(
    FR frInput,
    const asmjit::Label &defaultLabel,
    llvh::ArrayRef<const asmjit::Label *> labels,
    uint32_t minVal,
    uint32_t maxVal) {
  comment(
      "// uintSwitchImm r%u, min %u, max %u", frInput.index(), minVal, maxVal);

  asmjit::Error err;

  // End of the basic block.
  syncAllFRTempExcept({});

  // Load the input value into a double register to check if it's an int.
  HWReg hwInput = getOrAllocFRInVecD(frInput, true);

  HWReg hwTempInput = allocTempGpX();
  HWReg hwTempTarget = allocTempGpX();
  HWReg hwTempD = allocTempVecD();
  freeReg(hwTempInput);
  freeReg(hwTempTarget);
  freeReg(hwTempD);

  a64::VecD dInput = hwInput.a64VecD();
  a64::GpW wTempInput = hwTempInput.a64GpX().w();

  // Convert the input to an integer and back to double,
  // and check if the value remained the same.
  // If it didn't, jump to the default label.
  emit_double_is_uint32(a, wTempInput, hwTempD.a64VecD(), dInput);
  a.b_ne(defaultLabel);

  // Check if the integer value in xTemp is in range.
  // First check minVal.
  EXPECT_ERROR(asmjit::kErrorInvalidImmediate, err = a.cmp(wTempInput, minVal));
  if (err) {
    a.mov(hwTempTarget.a64GpX().w(), minVal);
    a.cmp(wTempInput, hwTempTarget.a64GpX().w());
  }
  // If the value is lower than minVal, jump to the default label.
  a.b_lo(defaultLabel);

  // Now check maxVal.
  EXPECT_ERROR(asmjit::kErrorInvalidImmediate, err = a.cmp(wTempInput, maxVal));
  if (err) {
    a.mov(hwTempTarget.a64GpX().w(), maxVal);
    a.cmp(wTempInput, hwTempTarget.a64GpX().w());
  }
  // If the value is higher than maxVal, jump to the default label.
  a.b_hi(defaultLabel);

  // Compute the offset into the jump table, dereference, and jump.
  // Offset by the minVal if necessary.
  if (minVal != 0) {
    EXPECT_ERROR(
        asmjit::kErrorInvalidImmediate,
        err = a.sub(wTempInput, wTempInput, minVal));
    if (err) {
      a.mov(hwTempTarget.a64GpX().w(), minVal);
      a.sub(wTempInput, wTempInput, hwTempTarget.a64GpX().w());
    }
  }

  // Label for the start of the jump table and the base of the br instruction
  // that actually executes the switch.
  // Used for both purposes due to placement of the jump table directly after
  // the br.
  asmjit::Label tableLab = a.newLabel();

  // wTempInput contains the index into the jump table.
  a64::GpX xTempTarget = hwTempTarget.a64GpX();
  // Load the jump offset into wTempInput by using adr to find the address of
  // the table and then reading 4 bytes from an offset of wTempInput bytes.
  a.adr(xTempTarget, tableLab);
  // Left shift 2 to get the byte offset into the table.
  a.ldr(
      wTempInput,
      a64::Mem(xTempTarget, wTempInput, a64::Shift(a64::ShiftOp::kLSL, 2)));
  // Add the jump offset to the base of the table to get the target address.
  a.add(xTempTarget, xTempTarget, wTempInput.x(), a64::sxtw(0));
  // Branch to the target address.
  a.br(xTempTarget);

  // Emit the jump table.
  // NOTE: The jump table is emitted immediately after the br instruction that
  // uses it.
  a.bind(tableLab);
  for (const asmjit::Label *label : labels) {
    a.embedLabelDelta(*label, tableLab, /* size */ 4);
  }

  // Do this always, since this could be the end of the BB.
  freeAllFRTempExcept({});
}

void Emitter::stringSwitchImm(
    FR frInput,
    RuntimeModule *runtimeModule,
    uint32_t tableIndex,
    const asmjit::Label &defaultLabel,
    llvh::ArrayRef<StringSwitchCase> cases) {
  comment("// stringSwitchImm r%u, size %zu", frInput.index(), cases.size());

  // End of the basic block.
  syncAllFRTempExcept({});
  // The handler reads the value through the frame address passed below, so
  // the slot has to hold the current value.
  syncToFrame(frInput);

  a.mov(a64::x0, (uint64_t)runtimeModule);
  a.mov(a64::w1, tableIndex);
  loadFrameAddr(a64::x2, frInput);

  EMIT_RUNTIME_CALL_WITHOUT_SAVED_IP(
      *this,
      void *(*)(RuntimeModule *, uint32_t, SHLegacyValue *),
      _jit_string_switch_imm_table_lookup);

  a.cbz(a64::x0, defaultLabel);
  // Otherwise, branch to the address that was returned.
  a.br(a64::x0);

  // Do this always, since this could be the end of the BB.
  freeAllFRTempExcept({});
}

asmjit::Label Emitter::newPrefLabel(const char *pref, size_t index) {
  char buf[16];
  snprintf(buf, sizeof(buf), "%s%lu", pref, index);
  return a.newNamedLabel(buf);
}

int32_t Emitter::reserveData(
    int32_t dsize,
    size_t align,
    asmjit::TypeId typeId,
    int32_t itemCount,
    const char *comment) {
  // Align the new data.
  size_t oldSize = roData_.size();
  size_t dataOfs = (roData_.size() + align - 1) & ~(align - 1);
  if (dataOfs >= INT32_MAX)
    hermes::hermes_fatal("JIT RO data overflow");
  // Grow to include the data.
  roData_.resize(dataOfs + dsize);

  // If logging is enabled, generate data descriptors.
  if (hasLogger()) {
    // Optional padding descriptor.
    if (dataOfs != oldSize) {
      int32_t gap = (int32_t)(dataOfs - oldSize);
      roDataDesc_.push_back(
          {.size = gap, .typeId = asmjit::TypeId::kUInt8, .itemCount = gap});
    }

    roDataDesc_.push_back(
        {.size = dsize,
         .typeId = typeId,
         .itemCount = itemCount,
         .comment = comment});
  }

  return (int32_t)dataOfs;
}

/// Register a 64-bit constant in RO DATA and return its offset.
int32_t Emitter::uint64Const(uint64_t bits, const char *comment) {
  auto [it, inserted] = fp64ConstMap_.try_emplace(bits, 0);
  if (inserted) {
    int32_t dataOfs = reserveData(
        sizeof(double), sizeof(double), asmjit::TypeId::kFloat64, 1, comment);
    memcpy(roData_.data() + dataOfs, &bits, sizeof(double));
    it->second = dataOfs;
  }
  return it->second;
}

asmjit::Label Emitter::registerThunk(void *fn, const char *name) {
  auto [it, inserted] = thunkMap_.try_emplace(fn, 0);
  // Is this a new thunk?
  if (inserted) {
    it->second = thunks_.size();
    int32_t dataOfs =
        reserveData(sizeof(fn), sizeof(fn), asmjit::TypeId::kUInt64, 1, name);
    memcpy(roData_.data() + dataOfs, &fn, sizeof(fn));
    thunks_.emplace_back(name ? a.newNamedLabel(name) : a.newLabel(), dataOfs);
  }

  return thunks_[it->second].first;
}

void Emitter::emitCatchTable(
    llvh::ArrayRef<const asmjit::Label *> exceptionHandlers) {
  // No trys in the function, nothing to do here.
  if (!catchTableLabel_.isValid())
    return;

  a.bind(catchTableLabel_);

  asmjit::Label addressTableLab = a.newLabel();

  // Find the catch target for the exception.
  a.mov(a64::x0, xRuntime);
  loadBits64InGp(a64::x1, (uint64_t)codeBlock_, "CodeBlock");
  a.mov(a64::x2, xFrame);
  a.add(a64::x3, a64::sp, getJmpBufOffset());
  a.ldr(a64::x4, a64::Mem(a64::sp, getSavedSHLocalsOffset()));
  a.adr(a64::x5, addressTableLab);
  EMIT_RUNTIME_CALL_WITHOUT_THUNK_AND_SAVED_IP(
      *this,
      void *(*)(SHRuntime *,
                SHCodeBlock *,
                SHLegacyValue *,
                SHJmpBuf *,
                SHLocals *,
                int32_t *),
      _jit_find_catch_target);

  // The address to branch to was returned here.
  a.br(a64::x0);

  // Table of offsets from addressTableLab to jump to.
  a.bind(addressTableLab);
  for (const asmjit::Label *handler : exceptionHandlers) {
    a.embedLabelDelta(*handler, addressTableLab, /* size */ 4);
  }
}

void Emitter::emitSlowPaths() {
  while (!slowPaths_.empty()) {
    SlowPath &sp = slowPaths_.front();
    emittingIP = sp.emittingIP;
    sp.emit(*this, sp);
    slowPaths_.pop_front();
  }
  emittingIP = nullptr;
}

void Emitter::emitThunks() {
  comment("// Thunks");
  for (const auto &th : thunks_) {
    a.bind(th.first);
    a.ldr(xScratch, a64::Mem(roDataLabel_, th.second));
    a.br(xScratch);
  }
}

void Emitter::emitROData() {
  a.bind(roDataLabel_);
  if (!hasLogger()) {
    a.embed(roData_.data(), roData_.size());
  } else {
    int32_t ofs = 0;
    for (const auto &desc : roDataDesc_) {
      if (desc.comment)
        comment("// %s", desc.comment);
      a.embedDataArray(desc.typeId, roData_.data() + ofs, desc.itemCount);
      ofs += desc.size;
    }
  }
}

void Emitter::arithUnop(
    bool forceNumber,
    FR frRes,
    FR frInput,
    const char *name,
    void (*fast)(
        a64::Assembler &a,
        const a64::VecD &dst,
        const a64::VecD &src,
        const a64::VecD &tmp),
    void *slowCall,
    const char *slowCallName) {
  comment("// %s r%u, r%u", name, frRes.index(), frInput.index());

  HWReg hwRes, hwInput;
  asmjit::Label slowPathLab;
  asmjit::Label contLab;
  bool inputIsNum;

  if (forceNumber) {
    frameRegs_[frInput.index()].localType = FRType::Number;
    inputIsNum = true;
  } else {
    inputIsNum = isFRKnownNumber(frInput);
  }

  hwInput = getOrAllocFRInVecD(frInput, true);
  if (!inputIsNum) {
    slowPathLab = newSlowPathLabel();
    contLab = newContLabel();
    syncAllFRTempExcept(frRes != frInput ? frRes : FR());
    syncToFrame(frInput);

    // Since HermesValue is NaN-boxed we know that all non-number values will be
    // NaN. So we can conveniently test for non-number values by checking for
    // NaN (which does not compare equal to itself).
    static_assert(HERMESVALUE_VERSION == 2, "Non-numbers must be NaN");
    a.fcmp(hwInput.a64VecD(), hwInput.a64VecD());
    a.b_ne(slowPathLab);
  }

  hwRes = getOrAllocFRInVecD(frRes, false);
  HWReg hwTmp = hwRes != hwInput ? hwRes : allocTempVecD();
  fast(a, hwRes.a64VecD(), hwInput.a64VecD(), hwTmp.a64VecD());
  if (hwRes == hwInput)
    freeReg(hwTmp);

  frUpdatedWithHW(
      frRes, hwRes, inputIsNum ? FRType::Number : FRType::UnknownPtr);

  if (inputIsNum)
    return;

  freeAllFRTempExcept(frRes);
  a.bind(contLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .name = name,
       .frRes = frRes,
       .frInput1 = frInput,
       .hwRes = hwRes,
       .slowCall = slowCall,
       .slowCallName = slowCallName,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// Slow path: %s r%u, r%u",
             sl.name,
             sl.frRes.index(),
             sl.frInput1.index());
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         em.loadFrameAddr(a64::x1, sl.frInput1);
         em.callThunkWithSavedIP(sl.slowCall, sl.slowCallName);
         em.movHWFromHW<false>(sl.hwRes, HWReg::gpX(0));
         em.a.b(sl.contLab);
       }});
}

void Emitter::booleanNot(FR frRes, FR frInput) {
  comment("// Not r%u, r%u", frRes.index(), frInput.index());

  // TODO: Add a fast path, perhaps by sharing some code with JmpTrue.
  syncAndFreeTempReg(HWReg::gpX(0));
  movHWFromFR(HWReg::gpX(0), frInput);

  // Since we already loaded the input, no need to check for frRes == frInput.
  syncAllFRTempExcept(frRes);
  freeAllFRTempExcept({});
  EMIT_RUNTIME_CALL(*this, bool (*)(SHLegacyValue), _sh_ljs_to_boolean);

  HWReg hwRes = getOrAllocFRInGpX(frRes, false);
  // Negate the result.
  a.eor(hwRes.a64GpX(), a64::x0, 1);
  // Add the bool tag.
  emit_sh_ljs_bool(a, hwRes.a64GpX());
  frUpdatedWithHW(frRes, hwRes, FRType::Bool);
}

void Emitter::bitNot(FR frRes, FR frInput) {
  comment("// BitNot r%u, r%u", frRes.index(), frInput.index());

  HWReg hwTempGpX = allocTempGpX();
  HWReg hwTempVecD = allocTempVecD();

  syncAllFRTempExcept(frRes != frInput ? frRes : FR());
  // TODO: As with binary bit ops, it should be possible to only do this in the
  // slow path.
  syncToFrame(frInput);

  asmjit::Label slowPathLab = newSlowPathLabel();
  asmjit::Label contLab = newContLabel();

  HWReg hwInput = getOrAllocFRInVecD(frInput, true);
  emit_double_is_int(
      a, hwTempGpX.a64GpX(), hwTempVecD.a64VecD(), hwInput.a64VecD());
  a.b_ne(slowPathLab);

  // Done allocating registers. Free them all and allocate the result.
  freeAllFRTempExcept({});
  freeReg(hwTempGpX);
  freeReg(hwTempVecD);
  HWReg hwRes = getOrAllocFRInVecD(frRes, false);
  frUpdatedWithHW(
      frRes,
      hwRes,
      isFRKnownType(frInput, FRType::Number) ? FRType::Number
                                             : FRType::UnknownPtr);

  // Perform the negation and write it to the result.
  a.mvn(hwTempGpX.a64GpX().w(), hwTempGpX.a64GpX().w());
  a.scvtf(hwRes.a64VecD(), hwTempGpX.a64GpX().w());

  a.bind(contLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .frRes = frRes,
       .frInput1 = frInput,
       .hwRes = hwRes,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// Slow path: bitNot r%u, r%u",
             sl.frRes.index(),
             sl.frInput1.index());
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         em.loadFrameAddr(a64::x1, sl.frInput1);
         EMIT_RUNTIME_CALL(
             em,
             SHLegacyValue (*)(SHRuntime *, const SHLegacyValue *),
             _sh_ljs_bit_not_rjs);
         em.movHWFromHW<false>(sl.hwRes, HWReg::gpX(0));
         em.a.b(sl.contLab);
       }});
}

void Emitter::typeOf(FR frRes, FR frInput) {
  comment("// TypeOf r%u, r%u", frRes.index(), frInput.index());
  syncAllFRTempExcept(frRes == frInput ? FR() : frRes);
  syncToFrame(frInput);
  freeAllFRTempExcept(FR());

  a.mov(a64::x0, xRuntime);
  loadFrameAddr(a64::x1, frInput);
  // TODO: Use a function that preserves temporary registers.
  EMIT_RUNTIME_CALL(
      *this, SHLegacyValue (*)(SHRuntime *, SHLegacyValue *), _sh_ljs_typeof);

  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::gpX(0));
  movHWFromHW<false>(hwRes, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::getPNameList(FR frRes, FR frObj, FR frIdx, FR frSize) {
  comment(
      "// GetPNameList r%u, r%u, r%u, r%u",
      frRes.index(),
      frObj.index(),
      frIdx.index(),
      frSize.index());
  syncAllFRTempExcept({});
  // We have to sync frObj to the frame since it is an in/out parameter.
  syncToFrame(frObj);
  // No need to sync frIdx and frSize since they are just out parameters.
  freeAllFRTempExcept({});
  a.mov(a64::x0, xRuntime);
  loadFrameAddr(a64::x1, frObj);
  loadFrameAddr(a64::x2, frIdx);
  loadFrameAddr(a64::x3, frSize);
  EMIT_RUNTIME_CALL(
      *this,
      SHLegacyValue (*)(
          SHRuntime *, SHLegacyValue *, SHLegacyValue *, SHLegacyValue *),
      _sh_ljs_get_pname_list_rjs);

  // Ensure that the out params have their frame location marked as up-to-date,
  // and any global register is updated.
  syncFrameOutParam(frObj);
  syncFrameOutParam(frIdx);
  syncFrameOutParam(frSize);

  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::gpX(0));
  movHWFromHW<false>(hwRes, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::getNextPName(
    FR frRes,
    FR frProps,
    FR frObj,
    FR frIdx,
    FR frSize) {
  comment(
      "// GetNextPName r%u, r%u, r%u, r%u, r%u",
      frRes.index(),
      frProps.index(),
      frObj.index(),
      frIdx.index(),
      frSize.index());

  syncAllFRTempExcept({});
  syncToFrame(frProps);
  syncToFrame(frObj);
  syncToFrame(frIdx);
  syncToFrame(frSize);
  freeAllFRTempExcept({});
  a.mov(a64::x0, xRuntime);
  loadFrameAddr(a64::x1, frProps);
  loadFrameAddr(a64::x2, frObj);
  loadFrameAddr(a64::x3, frIdx);
  loadFrameAddr(a64::x4, frSize);
  EMIT_RUNTIME_CALL(
      *this,
      SHLegacyValue (*)(
          SHRuntime *,
          SHLegacyValue *,
          SHLegacyValue *,
          SHLegacyValue *,
          SHLegacyValue *),
      _sh_ljs_get_next_pname_rjs);

  // Ensure that the updated frame location is sync'd back to any global reg.
  syncFrameOutParam(frIdx);

  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::gpX(0));
  movHWFromHW<false>(hwRes, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::toPropertyKey(FR frRes, FR frVal) {
  comment("// ToPropertyKey r%u, r%u", frRes.index(), frVal.index());
  syncAllFRTempExcept(frRes != frVal ? frRes : FR());
  syncToFrame(frVal);
  freeAllFRTempExcept({});

  a.mov(a64::x0, xRuntime);
  loadFrameAddr(a64::x1, frVal);
  EMIT_RUNTIME_CALL(
      *this,
      SHLegacyValue (*)(SHRuntime *, const SHLegacyValue *),
      _sh_ljs_to_property_key);

  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::gpX(0));
  movHWFromHW<false>(hwRes, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::addS(FR frRes, FR frLeft, FR frRight) {
  comment(
      "// AddS r%u, r%u, r%u", frRes.index(), frLeft.index(), frRight.index());

  syncAllFRTempExcept(frRes != frLeft && frRes != frRight ? frRes : FR());
  syncToFrame(frLeft);
  syncToFrame(frRight);
  freeAllFRTempExcept({});

  a.mov(a64::x0, xRuntime);
  loadFrameAddr(a64::x1, frLeft);
  loadFrameAddr(a64::x2, frRight);
  EMIT_RUNTIME_CALL(
      *this,
      SHLegacyValue (*)(SHRuntime *, SHLegacyValue *, SHLegacyValue *),
      _sh_ljs_string_add);
  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::gpX(0));
  movHWFromHW<false>(hwRes, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes);
}

void Emitter::mod(bool forceNumber, FR frRes, FR frLeft, FR frRight) {
  comment(
      "// %s%s r%u, r%u, r%u",
      "mod",
      forceNumber ? "N" : "",
      frRes.index(),
      frLeft.index(),
      frRight.index());
  HWReg hwRes, hwLeft, hwRight;
  asmjit::Label slowPathLab;
  asmjit::Label contLab;
  bool leftIsNum, rightIsNum, slow;

  if (forceNumber) {
    frameRegs_[frLeft.index()].localType = FRType::Number;
    frameRegs_[frRight.index()].localType = FRType::Number;
    leftIsNum = rightIsNum = true;
    slow = false;
  } else {
    leftIsNum = isFRKnownNumber(frLeft);
    rightIsNum = isFRKnownNumber(frRight);
    slow = !(rightIsNum && leftIsNum);
  }

  syncAllFRTempExcept(frRes != frLeft && frRes != frRight ? frRes : FR());

  if (slow) {
    slowPathLab = newSlowPathLabel();
    contLab = newContLabel();
    syncToFrame(frLeft);
    syncToFrame(frRight);
  }

  hwLeft = getOrAllocFRInVecD(frLeft, true);
  hwRight = getOrAllocFRInVecD(frRight, true);

  if (slow) {
    // Since HermesValue is NaN-boxed we know that all non-number values will be
    // NaN. So we can conveniently test for non-number values by checking for
    // NaN. We can do that with the VS condition code, which is set if either
    // operand to fcmp is NaN.
    static_assert(HERMESVALUE_VERSION == 2, "Non-numbers must be NaN");
    a.fcmp(hwLeft.a64VecD(), hwRight.a64VecD());
    a.b_vs(slowPathLab);
  }

  // Make sure d0, d1 are unused.
  syncAndFreeTempReg(HWReg::vecD(0));
  movHWFromFR(HWReg::vecD(0), frLeft);
  syncAndFreeTempReg(HWReg::vecD(1));
  movHWFromFR(HWReg::vecD(1), frRight);

  EMIT_RUNTIME_CALL(*this, double (*)(double, double), _sh_mod_double);
  freeAllFRTempExcept({});
  hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::vecD(0));
  movHWFromHW<false>(hwRes, HWReg::vecD(0));
  frUpdatedWithHW(frRes, hwRes);

  if (!slow)
    return;

  a.bind(contLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .frRes = frRes,
       .frInput1 = frLeft,
       .frInput2 = frRight,
       .hwRes = hwRes,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// mod r%u, r%u, r%u",
             sl.frRes.index(),
             sl.frInput1.index(),
             sl.frInput2.index());
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         em.loadFrameAddr(a64::x1, sl.frInput1);
         em.loadFrameAddr(a64::x2, sl.frInput2);
         EMIT_RUNTIME_CALL(
             em,
             SHLegacyValue (*)(
                 SHRuntime *, const SHLegacyValue *, const SHLegacyValue *),
             _sh_ljs_mod_rjs);
         em.movHWFromHW<false>(sl.hwRes, HWReg::gpX(0));
         em.a.b(sl.contLab);
       }});
}

void Emitter::arithBinOp(
    bool forceNumber,
    FR frRes,
    FR frLeft,
    FR frRight,
    const char *name,
    void (*fast)(
        a64::Assembler &a,
        const a64::VecD &res,
        const a64::VecD &dl,
        const a64::VecD &dr),
    void *slowCall,
    const char *slowCallName) {
  comment(
      "// %s r%u, r%u, r%u",
      name,
      frRes.index(),
      frLeft.index(),
      frRight.index());
  HWReg hwRes, hwLeft, hwRight;
  asmjit::Label slowPathLab;
  asmjit::Label contLab;
  bool leftIsNum, rightIsNum, slow;

  if (forceNumber) {
    frameRegs_[frLeft.index()].localType = FRType::Number;
    frameRegs_[frRight.index()].localType = FRType::Number;
    leftIsNum = rightIsNum = true;
    slow = false;
  } else {
    leftIsNum = isFRKnownNumber(frLeft);
    rightIsNum = isFRKnownNumber(frRight);
    slow = !(rightIsNum && leftIsNum);
  }

  hwLeft = getOrAllocFRInVecD(frLeft, true);
  hwRight = getOrAllocFRInVecD(frRight, true);

  if (slow) {
    slowPathLab = newSlowPathLabel();
    contLab = newContLabel();
    syncAllFRTempExcept(frRes != frLeft && frRes != frRight ? frRes : FR());
    syncToFrame(frLeft);
    syncToFrame(frRight);
    freeAllFRTempExcept({});
  }

  hwRes = getOrAllocFRInVecD(frRes, false);
  frUpdatedWithHW(frRes, hwRes, !slow ? FRType::Number : FRType::UnknownPtr);

  if (slow) {
    // Since HermesValue is NaN-boxed we know that all non-number values will be
    // NaN. So we can conveniently test for non-number values by checking for
    // NaN. We can do that with the VS condition code, which is set if either
    // operand to fcmp is NaN.
    static_assert(HERMESVALUE_VERSION == 2, "Non-numbers must be NaN");
    a.fcmp(hwLeft.a64VecD(), hwRight.a64VecD());
    a.b_vs(slowPathLab);
  }

  fast(a, hwRes.a64VecD(), hwLeft.a64VecD(), hwRight.a64VecD());

  if (!slow)
    return;

  a.bind(contLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .name = name,
       .frRes = frRes,
       .frInput1 = frLeft,
       .frInput2 = frRight,
       .hwRes = hwRes,
       .slowCall = slowCall,
       .slowCallName = slowCallName,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// %s r%u, r%u, r%u",
             sl.name,
             sl.frRes.index(),
             sl.frInput1.index(),
             sl.frInput2.index());
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         em.loadFrameAddr(a64::x1, sl.frInput1);
         em.loadFrameAddr(a64::x2, sl.frInput2);
         em.callThunkWithSavedIP(sl.slowCall, sl.slowCallName);
         em.movHWFromHW<false>(sl.hwRes, HWReg::gpX(0));
         em.a.b(sl.contLab);
       }});
}

void Emitter::bitBinOp(
    FR frRes,
    FR frLeft,
    FR frRight,
    bool unsignedRes,
    const char *name,
    SHLegacyValue (*slowCall)(
        SHRuntime *shr,
        const SHLegacyValue *a,
        const SHLegacyValue *b),
    const char *slowCallName,
    void (*fast)(
        a64::Assembler &a,
        const a64::GpX &res,
        const a64::GpX &dl,
        const a64::GpX &dr)) {
  comment(
      "// %s r%u, r%u, r%u",
      name,
      frRes.index(),
      frLeft.index(),
      frRight.index());

  HWReg hwTempLGpX = allocTempGpX();
  HWReg hwTempRGpX = allocTempGpX();
  HWReg hwTempLVecD = allocTempVecD();
  HWReg hwTempRVecD = allocTempVecD();

  syncAllFRTempExcept(frRes != frLeft && frRes != frRight ? frRes : FR());
  // TODO: In principle, it should be possible to only sync these in the slow
  // path. If we do that, we have to ensure that the frameUpToDate bit is not
  // set, since subsequent instructions cannot rely on it. To do this, we would
  // need to preserve information for the slow path to know whether they were
  // already sync'd to memory.
  syncToFrame(frLeft);
  syncToFrame(frRight);

  asmjit::Label slowPathLab = newSlowPathLabel();
  asmjit::Label contLab = newContLabel();

  HWReg hwLeft = getOrAllocFRInVecD(frLeft, true);
  emit_double_is_int(
      a, hwTempLGpX.a64GpX(), hwTempLVecD.a64VecD(), hwLeft.a64VecD());
  a.b_ne(slowPathLab);

  // Do the same for the RHS.
  HWReg hwRight = getOrAllocFRInVecD(frRight, true);
  emit_double_is_int(
      a, hwTempRGpX.a64GpX(), hwTempRVecD.a64VecD(), hwRight.a64VecD());
  a.b_ne(slowPathLab);

  // Done allocating registers. Free them all and allocate the result.
  freeAllFRTempExcept({});
  freeReg(hwTempLGpX);
  freeReg(hwTempRGpX);
  freeReg(hwTempLVecD);
  freeReg(hwTempRVecD);
  HWReg hwRes = getOrAllocFRInVecD(frRes, false);
  frUpdatedWithHW(
      frRes,
      hwRes,
      isFRKnownNumber(frLeft) && isFRKnownNumber(frRight) ? FRType::Number
                                                          : FRType::UnknownPtr);

  // Invoke the fast path, and move the result back as a 32 bit integer.
  fast(a, hwTempLGpX.a64GpX(), hwTempLGpX.a64GpX(), hwTempRGpX.a64GpX());
  if (unsignedRes)
    a.ucvtf(hwRes.a64VecD(), hwTempLGpX.a64GpX().w());
  else
    a.scvtf(hwRes.a64VecD(), hwTempLGpX.a64GpX().w());

  a.bind(contLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .name = name,
       .frRes = frRes,
       .frInput1 = frLeft,
       .frInput2 = frRight,
       .hwRes = hwRes,
       .slowCall = (void *)slowCall,
       .slowCallName = slowCallName,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// %s r%u, r%u, r%u",
             sl.name,
             sl.frRes.index(),
             sl.frInput1.index(),
             sl.frInput2.index());
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         em.loadFrameAddr(a64::x1, sl.frInput1);
         em.loadFrameAddr(a64::x2, sl.frInput2);
         em.callThunkWithSavedIP(sl.slowCall, sl.slowCallName);
         em.movHWFromHW<false>(sl.hwRes, HWReg::gpX(0));
         em.a.b(sl.contLab);
       }});
}
void Emitter::jmpTrueFalse(
    bool onTrue,
    const asmjit::Label &target,
    FR frInput) {
  comment("// Jmp%s r%u", onTrue ? "True" : "False", frInput.index());

  // Do this always, since this could be the end of the BB.
  syncAllFRTempExcept(FR());

  if (isFRKnownType(frInput, FRType::Number)) {
    HWReg hwInput = getOrAllocFRInVecD(frInput, true);
    a.fcmp(hwInput.a64VecD(), 0.0);
    if (onTrue) {
      // Branch on < 0 and > 0. All that remains is 0 and NaN.
      a.b_mi(target);
      a.b_gt(target);
    } else {
      asmjit::Label label = a.newLabel();
      a.b_mi(label);
      a.b_gt(label);
      a.b(target);
      a.bind(label);
    }
  } else if (isFRKnownType(frInput, FRType::Bool)) {
    HWReg hwInput = getOrAllocFRInGpX(frInput, true);
    a64::GpX xInput = hwInput.a64GpX();

    static_assert(
        HERMESVALUE_VERSION == 2, "bool is encoded as a bit at kHV_BoolBitIdx");
    // We don't use tbz/tbnz here because they have a very restricted range.
    a.tst(xInput, 1ull << kHV_BoolBitIdx);
    a.b(onTrue ? a64::CondCode::kNotZero : a64::CondCode::kZero, target);
  } else {
    // TODO: we should inline all of it.
    syncAllFRTempExcept({});
    movHWFromFR(HWReg::gpX(0), frInput);
    EMIT_RUNTIME_CALL(*this, bool (*)(SHLegacyValue), _sh_ljs_to_boolean);
    if (onTrue)
      a.cbnz(a64::w0, target);
    else
      a.cbz(a64::w0, target);
    freeAllFRTempExcept(FR());
  }
}

void Emitter::jmp(const asmjit::Label &target) {
  comment("// Jmp Lx");
  // Do this always, since this could be the end of the BB.
  syncAllFRTempExcept(FR());
  freeAllFRTempExcept(FR());
  a.b(target);
}

void Emitter::jmpUndefined(const asmjit::Label &target, FR frInput) {
  comment("// JmpUndefined r%u", frInput.index());

  // Do this always, since this could be the end of the BB.
  syncAllFRTempExcept(FR());
  freeAllFRTempExcept(FR());

  if (isFRKnownType(frInput, FRType::Number) ||
      isFRKnownType(frInput, FRType::Bool)) {
    return;
  }

  HWReg hwInput = getOrAllocFRInGpX(frInput, true);
  a64::GpX xInput = hwInput.a64GpX();
  HWReg hwTmpTag = allocTempGpX();
  a64::GpX xTmpTag = hwTmpTag.a64GpX();

  emit_sh_ljs_is_undefined(a, xTmpTag, xInput);
  a.b_eq(target);

  freeReg(hwTmpTag);
}

void Emitter::jmpBuiltinIs(
    bool invert,
    const asmjit::Label &target,
    uint8_t builtinIndex,
    FR frInput) {
  comment(
      "// JmpBuiltinIs%s r%u, %u",
      invert ? "Not" : "",
      frInput.index(),
      builtinIndex);

  // Do this always, since this could be the end of the BB.
  syncAllFRTempExcept({});
  HWReg hwInput = getOrAllocFRInGpX(frInput, true);
  HWReg hwBuiltin = allocTempGpX();
  freeReg(hwBuiltin);
  freeAllFRTempExcept({});

  // Load builtin pointer.
  emit_load_builtin_closure(a, hwBuiltin.a64GpX(), builtinIndex);

  // Encode an object HermesValue.
  emit_sh_ljs_object(a, hwBuiltin.a64GpX());

  // Compare the builtin pointer with the input, branch.
  a.cmp(hwBuiltin.a64GpX(), hwInput.a64GpX());
  if (!invert)
    a.b_eq(target);
  else
    a.b_ne(target);
}

void Emitter::jCond(
    bool forceNumber,
    bool invert,
    bool passArgsByVal,
    const asmjit::Label &target,
    FR frLeft,
    FR frRight,
    const char *name,
    a64::CondCode condCode,
    void *slowCall,
    const char *slowCallName) {
  comment(
      "// j_%s_%s Lx, r%u, r%u",
      invert ? "not" : "",
      name,
      frLeft.index(),
      frRight.index());
  HWReg hwLeft, hwRight;
  asmjit::Label slowPathLab;
  asmjit::Label contLab;
  bool leftIsNum, rightIsNum, slow;

  if (forceNumber) {
    frameRegs_[frLeft.index()].localType = FRType::Number;
    frameRegs_[frRight.index()].localType = FRType::Number;
    leftIsNum = rightIsNum = true;
    slow = false;
  } else {
    leftIsNum = isFRKnownNumber(frLeft);
    rightIsNum = isFRKnownNumber(frRight);
    slow = !(rightIsNum && leftIsNum);
  }

  if (slow) {
    slowPathLab = newSlowPathLabel();
    contLab = newContLabel();
    syncToFrame(frLeft);
    syncToFrame(frRight);
  }
  // Do this always, since this could be the end of the BB.
  syncAllFRTempExcept(FR());

  hwLeft = getOrAllocFRInVecD(frLeft, true);
  hwRight = getOrAllocFRInVecD(frRight, true);

  a.fcmp(hwLeft.a64VecD(), hwRight.a64VecD());

  // If the condition is not inverted, then it can only produce true if both
  // operands are numbers. Since we use NaN boxing, we know that all non-number
  // values will be NaN and therefore produce false. So if the result is true,
  // we can take the jump without checking for numbers.
  if (!invert)
    a.b(condCode, target);

  if (slow) {
    // Since HermesValue is NaN-boxed we know that all non-number values will be
    // NaN. So we can conveniently test for non-number values by checking for
    // NaN. We can do that with the VS condition code, which is set if either
    // operand to fcmp is NaN.
    static_assert(HERMESVALUE_VERSION == 2, "Non-numbers must be NaN");
    a.b_vs(slowPathLab);
  }

  // If the condition is inverted, it will produce true if one of the operands
  // is a NaN, so we can only check it after the slow path check, since it would
  // incorrectly be taken for non-numbers.
  if (invert)
    a.b(a64::negateCond(condCode), target);

  if (!slow)
    return;

  a.bind(contLab);

  // Do this always, since this is the end of the BB.
  freeAllFRTempExcept(FR());

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .target = target,
       .name = name,
       .frInput1 = frLeft,
       .frInput2 = frRight,
       .invert = invert,
       .passArgsByVal = passArgsByVal,
       .slowCall = slowCall,
       .slowCallName = slowCallName,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// Slow path: j_%s%s Lx, r%u, r%u",
             sl.invert ? "not_" : "",
             sl.name,
             sl.frInput1.index(),
             sl.frInput2.index());
         em.a.bind(sl.slowPathLab);
         if (sl.passArgsByVal) {
           em._loadFrame(HWReg::gpX(0), sl.frInput1);
           em._loadFrame(HWReg::gpX(1), sl.frInput2);
         } else {
           em.a.mov(a64::x0, xRuntime);
           em.loadFrameAddr(a64::x1, sl.frInput1);
           em.loadFrameAddr(a64::x2, sl.frInput2);
         }
         em.callThunkWithSavedIP(sl.slowCall, sl.slowCallName);
         if (!sl.invert)
           em.a.cbnz(a64::w0, sl.target);
         else
           em.a.cbz(a64::w0, sl.target);
         em.a.b(sl.contLab);
       }});
}

void Emitter::jStrictEqual(
    bool invert,
    const asmjit::Label &target,
    FR frLeft,
    FR frRight) {
  comment(
      "// JStrict%sEq Lx, r%u, r%u",
      invert ? "Not" : "",
      frLeft.index(),
      frRight.index());

  // Fast path for raw-only comparison. One of the operands is a non-number
  // non-pointer value, meaning we can compare bits directly.
  if (isFRKnownBool(frLeft) || isFRKnownBool(frRight) ||
      isFRKnownOtherNonPtr(frLeft) || isFRKnownOtherNonPtr(frRight)) {
    // Do this always, since this could be the end of the BB.
    syncAllFRTempExcept({});
    HWReg hwLeft = getOrAllocFRInGpX(frLeft, true);
    HWReg hwRight = getOrAllocFRInGpX(frRight, true);
    freeAllFRTempExcept({});

    a.cmp(hwLeft.a64GpX(), hwRight.a64GpX());
    a.b(!invert ? a64::CondCode::kEQ : a64::CondCode::kNE, target);
    return;
  }

  // Fast path for number (double) comparison.
  // One of the operands is a number, so there's two cases:
  // * It's NaN: it'll fcmp false against everything which is what we want.
  // * It's not NaN: It won't have NaN tag bits so it'll compare false against
  //   all non-double HVs and correctly fcmp true against the same number.
  if (isFRKnownNumber(frLeft) || isFRKnownNumber(frRight)) {
    // Do this always, since this could be the end of the BB.
    syncAllFRTempExcept(FR());
    HWReg hwLeftD = getOrAllocFRInVecD(frLeft, true);
    HWReg hwRightD = getOrAllocFRInVecD(frRight, true);
    freeAllFRTempExcept({});
    a.fcmp(hwLeftD.a64VecD(), hwRightD.a64VecD());
    a.b(!invert ? a64::CondCode::kEQ : a64::CondCode::kNE, target);
    return;
  }

  // Do this always, since this could be the end of the BB.
  syncAllFRTempExcept(FR());

  HWReg hwLeftD = getOrAllocFRInVecD(frLeft, true);
  HWReg hwRightD = getOrAllocFRInVecD(frRight, true);

  HWReg hwTmpLeft = allocTempGpX();
  a64::GpX xTmpLeft = hwTmpLeft.a64GpX();
  HWReg hwTmpRight = allocTempGpX();
  a64::GpX xTmpRight = hwTmpRight.a64GpX();
  HWReg hwLeft = getOrAllocFRInGpX(frLeft, true);
  HWReg hwRight = getOrAllocFRInGpX(frRight, true);
  freeReg(hwTmpLeft);
  freeReg(hwTmpRight);

  // Label for non-number comparisons.
  auto nonNumberLab = a.newLabel();

  // Set up slow path.
  auto slowPathLab = newSlowPathLabel();
  auto contLab = newContLabel();
  syncToFrame(frLeft);
  syncToFrame(frRight);
  freeAllFRTempExcept(FR());

  a.fcmp(hwLeftD.a64VecD(), hwRightD.a64VecD());
  // If not inverted then equality here is real equality.
  // If the equality check fails, we don't know anything.
  if (!invert)
    a.b_eq(target);
  // Since HermesValue is NaN-boxed we know that all non-number values will be
  // NaN. So we can conveniently test for non-number values by checking for
  // NaN. We can do that with the VS condition code, which is set if either
  // operand to fcmp is NaN.
  static_assert(HERMESVALUE_VERSION == 2, "Non-numbers must be NaN");
  a.b_vs(nonNumberLab);
  // If neither number is NaN, then we can just jump to the correct endpoint.
  // We already checked equality above for the non-inverted case,
  // so just check the inverted case here to know if the values are not equal.
  // If all that fails, then the branch failed and we go to contLab.
  if (invert)
    a.b_ne(target);
  a.b(contLab);

  // Convenience labels so we don't have to think too hard about inverted logic
  // below.
  const asmjit::Label &equalLab = !invert ? target : contLab;
  const asmjit::Label &notEqualLab = !invert ? contLab : target;

  a.bind(nonNumberLab);
  emit_sh_ljs_is_double(a, hwLeft.a64GpX(), xTmpLeft);
  // Left is actually the JS NaN, which is never equal to anything.
  // No need to check RHS for JS NaN, because it won't cause false positive
  // on the raw bit check below.
  a.b_lo(notEqualLab);

  // Compare bits directly.
  // If they match exactly, the two values are equal.
  a.cmp(hwLeft.a64GpX(), hwRight.a64GpX());
  a.b_eq(equalLab);

  // First compare the tags. If they don't match, the two values are NOT equal.
  emit_sh_ljs_get_tag(a, xTmpLeft, hwLeft.a64GpX());
  emit_sh_ljs_get_tag(a, xTmpRight, hwRight.a64GpX());
  a.cmp(xTmpLeft, xTmpRight);
  a.b_ne(notEqualLab);

  // If the LHS is either a non-pointer or an object, we can compare raw values
  // only. We've already checked and we know that the raw values are not the
  // same, so if this is a non-pointer or an object, then the two values are NOT
  // strictly equal.
  emit_sh_ljs_tag_is_pointer(a, xTmpLeft);
  a.b_lo(notEqualLab);
  emit_sh_ljs_tag_is_object(a, xTmpLeft);
  a.b_eq(notEqualLab);

  // Now we know that the LHS is a non-object pointer.

  // Fast string path: string inequality can be easily determined by checking
  // the lengths. If the LHS isn't a string, go to slow path.
  emit_sh_ljs_tag_is_string(a, xTmpLeft);
  a.b_ne(slowPathLab);

  emit_stringprim_get_length_and_flags(a, xTmpLeft, hwLeft.a64GpX());
  emit_stringprim_get_length_and_flags(a, xTmpRight, hwRight.a64GpX());
  // XOR the lengths together and mask the result.
  // xTmpLeft will be nonzero if the lengths don't match.
  a.eor(xTmpLeft, xTmpLeft, xTmpRight);
  a.and_(xTmpLeft, xTmpLeft, RuntimeOffsets::stringPrimitiveLengthMask);
  // Length mismatch means we're done, the two values are NOT equal.
  a.cbnz(xTmpLeft, notEqualLab);
  a.b(slowPathLab);

  a.bind(contLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .target = target,
       .name = invert ? "j_strict_not_eq" : "j_strict_eq",
       .frInput1 = frLeft,
       .frInput2 = frRight,
       .invert = invert,
       .slowCall = (void *)_sh_ljs_strict_equal,
       .slowCallName = "_sh_ljs_strict_equal",
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// Slow path: %s Lx, r%u, r%u",
             sl.name,
             sl.frInput1.index(),
             sl.frInput2.index());
         em.a.bind(sl.slowPathLab);
         em._loadFrame(HWReg::gpX(0), sl.frInput1);
         em._loadFrame(HWReg::gpX(1), sl.frInput2);
         em.callThunkWithSavedIP(sl.slowCall, sl.slowCallName);
         if (!sl.invert)
           em.a.cbnz(a64::w0, sl.target);
         else
           em.a.cbz(a64::w0, sl.target);
         em.a.b(sl.contLab);
       }});
}

void Emitter::strictEqualImpl(bool invert, FR frRes, FR frLeft, FR frRight) {
  comment(
      "// %s r%u, r%u, r%u",
      !invert ? "StrictEq" : "StrictNEq",
      frRes.index(),
      frLeft.index(),
      frRight.index());

  // Fast path for raw-only comparison. One of the operands is a non-number
  // non-pointer value, meaning we can compare bits directly.
  if (isFRKnownBool(frLeft) || isFRKnownBool(frRight) ||
      isFRKnownOtherNonPtr(frLeft) || isFRKnownOtherNonPtr(frRight)) {
    HWReg hwLeft = getOrAllocFRInGpX(frLeft, true);
    HWReg hwRight = getOrAllocFRInGpX(frRight, true);
    HWReg hwRes = getOrAllocFRInGpX(frRes, false);
    frUpdatedWithHW(frRes, hwRes, FRType::Bool);

    a.cmp(hwLeft.a64GpX(), hwRight.a64GpX());
    a.cset(hwRes.a64GpX(), !invert ? a64::CondCode::kEQ : a64::CondCode::kNE);
    emit_sh_ljs_bool(a, hwRes.a64GpX());
    return;
  }

  // Fast path for number (double) comparison.
  // One of the operands is a number, so there's two cases:
  // * It's NaN: it'll fcmp false against everything which is what we want.
  // * It's not NaN: It won't have NaN tag bits so it'll compare false against
  //   all non-double HVs and correctly fcmp true against the same number.
  if (isFRKnownNumber(frLeft) || isFRKnownNumber(frRight)) {
    // Do this always, since this could be the end of the BB.
    HWReg hwLeftD = getOrAllocFRInVecD(frLeft, true);
    HWReg hwRightD = getOrAllocFRInVecD(frRight, true);
    HWReg hwRes = getOrAllocFRInGpX(frRes, false);
    frUpdatedWithHW(frRes, hwRes, FRType::Bool);

    a.fcmp(hwLeftD.a64VecD(), hwRightD.a64VecD());
    a.cset(hwRes.a64GpX(), !invert ? a64::CondCode::kEQ : a64::CondCode::kNE);
    emit_sh_ljs_bool(a, hwRes.a64GpX());
    return;
  }

  HWReg hwLeftD = getOrAllocFRInVecD(frLeft, true);
  HWReg hwRightD = getOrAllocFRInVecD(frRight, true);

  // Allocate registers used for non-number comparisons.
  HWReg hwLeft = getOrAllocFRInGpX(frLeft, true);
  HWReg hwRight = getOrAllocFRInGpX(frRight, true);
  HWReg hwTmpLeft = allocTempGpX();
  HWReg hwTmpRight = allocTempGpX();
  a64::GpX xTmpLeft = hwTmpLeft.a64GpX();
  a64::GpX xTmpRight = hwTmpRight.a64GpX();
  freeReg(hwTmpLeft);
  freeReg(hwTmpRight);

  // Labels for non-number comparisons.
  auto nonNumberLab = a.newLabel();
  auto equalLab = a.newLabel();
  auto notEqualLab = a.newLabel();

  // Set up slow path.
  auto slowPathLab = newSlowPathLabel();
  auto contLab = newContLabel();
  syncAllFRTempExcept(frRes != frLeft && frRes != frRight ? frRes : FR());
  syncToFrame(frLeft);
  syncToFrame(frRight);
  freeAllFRTempExcept({});

  HWReg hwRes = getOrAllocFRInGpX(frRes, false, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes, FRType::Bool);
  a64::GpX xRes = hwRes.a64GpX();

  // Start by comparing doubles with fcmp.
  a.fcmp(hwLeftD.a64VecD(), hwRightD.a64VecD());

  // Since HermesValue is NaN-boxed we know that all non-number values will be
  // NaN. So we can conveniently test for non-number values by checking for
  // NaN. We can do that with the VS condition code, which is set if either
  // operand to fcmp is NaN.
  static_assert(HERMESVALUE_VERSION == 2, "Non-numbers must be NaN");
  a.b_vs(nonNumberLab);

  // Store the result of the comparison in the lowest bit of tmpCmpRes.
  // asmjit will convert CondCode to the correct encoding for use in the opcode.
  a.cset(xRes, invert ? a64::CondCode::kNE : a64::CondCode::kEQ);
  emit_sh_ljs_bool(a, xRes);

  a.b(contLab);

  // May be JS NaN (not a number, but really it is a number).
  a.bind(nonNumberLab);
  emit_sh_ljs_is_double(a, hwLeft.a64GpX(), xTmpLeft);
  // Left is actually the JS NaN, which is never equal to anything.
  // No need to check RHS for JS NaN, because it won't cause false positive
  // on the raw bit check below.
  a.b_lo(notEqualLab);

  // Compare bits directly.
  // If they match exactly, the two values are equal.
  a.cmp(hwLeft.a64GpX(), hwRight.a64GpX());
  a.b_eq(equalLab);

  // First compare the tags. If they don't match, the two values are NOT equal.
  emit_sh_ljs_get_tag(a, xTmpLeft, hwLeft.a64GpX());
  emit_sh_ljs_get_tag(a, xTmpRight, hwRight.a64GpX());
  a.cmp(xTmpLeft, xTmpRight);
  a.b_ne(notEqualLab);

  // If the LHS is either a non-pointer or an object, we can compare raw values
  // only. We've already checked and we know that the raw values are not the
  // same, so if this is a non-pointer or an object, then the two values are NOT
  // strictly equal.
  emit_sh_ljs_tag_is_pointer(a, xTmpLeft);
  a.b_lo(notEqualLab);
  emit_sh_ljs_tag_is_object(a, xTmpLeft);
  a.b_eq(notEqualLab);

  // Now we know that the LHS is a non-object pointer.

  // Fast string path: string inequality can be easily determined by checking
  // the lengths. If the LHS isn't a string, go to slow path.
  emit_sh_ljs_tag_is_string(a, xTmpLeft);
  a.b_ne(slowPathLab);

  emit_stringprim_get_length_and_flags(a, xTmpLeft, hwLeft.a64GpX());
  emit_stringprim_get_length_and_flags(a, xTmpRight, hwRight.a64GpX());
  // XOR the lengths together and mask the result.
  // xTmpLeft will be nonzero if the lengths don't match.
  a.eor(xTmpLeft, xTmpLeft, xTmpRight);
  a.and_(xTmpLeft, xTmpLeft, RuntimeOffsets::stringPrimitiveLengthMask);
  // Length mismatch means we're done, the two values are NOT equal.
  a.cbnz(xTmpLeft, notEqualLab);
  a.b(slowPathLab);

  // Jump here if the result should be "equal".
  // Returns true if not inverted, false if inverted.
  a.bind(equalLab);
  emit_sh_ljs_bool_const(a, xRes, !invert);
  a.b(contLab);

  // Jump here if the result should be "not equal".
  // Returns false if not inverted, true if inverted.
  a.bind(notEqualLab);
  emit_sh_ljs_bool_const(a, xRes, invert);

  a.bind(contLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .name = !invert ? "StrictEq" : "StrictNEq",
       .frRes = frRes,
       .frInput1 = frLeft,
       .frInput2 = frRight,
       .hwRes = hwRes,
       .invert = invert,
       .passArgsByVal = true,
       .slowCall = (void *)_sh_ljs_strict_equal,
       .slowCallName = "_sh_ljs_strict_equal",
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// Slow path: %s r%u, r%u, r%u",
             sl.name,
             sl.frRes.index(),
             sl.frInput1.index(),
             sl.frInput2.index());
         em.a.bind(sl.slowPathLab);
         if (sl.passArgsByVal) {
           em._loadFrame(HWReg::gpX(0), sl.frInput1);
           em._loadFrame(HWReg::gpX(1), sl.frInput2);
         } else {
           em.a.mov(a64::x0, xRuntime);
           em.loadFrameAddr(a64::x1, sl.frInput1);
           em.loadFrameAddr(a64::x2, sl.frInput2);
         }
         em.callThunkWithSavedIP(sl.slowCall, sl.slowCallName);

         // Invert the slow path result if needed.
         if (sl.invert)
           em.a.eor(sl.hwRes.a64GpX(), a64::x0, 1);
         else
           em.movHWFromHW<false>(sl.hwRes, HWReg::gpX(0));

         // Comparison functions return bool, so encode it.
         emit_sh_ljs_bool(em.a, sl.hwRes.a64GpX());
         em.a.b(sl.contLab);
       }});
}

void Emitter::compareImpl(
    FR frRes,
    FR frLeft,
    FR frRight,
    const char *name,
    a64::CondCode condCode,
    void *slowCall,
    const char *slowCallName,
    bool invSlow,
    bool passArgsByVal) {
  comment(
      "// %s r%u, r%u, r%u",
      name,
      frRes.index(),
      frLeft.index(),
      frRight.index());
  HWReg hwLeft, hwRight;
  asmjit::Label slowPathLab;
  asmjit::Label contLab;
  bool leftIsNum, rightIsNum, slow;

  leftIsNum = isFRKnownNumber(frLeft);
  rightIsNum = isFRKnownNumber(frRight);
  slow = !(rightIsNum && leftIsNum);

  hwLeft = getOrAllocFRInVecD(frLeft, true);
  hwRight = getOrAllocFRInVecD(frRight, true);
  if (slow) {
    slowPathLab = newSlowPathLabel();
    contLab = newContLabel();
    syncAllFRTempExcept(frRes != frLeft && frRes != frRight ? frRes : FR());
    syncToFrame(frLeft);
    syncToFrame(frRight);
    freeAllFRTempExcept({});
  }

  HWReg hwRes = getOrAllocFRInGpX(frRes, false, HWReg::gpX(0));
  a64::GpX xRes = hwRes.a64GpX();

  a.fcmp(hwLeft.a64VecD(), hwRight.a64VecD());

  if (slow) {
    // Since HermesValue is NaN-boxed we know that all non-number values will be
    // NaN. So we can conveniently test for non-number values by checking for
    // NaN. We can do that with the VS condition code, which is set if either
    // operand to fcmp is NaN.
    static_assert(HERMESVALUE_VERSION == 2, "Non-numbers must be NaN");
    a.b_vs(slowPathLab);
  }

  // Store the result of the comparison in the lowest bit of tmpCmpRes.
  // asmjit will convert CondCode to the correct encoding for use in the opcode.
  a.cset(xRes, condCode);

  // Encode bool.
  emit_sh_ljs_bool(a, xRes);
  frUpdatedWithHW(frRes, hwRes, FRType::Bool);

  if (!slow)
    return;

  a.bind(contLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .name = name,
       .frRes = frRes,
       .frInput1 = frLeft,
       .frInput2 = frRight,
       .hwRes = hwRes,
       .invert = invSlow,
       .passArgsByVal = passArgsByVal,
       .slowCall = slowCall,
       .slowCallName = slowCallName,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment(
             "// Slow path: j_%s r%u, r%u, r%u",
             sl.name,
             sl.frRes.index(),
             sl.frInput1.index(),
             sl.frInput2.index());
         em.a.bind(sl.slowPathLab);
         if (sl.passArgsByVal) {
           em._loadFrame(HWReg::gpX(0), sl.frInput1);
           em._loadFrame(HWReg::gpX(1), sl.frInput2);
         } else {
           em.a.mov(a64::x0, xRuntime);
           em.loadFrameAddr(a64::x1, sl.frInput1);
           em.loadFrameAddr(a64::x2, sl.frInput2);
         }
         em.callThunkWithSavedIP(sl.slowCall, sl.slowCallName);

         // Invert the slow path result if needed.
         if (sl.invert)
           em.a.eor(sl.hwRes.a64GpX(), a64::x0, 1);
         else
           em.movHWFromHW<false>(sl.hwRes, HWReg::gpX(0));

         // Comparison functions return bool, so encode it.
         emit_sh_ljs_bool(em.a, sl.hwRes.a64GpX());
         em.a.b(sl.contLab);
       }});
}

void Emitter::getArgumentsPropByValImpl(
    FR frRes,
    FR frIndex,
    FR frLazyReg,
    const char *name,
    SHLegacyValue (*shImpl)(
        SHRuntime *shr,
        SHLegacyValue *frame,
        SHLegacyValue *idx,
        SHLegacyValue *lazyReg),
    const char *shImplName) {
  comment(
      "// %s r%u, r%u, r%u",
      name,
      frRes.index(),
      frIndex.index(),
      frLazyReg.index());

  asmjit::Label slowPathLab = newSlowPathLabel();
  asmjit::Label contLab = newContLabel();

  syncAllFRTempExcept(frRes != frIndex && frRes != frLazyReg ? frRes : FR());
  syncToFrame(frIndex);
  syncToFrame(frLazyReg);
  HWReg hwLazyReg = getOrAllocFRInGpX(frLazyReg, true);
  HWReg hwIndex = getOrAllocFRInVecD(frIndex, true);
  HWReg hwTempIndex = allocTempGpX();
  a64::GpW wTempIndex = hwTempIndex.a64GpX().w();
  HWReg hwTempArgCount = allocTempGpX();
  HWReg hwTempVecD = allocTempVecD();
  freeAllFRTempExcept({});
  freeReg(hwTempIndex);
  freeReg(hwTempArgCount);
  freeReg(hwTempVecD);
  HWReg hwRes = getOrAllocFRInAnyReg(frRes, false, HWReg::gpX(0));
  frUpdatedWithHW(frRes, hwRes);

  // If lazyReg is an object, go to slow path.
  emit_sh_ljs_is_object(a, hwTempIndex.a64GpX(), hwLazyReg.a64GpX());
  a.b_eq(slowPathLab);

  // If index is not an array index, go to slow path.
  emit_double_is_int(
      a, hwTempIndex.a64GpX(), hwTempVecD.a64VecD(), hwIndex.a64VecD());
  a.b_ne(slowPathLab);

  // If index >= arg count or index < 0, go to slow path.
  // Use an unsigned comparison to handle the negative index case.
  a.ldur(
      hwTempArgCount.a64GpX().w(),
      a64::Mem(
          xFrame,
          (int)StackFrameLayout::ArgCount * (int)sizeof(SHLegacyValue)));
  a.cmp(hwTempIndex.a64GpX(), hwTempArgCount.a64GpX());
  a.b_hs(slowPathLab);

  // Load the argument from the stack.
  // We want framePtr[(firstArg - index) * 8].
  // Use shift SXTW to shift the signed w register by 3.
  a.mov(hwTempArgCount.a64GpX().w(), (int)StackFrameLayout::FirstArg);
  a.sub(wTempIndex, hwTempArgCount.a64GpX().w(), wTempIndex);
  a.ldr(
      hwRes.a64GpX(),
      a64::Mem(xFrame, wTempIndex, a64::Shift(a64::ShiftOp::kSXTW, 3)));

  a.bind(contLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .name = name,
       .frRes = frRes,
       .frInput1 = frIndex,
       .frInput2 = frLazyReg,
       .hwRes = hwRes,
       .slowCall = (void *)shImpl,
       .slowCallName = shImplName,
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment("// Slow path: %s r%u", sl.name, sl.frInput1.index());
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         em.a.mov(a64::x1, xFrame);
         em.loadFrameAddr(a64::x2, sl.frInput1);
         em.loadFrameAddr(a64::x3, sl.frInput2);
         em.callThunkWithSavedIP(sl.slowCall, sl.slowCallName);
         em.movHWFromHW<false>(sl.hwRes, HWReg::gpX(0));
         em.a.b(sl.contLab);
       }});
}

void Emitter::reifyArgumentsImpl(FR frLazyReg, bool strict, const char *name) {
  comment("// %s r%u", name, frLazyReg.index());

  asmjit::Label slowPathLab = newSlowPathLabel();
  asmjit::Label contLab = newContLabel();

  syncAllFRTempExcept({});
  syncToFrame(frLazyReg);

  HWReg hwLazyReg = getOrAllocFRInGpX(frLazyReg, true);
  HWReg hwTemp = allocTempGpX();
  freeAllFRTempExcept({});
  freeReg(hwTemp);

  emit_sh_ljs_is_object(a, hwTemp.a64GpX(), hwLazyReg.a64GpX());
  // If the lazyReg is not an object, it needs to be reified, go to slow path.
  a.b_ne(slowPathLab);

  // Fast path: do nothing.
  a.bind(contLab);

  slowPaths_.push_back(
      {.slowPathLab = slowPathLab,
       .contLab = contLab,
       .name = name,
       .frInput1 = frLazyReg,
       // Use hwRes field to pass the global reg for the in/out param.
       .hwRes = frameRegs_[frLazyReg.index()].globalReg,
       .slowCall = strict ? (void *)_sh_ljs_reify_arguments_strict
                          : (void *)_sh_ljs_reify_arguments_loose,
       .slowCallName = strict ? "_sh_ljs_reify_arguments_strict"
                              : "_sh_ljs_reify_arguments_loose",
       .emittingIP = emittingIP,
       .emit = [](Emitter &em, SlowPath &sl) {
         em.comment("// Slow path: %s r%u", sl.name, sl.frInput1.index());
         em.a.bind(sl.slowPathLab);
         em.a.mov(a64::x0, xRuntime);
         em.a.mov(a64::x1, xFrame);
         em.loadFrameAddr(a64::x2, sl.frInput1);
         em.callThunkWithSavedIP(sl.slowCall, sl.slowCallName);
         // Slow path modifies the frame so we need to sync it if there's a
         // global reg.
         if (sl.hwRes.isValid()) {
           em._loadFrame(sl.hwRes, sl.frInput1);
         }
         em.a.b(sl.contLab);
       }});
}

} // namespace hermes::vm::arm64
#endif // HERMESVM_JIT
