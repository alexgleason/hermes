/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hermes/VM/JIT/Config.h"
#if HERMESVM_JIT

#include "JitEmitter-internal.h"

namespace hermes::vm::arm64 {

void Emit_sh_shv_decode::emitFirstCase(a64::Assembler &a) {
#ifdef HERMESVM_BOXED_DOUBLES
  static_assert(HERMESVALUE_VERSION == 2, "Constructing HermesValue from bits");
  static_assert(HermesValue32::kVersion == 1, "Decoding HV32 bits");
  constexpr uint64_t kHV32TagMask = (1 << HermesValue32::kNumTagBits) - 1;

  // Start by testing the tag bits for 0, if so, this is a "compressed HV64".
  static_assert((int)HermesValue32::Tag::CompressedHV64 == 0);
  a.tst(xInOut, kHV32TagMask);
  // If the tag is non-zero, jump to the pointer check.
  a.b_ne(ptrLab);
  // CompressedHV64, just shift the bits if needed. Note the case where
  // HermesValue32 is actually 64 bits is only used for testing, so it is not
  // worth optimising here.
  if constexpr (sizeof(SmallHermesValue) < sizeof(HermesValue))
    a.lsl(xInOut, xInOut, 32);
#endif
}

void Emit_sh_shv_decode::emitRestCases(a64::Assembler &a) {
#ifndef NDEBUG
  assert(!restEmitted && "Rest cases already emitted");
  restEmitted = true;
#endif
#ifdef HERMESVM_BOXED_DOUBLES
  asmjit::Label symLab = a.newLabel();
  asmjit::Label bdLab = a.newLabel();

  static_assert(HERMESVALUE_VERSION == 2, "Constructing HermesValue from bits");
  static_assert(HermesValue32::kVersion == 1, "Decoding HV32 bits");
  constexpr uint64_t kHV32TagMask = (1 << HermesValue32::kNumTagBits) - 1;

  a.bind(ptrLab);

  // See the comments below for why the exact bits are important.
  static_assert((int)HermesValue32::Tag::String == 0b01);
  static_assert((int)HermesValue32::Tag::BigInt == 0b10);
  static_assert((int)HermesValue32::Tag::Object == 0b11);
  // We know that the tag is 3 bits, and we have already checked for 0, so we
  // can check for the 3 interesting pointer tags (which are in the range 1-3,
  // i.e., less than 2^2) just by testing the most significant bit of the tag.
  a.tbnz(xInOut, 2, symLab);
  // We can decompress the pointer right away, since the HV32 representation
  // just makes it look like an unaligned compressed pointer.
  emit_sh_cp_decode_non_null(a, xInOut);
  // Now we insert the new tag. First compute the "base" tag. The HV64 tag for a
  // pointer can be computed by adding the HV32 tag to this base.
  constexpr uint16_t baseTag =
      (uint16_t)HermesValue::Tag::Str - (uint16_t)HermesValue32::Tag::String;
  // Move in the base tag.
  a.movk(xInOut, baseTag, kHV_NumDataBits);

  // We know that the base tag has zeroes in its low 2 bits, and the HV32 tag is
  // known to be in the low 2 bits at this point. So we can use bfi to simply
  // copy those low bits over.
  static_assert((baseTag & 0b11) == 0);
  a.bfi(xInOut, xInOut, kHV_NumDataBits, 2);

  // Clear the HV32 tag in the low bits.
  a.and_(xInOut, xInOut, ~kHV32TagMask);
  a.b(doneLab);

  a.bind(symLab);
  // There are only two cases left, Symbol and BoxedDouble. We can distinguish
  // them with a single bit.
  static_assert((int)HermesValue32::Tag::BoxedDouble == 0b100);
  static_assert((int)HermesValue32::Tag::Symbol == 0b101);
  static_assert((int)HermesValue32::Tag::_Last == 0b110);
  // Test the low bit, to distinguish Symbol and BoxedDouble.
  a.tbz(xInOut, 0, bdLab);
  // SymbolID has a 17-bit tag in HV64, which means the tag cannot be inserted
  // in a single instruction. However, we can exploit the fact that HV64 tags
  // are particular double NaN's, with a sign bit of 1 and all-1 exponents, so
  // they start with 0xfff. This means that the most significant bits of the tag
  // are 1, and we can shift in those bits with an asr below as we shift out the
  // HV32 tag. First, insert the low 14 bits of the symbol tag as a 16 bit
  // immediate.
  constexpr uint16_t symTag =
      (uint16_t)((uint32_t)HVETag_Symbol << (HermesValue32::kNumTagBits - 1));
  // Assert that the top bits are preserved with an arithmetic right shift.
  static_assert(
      ((int16_t)symTag) >> (HermesValue32::kNumTagBits - 1) == HVETag_Symbol);
  a.movk(xInOut, symTag, kHV_NumDataBits);
  // Next, shift in the high 1's as we shift out the HV32 tag.
  a.asr(xInOut, xInOut, HermesValue32::kNumTagBits);
  a.b(doneLab);

  a.bind(bdLab);
  // It is a boxed double. As with pointers, we can decode it as a compressed
  // pointer first.
  emit_sh_cp_decode_non_null(a, xInOut);
  // Since the tag is still present, subtract it from the offset.
  constexpr size_t bdOffs = RuntimeOffsets::boxedDoubleValue -
      (size_t)HermesValue32::Tag::BoxedDouble;
  a.ldr(xInOut, a64::Mem(xInOut, bdOffs));
#endif
}

} // namespace hermes::vm::arm64

#endif // HERMESVM_JIT
