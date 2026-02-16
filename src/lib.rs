//! [![github]](https://github.com/dtolnay/zmij)&ensp;[![crates-io]](https://crates.io/crates/zmij)&ensp;[![docs-rs]](https://docs.rs/zmij)
//!
//! [github]: https://img.shields.io/badge/github-8da0cb?style=for-the-badge&labelColor=555555&logo=github
//! [crates-io]: https://img.shields.io/badge/crates.io-fc8d62?style=for-the-badge&labelColor=555555&logo=rust
//! [docs-rs]: https://img.shields.io/badge/docs.rs-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs
//!
//! <br>
//!
//! A double-to-string conversion algorithm based on [Schubfach] and [yy].
//!
//! This Rust implementation is a line-by-line port of Victor Zverovich's
//! implementation in C++, <https://github.com/vitaut/zmij>.
//!
//! [Schubfach]: https://fmt.dev/papers/Schubfach4.pdf
//! [yy]: https://github.com/ibireme/c_numconv_benchmark/blob/master/vendor/yy_double/yy_double.c
//!
//! <br>
//!
//! # Example
//!
//! ```
//! fn main() {
//!     let mut buffer = zmij::Buffer::new();
//!     let printed = buffer.format(1.234);
//!     assert_eq!(printed, "1.234");
//! }
//! ```
//!
//! <br>
//!
//! ## Performance
//!
//! The [dtoa-benchmark] compares this library and other Rust floating point
//! formatting implementations across a range of precisions. The vertical axis
//! in this chart shows nanoseconds taken by a single execution of
//! `zmij::Buffer::new().format_finite(value)` so a lower result indicates a
//! faster library.
//!
//! [dtoa-benchmark]: https://github.com/dtolnay/dtoa-benchmark
//!
//! ![performance](https://raw.githubusercontent.com/dtolnay/zmij/master/dtoa-benchmark.png)

#![no_std]
#![doc(html_root_url = "https://docs.rs/zmij/1.0.19")]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(non_camel_case_types, non_snake_case)]
#![allow(
    clippy::blocks_in_conditions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_ptr_alignment,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::incompatible_msrv,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::must_use_candidate,
    clippy::needless_doctest_main,
    clippy::never_loop,
    clippy::redundant_else,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::used_underscore_items,
    clippy::while_immutable_condition,
    clippy::wildcard_imports
)]

#![feature(f16)]

#[cfg(test)]
mod tests;
mod traits;

#[cfg(not(zmij_no_select_unpredictable))]
use core::hint;
use core::mem::{self, MaybeUninit};
use core::ptr;
use core::slice;
use core::str;
#[cfg(feature = "no-panic")]
use no_panic::no_panic;

const BUFFER_SIZE: usize = 24;
const NAN: &str = "NaN";
const INFINITY: &str = "inf";
const NEG_INFINITY: &str = "-inf";

// Returns true_value if lhs < rhs, else false_value, without branching.
#[inline]
fn select_if_less(lhs: u64, rhs: u64, true_value: i64, false_value: i64) -> i64 {
    hint::select_unpredictable(lhs < rhs, true_value, false_value)
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct uint128 {
    hi: u64,
    lo: u64,
}

// Use umul128_hi64 for division.
const USE_UMUL128_HI64: bool = cfg!(target_vendor = "apple");

// Computes 128-bit result of multiplication of two 64-bit unsigned integers.
const fn umul128(x: u64, y: u64) -> u128 {
    x as u128 * y as u128
}

const fn umul128_hi64(x: u64, y: u64) -> u64 {
    (umul128(x, y) >> 64) as u64
}

#[cfg_attr(feature = "no-panic", no_panic)]
fn umul192_hi128(x_hi: u64, x_lo: u64, y: u64) -> uint128 {
    let p = umul128(x_hi, y);
    let lo = (p as u64).wrapping_add((umul128(x_lo, y) >> 64) as u64);
    uint128 {
        hi: (p >> 64) as u64 + u64::from(lo < p as u64),
        lo,
    }
}

// Computes high 64 bits of multiplication of x and y, discards the least
// significant bit and rounds to odd, where x = uint128_t(x_hi << 64) | x_lo.
#[cfg_attr(feature = "no-panic", no_panic)]
fn umulhi_inexact_to_odd<UInt>(x_hi: u64, x_lo: u64, y: UInt) -> UInt
where
    UInt: traits::UInt,
{
    let num_bits = mem::size_of::<UInt>() * 8;
    if num_bits == 64 {
        let p = umul192_hi128(x_hi, x_lo, y.into());
        UInt::truncate(p.hi | u64::from((p.lo >> 1) != 0))
    } else if num_bits == 32 {
        let p = (umul128(x_hi, y.into()) >> 32) as u64;
        UInt::enlarge((p >> 32) as u32 | u32::from((p as u32 >> 1) != 0))
    } else {
        let p = (umul128(x_hi, y.into()) >> 16) as u64;
        UInt::truncate((p >> 16) | u64::from((p >> 1) != 0))
    }
}

trait FloatTraits: traits::Float {
    const NUM_BITS: i32;
    const NUM_SIG_BITS: i32 = Self::MANTISSA_DIGITS as i32 - 1;
    const NUM_EXP_BITS: i32 = Self::NUM_BITS - Self::NUM_SIG_BITS - 1;
    const EXP_MASK: i32 = (1 << Self::NUM_EXP_BITS) - 1;
    const EXP_BIAS: i32 = (1 << (Self::NUM_EXP_BITS - 1)) - 1;
    const EXP_OFFSET: i32 = Self::EXP_BIAS + Self::NUM_SIG_BITS;

    type SigType: traits::UInt;
    const IMPLICIT_BIT: Self::SigType;

    fn to_bits(self) -> Self::SigType;

    fn is_negative(bits: Self::SigType) -> bool {
        (bits >> (Self::NUM_BITS - 1)) != Self::SigType::from(0)
    }

    fn get_sig(bits: Self::SigType) -> Self::SigType {
        bits & (Self::IMPLICIT_BIT - Self::SigType::from(1))
    }

    fn get_exp(bits: Self::SigType) -> i64 {
        (bits << 1u8 >> (Self::NUM_SIG_BITS + 1)).into() as i64
    }
}

impl FloatTraits for f16 {
    const NUM_BITS: i32 = 16;
    const IMPLICIT_BIT: Self::SigType = 1 << Self::NUM_SIG_BITS;

    type SigType = u16;

    fn to_bits(self) -> Self::SigType {
        self.to_bits()
    }
}

impl FloatTraits for f32 {
    const NUM_BITS: i32 = 32;
    const IMPLICIT_BIT: u32 = 1 << Self::NUM_SIG_BITS;

    type SigType = u32;

    fn to_bits(self) -> Self::SigType {
        self.to_bits()
    }
}

impl FloatTraits for f64 {
    const NUM_BITS: i32 = 64;
    const IMPLICIT_BIT: u64 = 1 << Self::NUM_SIG_BITS;

    type SigType = u64;

    fn to_bits(self) -> Self::SigType {
        self.to_bits()
    }
}

#[repr(C, align(64))]
struct Pow10SignificandsTable {
    data: [u64; Self::NUM_POW10 * 2],
}

impl Pow10SignificandsTable {
    const SPLIT_TABLES: bool = cfg!(target_arch = "aarch64");
    const NUM_POW10: usize = 617;

    unsafe fn get_unchecked(&self, dec_exp: i32) -> uint128 {
        const DEC_EXP_MIN: i32 = -292;
        if !Self::SPLIT_TABLES {
            let index = ((dec_exp - DEC_EXP_MIN) * 2) as usize;
            return uint128 {
                hi: unsafe { *self.data.get_unchecked(index) },
                lo: unsafe { *self.data.get_unchecked(index + 1) },
            };
        }

        unsafe {
            let hi = self
                .data
                .as_ptr()
                .offset(Self::NUM_POW10 as isize + DEC_EXP_MIN as isize - 1);
            let lo = hi.add(Self::NUM_POW10);

            uint128 {
                hi: *hi.offset(-dec_exp as isize),
                lo: *lo.offset(-dec_exp as isize),
            }
        }
    }

    #[cfg(test)]
    fn get(&self, dec_exp: i32) -> uint128 {
        const DEC_EXP_MIN: i32 = -292;
        assert!((DEC_EXP_MIN..DEC_EXP_MIN + Self::NUM_POW10 as i32).contains(&dec_exp));
        unsafe { self.get_unchecked(dec_exp) }
    }
}

// 128-bit significands of powers of 10 rounded down.
// Generated using 192-bit arithmetic method by Dougall Johnson.
static POW10_SIGNIFICANDS: Pow10SignificandsTable = {
    let mut data = [0; Pow10SignificandsTable::NUM_POW10 * 2];

    struct uint192 {
        w0: u64, // least significant
        w1: u64,
        w2: u64, // most significant
    }

    // First element, rounded up to cancel out rounding down in the
    // multiplication, and minimize significant bits.
    let mut current = uint192 {
        w0: 0xe000000000000000,
        w1: 0x25e8e89c13bb0f7a,
        w2: 0xff77b1fcbebcdc4f,
    };
    let ten = 0xa000000000000000;
    let mut i = 0;
    while i < Pow10SignificandsTable::NUM_POW10 {
        if Pow10SignificandsTable::SPLIT_TABLES {
            data[Pow10SignificandsTable::NUM_POW10 - i - 1] = current.w2;
            data[Pow10SignificandsTable::NUM_POW10 * 2 - i - 1] = current.w1;
        } else {
            data[i * 2] = current.w2;
            data[i * 2 + 1] = current.w1;
        }

        let h0: u64 = umul128_hi64(current.w0, ten);
        let h1: u64 = umul128_hi64(current.w1, ten);

        let c0: u64 = h0.wrapping_add(current.w1.wrapping_mul(ten));
        let c1: u64 = ((c0 < h0) as u64 + h1).wrapping_add(current.w2.wrapping_mul(ten));
        let c2: u64 = (c1 < h1) as u64 + umul128_hi64(current.w2, ten); // dodgy carry

        // normalise
        if (c2 >> 63) != 0 {
            current = uint192 {
                w0: c0,
                w1: c1,
                w2: c2,
            };
        } else {
            current = uint192 {
                w0: c0 << 1,
                w1: c1 << 1 | c0 >> 63,
                w2: c2 << 1 | c1 >> 63,
            };
        }

        i += 1;
    }

    Pow10SignificandsTable { data }
};

// Computes the decimal exponent as floor(log10(2**bin_exp)) if regular or
// floor(log10(3/4 * 2**bin_exp)) otherwise, without branching.
const fn compute_dec_exp(bin_exp: i32, regular: bool) -> i32 {
    debug_assert!(bin_exp >= -1334 && bin_exp <= 2620);
    // log10_3_over_4_sig = -log10(3/4) * 2**log10_2_exp rounded to a power of 2
    const LOG10_3_OVER_4_SIG: i32 = 131_072;
    // log10_2_sig = round(log10(2) * 2**log10_2_exp)
    const LOG10_2_SIG: i32 = 315_653;
    const LOG10_2_EXP: i32 = 20;
    (bin_exp * LOG10_2_SIG - !regular as i32 * LOG10_3_OVER_4_SIG) >> LOG10_2_EXP
}

#[inline]
const fn do_compute_exp_shift(bin_exp: i32, dec_exp: i32) -> u8 {
    debug_assert!(dec_exp >= -350 && dec_exp <= 350);
    // log2_pow10_sig = round(log2(10) * 2**log2_pow10_exp) + 1
    const LOG2_POW10_SIG: i32 = 217_707;
    const LOG2_POW10_EXP: i32 = 16;
    // pow10_bin_exp = floor(log2(10**-dec_exp))
    let pow10_bin_exp = (-dec_exp * LOG2_POW10_SIG) >> LOG2_POW10_EXP;
    // pow10 = ((pow10_hi << 64) | pow10_lo) * 2**(pow10_bin_exp - 127)
    (bin_exp + pow10_bin_exp + 1) as u8
}

struct ExpShiftTable {
    data: [u8; if Self::ENABLE {
        f64::EXP_MASK as usize + 1
    } else {
        1
    }],
}

impl ExpShiftTable {
    const ENABLE: bool = true;
}

static EXP_SHIFTS: ExpShiftTable = {
    let mut data = [0u8; if ExpShiftTable::ENABLE {
        f64::EXP_MASK as usize + 1
    } else {
        1
    }];

    let mut raw_exp = 0;
    while raw_exp < data.len() && ExpShiftTable::ENABLE {
        let mut bin_exp = raw_exp as i32 - f64::EXP_OFFSET;
        if raw_exp == 0 {
            bin_exp += 1;
        }
        let dec_exp = compute_dec_exp(bin_exp, true);
        data[raw_exp] = do_compute_exp_shift(bin_exp, dec_exp) as u8;
        raw_exp += 1;
    }

    ExpShiftTable { data }
};

// Computes a shift so that, after scaling by a power of 10, the intermediate
// result always has a fixed 128-bit fractional part (for double).
//
// Different binary exponents can map to the same decimal exponent, but place
// the decimal point at different bit positions. The shift compensates for this.
//
// For example, both 3 * 2**59 and 3 * 2**60 have dec_exp = 2, but dividing by
// 10^dec_exp puts the decimal point in different bit positions:
//   3 * 2**59 / 100 = 1.72...e+16  (needs shift = 1 + 1)
//   3 * 2**60 / 100 = 3.45...e+16  (needs shift = 2 + 1)
#[inline]
unsafe fn compute_exp_shift<UInt, const ONLY_REGULAR: bool>(bin_exp: i32, dec_exp: i32) -> u8
where
    UInt: traits::UInt,
{
    let num_bits = mem::size_of::<UInt>() * 8;
    if num_bits == 64 && ExpShiftTable::ENABLE && ONLY_REGULAR {
        unsafe {
            *EXP_SHIFTS
                .data
                .as_ptr()
                .add((bin_exp + f64::EXP_OFFSET) as usize)
        }
    } else {
        do_compute_exp_shift(bin_exp, dec_exp)
    }
}

#[cfg_attr(feature = "no-panic", no_panic)]
fn count_trailing_nonzeros(x: u64) -> usize {
    // We count the number of bytes until there are only zeros left.
    // The code is equivalent to
    //    8 - x.leading_zeros() / 8
    // but if the BSR instruction is emitted (as gcc on x64 does with default
    // settings), subtracting the constant before dividing allows the compiler
    // to combine it with the subtraction which it inserts due to BSR counting
    // in the opposite direction.
    //
    // Additionally, the BSR instruction requires a zero check. Since the high
    // bit is unused we can avoid the zero check by shifting the datum left by
    // one and inserting a sentinel bit at the end. This can be faster than the
    // automatically inserted range check.
    (70 - ((x.to_le() << 1) | 1).leading_zeros() as usize) / 8
}

// Align data since unaligned access may be slower when crossing a
// hardware-specific boundary.
#[repr(C, align(2))]
struct Digits2([u8; 200]);

static DIGITS2: Digits2 = Digits2(
    *b"0001020304050607080910111213141516171819\
       2021222324252627282930313233343536373839\
       4041424344454647484950515253545556575859\
       6061626364656667686970717273747576777879\
       8081828384858687888990919293949596979899",
);

// Converts value in the range [0, 100) to a string. GCC generates a bit better
// code when value is pointer-size (https://www.godbolt.org/z/5fEPMT1cc).
#[cfg_attr(feature = "no-panic", no_panic)]
unsafe fn digits2(value: usize) -> &'static u16 {
    debug_assert!(value < 100);

    #[allow(clippy::cast_ptr_alignment)]
    unsafe {
        &*DIGITS2.0.as_ptr().cast::<u16>().add(value)
    }
}

const DIV10K_EXP: i32 = 40;
const DIV10K_SIG: u32 = ((1u64 << DIV10K_EXP) / 10000 + 1) as u32;
const NEG10K: u32 = ((1u64 << 32) - 10000) as u32;

const DIV100_EXP: i32 = 19;
const DIV100_SIG: u32 = (1 << DIV100_EXP) / 100 + 1;
const NEG100: u32 = (1 << 16) - 100;

const DIV10_EXP: i32 = 10;
const DIV10_SIG: u32 = (1 << DIV10_EXP) / 10 + 1;
const NEG10: u32 = (1 << 8) - 10;

const ZEROS: u64 = 0x0101010101010101 * b'0' as u64;

#[cfg_attr(feature = "no-panic", no_panic)]
fn to_bcd8(abcdefgh: u64) -> u64 {
    // An optimization from Xiang JunBo.
    // Three steps BCD. Base 10000 -> base 100 -> base 10.
    // div and mod are evaluated simultaneously as, e.g.
    //   (abcdefgh / 10000) << 32 + (abcdefgh % 10000)
    //      == abcdefgh + (2**32 - 10000) * (abcdefgh / 10000)))
    // where the division on the RHS is implemented by the usual multiply + shift
    // trick and the fractional bits are masked away.
    let abcd_efgh =
        abcdefgh + u64::from(NEG10K) * ((abcdefgh * u64::from(DIV10K_SIG)) >> DIV10K_EXP);
    let ab_cd_ef_gh = abcd_efgh
        + u64::from(NEG100) * (((abcd_efgh * u64::from(DIV100_SIG)) >> DIV100_EXP) & 0x7f0000007f);
    let a_b_c_d_e_f_g_h = ab_cd_ef_gh
        + u64::from(NEG10)
            * (((ab_cd_ef_gh * u64::from(DIV10_SIG)) >> DIV10_EXP) & 0xf000f000f000f);
    a_b_c_d_e_f_g_h.to_be()
}

unsafe fn write_if(buffer: *mut u8, digit: u32, condition: bool) -> *mut u8 {
    unsafe {
        *buffer = b'0' + digit as u8;
        buffer.add(usize::from(condition))
    }
}

unsafe fn write8(buffer: *mut u8, value: u64) {
    unsafe {
        buffer.cast::<u64>().write_unaligned(value);
    }
}

// Writes a significand and removes trailing zeros. value has up to 17 decimal
// digits (16-17 for normals) for double (num_bits == 64) and up to 9 digits
// (8-9 for normals) for float. The significant digits start from buffer[1].
// buffer[0] may contain '0' after this function if the leading digit is zero.
#[cfg_attr(feature = "no-panic", no_panic)]
#[inline]
unsafe fn write_significand<Float>(mut buffer: *mut u8, value: u64, extra_digit: bool) -> *mut u8
where
    Float: FloatTraits,
{
    if Float::NUM_BITS == 32 {
        buffer = unsafe { write_if(buffer, (value / 100_000_000) as u32, extra_digit) };
        let bcd = to_bcd8(value % 100_000_000);
        unsafe {
            write8(buffer, bcd + ZEROS);
            return buffer.add(count_trailing_nonzeros(bcd));
        }
    } else if Float::NUM_BITS == 16 {
        buffer = unsafe { write_if(buffer, (value / 100_000_000) as u32, extra_digit) };
        let bcd = to_bcd8(value % 100_000_000);
        unsafe {
            write8(buffer.sub(4), bcd + ZEROS);
            return buffer.sub(4).add(count_trailing_nonzeros(bcd));
        }
    }

        // Digits/pairs of digits are denoted by letters: value = abbccddeeffgghhii.
        let abbccddee = (value / 100_000_000) as u32;
        let ffgghhii = (value % 100_000_000) as u32;
        buffer = unsafe { write_if(buffer, abbccddee / 100_000_000, extra_digit) };
        let bcd = to_bcd8(u64::from(abbccddee % 100_000_000));
        unsafe {
            write8(buffer, bcd + ZEROS);
        }
        if ffgghhii == 0 {
            return unsafe { buffer.add(count_trailing_nonzeros(bcd)) };
        }
        let bcd = to_bcd8(u64::from(ffgghhii));
        unsafe {
            write8(buffer.add(8), bcd + ZEROS);
            buffer.add(8).add(count_trailing_nonzeros(bcd))
        }
}

struct ToDecimalResult {
    sig: i64,
    exp: i32,
}

#[cfg_attr(feature = "no-panic", no_panic)]
#[inline]
fn to_decimal_schubfach<UInt>(bin_sig: UInt, bin_exp: i64, regular: bool) -> ToDecimalResult
where
    UInt: traits::UInt,
{
    let num_bits = mem::size_of::<UInt>() as i32 * 8;
    let dec_exp = compute_dec_exp(bin_exp as i32, regular);
    let exp_shift = unsafe { compute_exp_shift::<UInt, false>(bin_exp as i32, dec_exp) };
    let mut pow10 = unsafe { POW10_SIGNIFICANDS.get_unchecked(-dec_exp) };

    // Fallback to Schubfach to guarantee correctness in boundary cases. This
    // requires switching to strict overestimates of powers of 10.
    if num_bits == 64 {
        pow10.lo += 1;
    } else {
        pow10.hi += 1;
    }

    // Shift the significand so that boundaries are integer.
    const BOUND_SHIFT: u32 = 2;
    let bin_sig_shifted = bin_sig << BOUND_SHIFT;

    // Compute the estimates of lower and upper bounds of the rounding interval
    // by multiplying them by the power of 10 and applying modified rounding.
    let lsb = bin_sig & UInt::from(1);
    let lower = (bin_sig_shifted - (UInt::from(regular) + UInt::from(1))) << exp_shift;
    let lower = umulhi_inexact_to_odd(pow10.hi, pow10.lo, lower) + lsb;
    let upper = (bin_sig_shifted + UInt::from(2)) << exp_shift;
    let upper = umulhi_inexact_to_odd(pow10.hi, pow10.lo, upper) - lsb;

    // The idea of using a single shorter candidate is by Cassio Neri.
    // It is less or equal to the upper bound by construction.
    let shorter = (upper >> BOUND_SHIFT) / UInt::from(10) * UInt::from(10);
    if (shorter << BOUND_SHIFT) >= lower {
        return ToDecimalResult {
            sig: shorter.into() as i64,
            exp: dec_exp,
        };
    }

    let scaled_sig = umulhi_inexact_to_odd(pow10.hi, pow10.lo, bin_sig_shifted << exp_shift);
    let longer_below = scaled_sig >> BOUND_SHIFT;
    let longer_above = longer_below + UInt::from(1);

    // Pick the closest of longer_below and longer_above and check if it's in
    // the rounding interval.
    let cmp = scaled_sig
        .wrapping_sub((longer_below + longer_above) << 1)
        .to_signed();
    let below_closer = cmp < UInt::from(0).to_signed()
        || (cmp == UInt::from(0).to_signed() && (longer_below & UInt::from(1)) == UInt::from(0));
    let below_in = (longer_below << BOUND_SHIFT) >= lower;
    let dec_sig = if below_closer & below_in {
        longer_below
    } else {
        longer_above
    };
    ToDecimalResult {
        sig: dec_sig.into() as i64,
        exp: dec_exp,
    }
}

// Here be 🐉s.
// Converts a binary FP number bin_sig * 2**bin_exp to the shortest decimal
// representation, where bin_exp = raw_exp - exp_offset.
#[cfg_attr(feature = "no-panic", no_panic)]
#[inline]
fn to_decimal_fast<Float, UInt>(bin_sig: UInt, raw_exp: i64, regular: bool) -> ToDecimalResult
where
    Float: FloatTraits,
    UInt: traits::UInt,
{
    let bin_exp = raw_exp - i64::from(Float::EXP_OFFSET);
    let num_bits = mem::size_of::<UInt>() as i32 * 8;
    // An optimization from yy by Yaoyuan Guo:
    while regular {
        let dec_exp = if USE_UMUL128_HI64 {
            umul128_hi64(bin_exp as u64, 0x4d10500000000000) as i32
        } else {
            compute_dec_exp(bin_exp as i32, true)
        };
        let exp_shift = unsafe { compute_exp_shift::<UInt, true>(bin_exp as i32, dec_exp) };
        let pow10 = unsafe { POW10_SIGNIFICANDS.get_unchecked(-dec_exp) };

        let integral; // integral part of bin_sig * pow10
        let fractional; // fractional part of bin_sig * pow10
        if num_bits == 64 {
            let p = umul192_hi128(pow10.hi, pow10.lo, (bin_sig << exp_shift).into());
            integral = UInt::truncate(p.hi);
            fractional = p.lo;
        } else {
            let p = umul128(pow10.hi, (bin_sig << exp_shift).into());
            integral = UInt::truncate((p >> 64) as u64);
            fractional = p as u64;
        }
        const HALF_ULP: u64 = 1 << 63;

        // Exact half-ulp tie when rounding to nearest integer.
        let cmp = fractional.wrapping_sub(HALF_ULP) as i64;
        if cmp == 0 {
            break;
        }

        // An optimization of integral % 10 by Dougall Johnson. Relies on range
        // calculation: (max_bin_sig << max_exp_shift) * max_u128.
        // (1 << 63) / 5 == (1 << 64) / 10 without an intermediate int128.
        const DIV10_SIG64: u64 = (1 << 63) / 5 + 1;
        let div10 = umul128_hi64(integral.into(), DIV10_SIG64);
        #[allow(unused_mut)]
        let mut digit = integral.into() - div10 * 10;
        // or it narrows to 32-bit and doesn't use madd/msub

        // Switch to a fixed-point representation with the least significant
        // integral digit in the upper bits and fractional digits in the lower
        // bits.
        let num_integral_bits = if num_bits == 64 { 4 } else { 32 };
        let num_fractional_bits = 64 - num_integral_bits;
        let ten = 10u64 << num_fractional_bits;
        // Fixed-point remainder of the scaled significand modulo 10.
        let scaled_sig_mod10 = (digit << num_fractional_bits) | (fractional >> num_integral_bits);

        // scaled_half_ulp = 0.5 * pow10 in the fixed-point format.
        // dec_exp is chosen so that 10**dec_exp <= 2**bin_exp < 10**(dec_exp + 1).
        // Since 1ulp == 2**bin_exp it will be in the range [1, 10) after scaling
        // by 10**dec_exp. Add 1 to combine the shift with division by two.
        let scaled_half_ulp = pow10.hi >> (num_integral_bits - exp_shift + 1);
        let upper = scaled_sig_mod10 + scaled_half_ulp;

        // value = 5.0507837461e-27
        // next  = 5.0507837461000010e-27
        //
        // c = integral.fractional' = 50507837461000003.153987... (value)
        //                            50507837461000010.328635... (next)
        //          scaled_half_ulp =                 3.587324...
        //
        // fractional' = fractional / 2**64, fractional = 2840565642863009226
        //
        //      50507837461000000       c               upper     50507837461000010
        //              s              l|   L             |               S
        // ───┬────┬────┼────┬────┬────┼*-──┼────┬────┬───*┬────┬────┬────┼-*--┬───
        //    8    9    0    1    2    3    4    5    6    7    8    9    0 |  1
        //            └─────────────────┼─────────────────┘                next
        //                             1ulp
        //
        // s - shorter underestimate, S - shorter overestimate
        // l - longer underestimate,  L - longer overestimate

        // Check for boundary case when rounding down to nearest 10 and
        // near-boundary case when rounding up to nearest 10.
        // Case where upper == ten is insufficient: 1.342178e+08f.
        if ten.wrapping_sub(upper) <= 1 // upper == ten || upper == ten - 1
            || scaled_sig_mod10 == scaled_half_ulp
        {
            break;
        }

        let shorter = (integral.into() - digit) as i64;
        let longer = (integral.into() + u64::from(cmp >= 0)) as i64;
        let dec_sig = select_if_less(scaled_sig_mod10, scaled_half_ulp, shorter, longer);
        return ToDecimalResult {
            sig: select_if_less(ten, upper, shorter + 10, dec_sig),
            exp: dec_exp,
        };
    }
    to_decimal_schubfach(bin_sig, bin_exp, regular)
}

/// Writes the shortest correctly rounded decimal representation of `value` to
/// `buffer`. `buffer` should point to a buffer of size `buffer_size` or larger.
#[cfg_attr(feature = "no-panic", no_panic)]
unsafe fn write<Float>(value: Float, mut buffer: *mut u8) -> *mut u8
where
    Float: FloatTraits,
{
    let bits = value.to_bits();
    // It is beneficial to extract exponent and significand early.
    let bin_exp = Float::get_exp(bits); // binary exponent
    let bin_sig = Float::get_sig(bits); // binary significand

    unsafe {
        *buffer = b'-';
    }
    buffer = unsafe { buffer.add(usize::from(Float::is_negative(bits))) };

    let mut dec;
    let threshold = if Float::NUM_BITS == 64 {
        10_000_000_000_000_000
    } else if Float::NUM_BITS == 32 {
        100_000_000
    } else {
        10_000
    };
    if bin_exp == 0 {
        if bin_sig == Float::SigType::from(0) {
            return unsafe {
                *buffer = b'0';
                *buffer.add(1) = b'.';
                *buffer.add(2) = b'0';
                buffer.add(3)
            };
        }
        dec = to_decimal_schubfach(bin_sig, i64::from(1 - Float::EXP_OFFSET), true);
        while dec.sig < threshold {
            dec.sig *= 10;
            dec.exp -= 1;
        }
    } else {
        dec = to_decimal_fast::<Float, Float::SigType>(
            bin_sig | Float::IMPLICIT_BIT,
            bin_exp,
            bin_sig != Float::SigType::from(0),
        );
    }
    let mut dec_exp = dec.exp;
    let extra_digit = dec.sig >= threshold;
    dec_exp += Float::MAX_DIGITS10 as i32 - 2 + i32::from(extra_digit);
    if (Float::NUM_BITS == 32 && dec.sig < 10_000_000) || (Float::NUM_BITS == 16 && dec.sig < 1_000) {
        dec.sig *= 10;
        dec_exp -= 1;
    }

    // Write significand.
    let end = unsafe { write_significand::<Float>(buffer.add(1), dec.sig as u64, extra_digit) };

    let length = unsafe { end.offset_from(buffer.add(1)) } as usize;

    if Float::NUM_BITS == 32 && (-6..=12).contains(&dec_exp)
        || Float::NUM_BITS == 64 && (-5..=15).contains(&dec_exp)
        || Float::NUM_BITS == 16 // TODO: figure out the range
    {
        if length as i32 - 1 <= dec_exp {
            // 1234e7 -> 12340000000.0
            return unsafe {
                ptr::copy(buffer.add(1), buffer, length);
                ptr::write_bytes(buffer.add(length), b'0', dec_exp as usize + 3 - length);
                *buffer.add(dec_exp as usize + 1) = b'.';
                buffer.add(dec_exp as usize + 3)
            };
        } else if 0 <= dec_exp {
            // 1234e-2 -> 12.34
            return unsafe {
                ptr::copy(buffer.add(1), buffer, dec_exp as usize + 1);
                *buffer.add(dec_exp as usize + 1) = b'.';
                buffer.add(length + 1)
            };
        } else {
            // 1234e-6 -> 0.001234
            return unsafe {
                ptr::copy(buffer.add(1), buffer.add((1 - dec_exp) as usize), length);
                ptr::write_bytes(buffer, b'0', (1 - dec_exp) as usize);
                *buffer.add(1) = b'.';
                buffer.add((1 - dec_exp) as usize + length)
            };
        }
    }

    unsafe {
        // 1234e30 -> 1.234e33
        *buffer = *buffer.add(1);
        *buffer.add(1) = b'.';
    }
    buffer = unsafe { buffer.add(length + usize::from(length > 1)) };

    // Write exponent.
    let sign_ptr = buffer;
    let e_sign = if dec_exp >= 0 {
        (u16::from(b'+') << 8) | u16::from(b'e')
    } else {
        (u16::from(b'-') << 8) | u16::from(b'e')
    };
    buffer = unsafe { buffer.add(1) };
    dec_exp = if dec_exp >= 0 { dec_exp } else { -dec_exp };
    buffer = unsafe { buffer.add(usize::from(dec_exp >= 10)) };
    if Float::MIN_10_EXP > -100 && Float::MAX_10_EXP < 100 {
        unsafe {
            buffer
                .cast::<u16>()
                .write_unaligned(*digits2(dec_exp as usize));
            sign_ptr.cast::<u16>().write_unaligned(e_sign.to_le());
            return buffer.add(2);
        }
    }
    // digit = dec_exp / 100
    let digit = if USE_UMUL128_HI64 {
        umul128_hi64(dec_exp as u64, 0x290000000000000) as u32
    } else {
        (dec_exp as u32 * DIV100_SIG) >> DIV100_EXP
    };
    unsafe {
        *buffer = b'0' + digit as u8;
    }
    buffer = unsafe { buffer.add(usize::from(dec_exp >= 100)) };
    unsafe {
        buffer
            .cast::<u16>()
            .write_unaligned(*digits2((dec_exp as u32 - digit * 100) as usize));
        sign_ptr.cast::<u16>().write_unaligned(e_sign.to_le());
        buffer.add(2)
    }
}

/// Safe API for formatting floating point numbers to text.
///
/// ## Example
///
/// ```
/// let mut buffer = zmij::Buffer::new();
/// let printed = buffer.format_finite(1.234);
/// assert_eq!(printed, "1.234");
/// ```
pub struct Buffer {
    bytes: [MaybeUninit<u8>; BUFFER_SIZE],
}

impl Buffer {
    /// This is a cheap operation; you don't need to worry about reusing buffers
    /// for efficiency.
    #[inline]
    #[cfg_attr(feature = "no-panic", no_panic)]
    pub fn new() -> Self {
        let bytes = [MaybeUninit::<u8>::uninit(); BUFFER_SIZE];
        Buffer { bytes }
    }

    /// Print a floating point number into this buffer and return a reference to
    /// its string representation within the buffer.
    ///
    /// # Special cases
    ///
    /// This function formats NaN as the string "NaN", positive infinity as
    /// "inf", and negative infinity as "-inf" to match std::fmt.
    ///
    /// If your input is known to be finite, you may get better performance by
    /// calling the `format_finite` method instead of `format` to avoid the
    /// checks for special cases.
    #[cfg_attr(feature = "no-panic", no_panic)]
    pub fn format<F: Float>(&mut self, f: F) -> &str {
        if f.is_nonfinite() {
            f.format_nonfinite()
        } else {
            self.format_finite(f)
        }
    }

    /// Print a floating point number into this buffer and return a reference to
    /// its string representation within the buffer.
    ///
    /// # Special cases
    ///
    /// This function **does not** check for NaN or infinity. If the input
    /// number is not a finite float, the printed representation will be some
    /// correctly formatted but unspecified numerical value.
    ///
    /// Please check [`is_finite`] yourself before calling this function, or
    /// check [`is_nan`] and [`is_infinite`] and handle those cases yourself.
    ///
    /// [`is_finite`]: f64::is_finite
    /// [`is_nan`]: f64::is_nan
    /// [`is_infinite`]: f64::is_infinite
    #[cfg_attr(feature = "no-panic", no_panic)]
    pub fn format_finite<F: Float>(&mut self, f: F) -> &str {
        unsafe {
            let end = f.write_to_zmij_buffer(self.bytes.as_mut_ptr().cast::<u8>());
            let len = end.offset_from(self.bytes.as_ptr().cast::<u8>()) as usize;
            let slice = slice::from_raw_parts(self.bytes.as_ptr().cast::<u8>(), len);
            str::from_utf8_unchecked(slice)
        }
    }
}

/// A floating point number, f32 or f64, that can be written into a
/// [`zmij::Buffer`][Buffer].
///
/// This trait is sealed and cannot be implemented for types outside of the
/// `zmij` crate.
#[allow(unknown_lints)] // rustc older than 1.74
#[allow(private_bounds)]
pub trait Float: private::Sealed {}
impl Float for f16 {}
impl Float for f32 {}
impl Float for f64 {}

mod private {
    pub trait Sealed: crate::traits::Float {
        fn is_nonfinite(self) -> bool;
        fn format_nonfinite(self) -> &'static str;
        unsafe fn write_to_zmij_buffer(self, buffer: *mut u8) -> *mut u8;
    }

    impl Sealed for f16 {
        #[inline]
        fn is_nonfinite(self) -> bool {
            const EXP_MASK: u16 = 0x7B00;
            let bits = self.to_bits();
            bits & EXP_MASK == EXP_MASK
        }

        #[cold]
        #[cfg_attr(feature = "no-panic", inline)]
        fn format_nonfinite(self) -> &'static str {
            const MANTISSA_MASK: u16 = 0x03ff;
            const SIGN_MASK: u16 = 0x8000;
            let bits = self.to_bits();
            if bits & MANTISSA_MASK != 0 {
                crate::NAN
            } else if bits & SIGN_MASK != 0 {
                crate::NEG_INFINITY
            } else {
                crate::INFINITY
            }
        }

        #[cfg_attr(feature = "no-panic", inline)]
        unsafe fn write_to_zmij_buffer(self, buffer: *mut u8) -> *mut u8 {
            unsafe { crate::write(self, buffer) }
        }
    }

    impl Sealed for f32 {
        #[inline]
        fn is_nonfinite(self) -> bool {
            const EXP_MASK: u32 = 0x7f800000;
            let bits = self.to_bits();
            bits & EXP_MASK == EXP_MASK
        }

        #[cold]
        #[cfg_attr(feature = "no-panic", inline)]
        fn format_nonfinite(self) -> &'static str {
            const MANTISSA_MASK: u32 = 0x007fffff;
            const SIGN_MASK: u32 = 0x80000000;
            let bits = self.to_bits();
            if bits & MANTISSA_MASK != 0 {
                crate::NAN
            } else if bits & SIGN_MASK != 0 {
                crate::NEG_INFINITY
            } else {
                crate::INFINITY
            }
        }

        #[cfg_attr(feature = "no-panic", inline)]
        unsafe fn write_to_zmij_buffer(self, buffer: *mut u8) -> *mut u8 {
            unsafe { crate::write(self, buffer) }
        }
    }

    impl Sealed for f64 {
        #[inline]
        fn is_nonfinite(self) -> bool {
            const EXP_MASK: u64 = 0x7ff0000000000000;
            let bits = self.to_bits();
            bits & EXP_MASK == EXP_MASK
        }

        #[cold]
        #[cfg_attr(feature = "no-panic", inline)]
        fn format_nonfinite(self) -> &'static str {
            const MANTISSA_MASK: u64 = 0x000fffffffffffff;
            const SIGN_MASK: u64 = 0x8000000000000000;
            let bits = self.to_bits();
            if bits & MANTISSA_MASK != 0 {
                crate::NAN
            } else if bits & SIGN_MASK != 0 {
                crate::NEG_INFINITY
            } else {
                crate::INFINITY
            }
        }

        #[cfg_attr(feature = "no-panic", inline)]
        unsafe fn write_to_zmij_buffer(self, buffer: *mut u8) -> *mut u8 {
            unsafe { crate::write(self, buffer) }
        }
    }
}

impl Default for Buffer {
    #[inline]
    #[cfg_attr(feature = "no-panic", no_panic)]
    fn default() -> Self {
        Buffer::new()
    }
}
