#[test]
fn utilities() {
    let count_lzero = u64::leading_zeros;
    assert_eq!(count_lzero(0), 64);
    assert_eq!(count_lzero(1), 63);
    assert_eq!(count_lzero(!0), 0);

    assert_eq!(crate::count_trailing_nonzeros(0x30303030_30303030), 0);
    assert_eq!(crate::count_trailing_nonzeros(0x30303030_30303031), 1);
    assert_eq!(crate::count_trailing_nonzeros(0x30303030_30303039), 1);
    assert_eq!(crate::count_trailing_nonzeros(0x30393030_39303030), 7);
    assert_eq!(crate::count_trailing_nonzeros(0x31303030_30303030), 8);
    assert_eq!(crate::count_trailing_nonzeros(0x39303030_30303030), 8);
}

#[test]
fn umul192_upper64_inexact_to_odd() {
    let (hi, lo) = crate::POW10_SIGNIFICANDS[0];
    assert_eq!(
        crate::umul192_upper64_inexact_to_odd(hi, lo, 0x1234567890abcdef << 1),
        0x24554a3ce60a45f5,
    );
    assert_eq!(
        crate::umul192_upper64_inexact_to_odd(hi, lo, 0x1234567890abce16 << 1),
        0x24554a3ce60a4643,
    );
}
