use core::ffi::CStr;
use core::mem::MaybeUninit;

fn dtoa(value: f64) -> String {
    let mut buffer = MaybeUninit::<[u8; zmij::BUFFER_SIZE]>::uninit();
    unsafe {
        zmij::dtoa(value, buffer.as_mut_ptr().cast::<u8>());
        CStr::from_ptr(buffer.as_ptr().cast::<i8>())
            .to_str()
            .unwrap()
            .to_owned()
    }
}

#[test]
fn normal() {
    assert_eq!(dtoa(6.62607015e-34), "6.62607015e-34");
}

#[test]
fn zero() {
    assert_eq!(dtoa(0.0), "0");
    assert_eq!(dtoa(-0.0), "-0");
}

#[test]
fn inf() {
    assert_eq!(dtoa(f64::INFINITY), "inf");
    assert_eq!(dtoa(f64::NEG_INFINITY), "-inf");
}

#[test]
fn shorter() {
    // A possibly shorter underestimate is picked (u' in Schubfach).
    assert_eq!(dtoa(-4.932096661796888e-226), "-4.932096661796888e-226");

    // A possibly shorter overestimate is picked (w' in Schubfach).
    assert_eq!(dtoa(3.439070283483335e+35), "3.439070283483335e+35");
}

#[test]
fn single_candidate() {
    // Only an underestimate is in the rounding region (u in Schubfach).
    assert_eq!(dtoa(6.606854224493745e-17), "6.606854224493745e-17");

    // Only an overestimate is in the rounding region (w in Schubfach).
    assert_eq!(dtoa(6.079537928711555e+61), "6.079537928711555e+61");
}
