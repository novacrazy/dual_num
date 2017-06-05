//! Dual Numbers
//!
//! This is a dual number implementation scavenged from other dual number libraries and articles around the web, including:
//!
//! * [https://github.com/FreeFull/dual_numbers](https://github.com/FreeFull/dual_numbers)
//! * [https://github.com/ibab/rust-ad](https://github.com/ibab/rust-ad)
//! * [https://github.com/tesch1/cxxduals](https://github.com/tesch1/cxxduals)
//!
//! The difference being is that I have checked each method against Wolfram Alpha for correctness and will
//! keep this implementation up to date and working with the latest stable Rust and `num-traits` crate.
//!
//! ## Usage
//!
//! ```rust
//! extern crate dual_num;
//!
//! use dual_num::{DualNumber, Float, differentiate};
//!
//! fn test<F: Float>(x: F) -> F {
//!     x.sqrt() + F::from(1.0).unwrap()
//! }
//!
//! fn main() {
//!     // find partial derivative at x=4.0
//!     let result = differentiate(4.0f64, test);
//!
//!     println!("{:.5}", result); // 0.25000
//! }
//! ```

// Note that the somewhat excessive #[inline] annotations are not harmful here,
// and can improve cross-crate inlining.
//
// Also, for clarity I've avoiding using .0 and .1 outside of the struct impl.
// They're even made private to encourage using .real() and .dual() instead.

extern crate num_traits;

use std::ops::{Add, Sub, Mul, Div, Rem, Neg};
use std::cmp::Ordering;
use std::num::FpCategory;
use std::fmt::{Display, Formatter, Result as FmtResult};

pub use num_traits::{One, Zero, Float, Num, NumCast, ToPrimitive};

/// Dual Number structure
///
/// Although `DualNumber` does implement `PartialEq` and `PartialOrd`,
/// it only compares the real part.
///
/// Additionally, `min` and `max` only compare the real parts, and keep the dual parts.
#[derive(Debug, Clone, Copy)]
pub struct DualNumber<T: Float>(T, T);

/// Evaluates the function using dual numbers to get the partial derivative at the input point
pub fn differentiate<T: Float, F>(x: T, f: F) -> T where F: Fn(DualNumber<T>) -> DualNumber<T> {
    f(DualNumber::new(x, T::one())).dual()
}

impl<T: Float> DualNumber<T> {
    /// Create a new dual number from its real and dual parts
    #[inline]
    pub fn new(real: T, dual: T) -> DualNumber<T> {
        DualNumber(real, dual)
    }

    /// Create a new dual number from a real number.
    ///
    /// The dual part is set to zero.
    #[inline]
    pub fn from_real(real: T) -> DualNumber<T> {
        DualNumber::new(real, T::zero())
    }

    /// Returns the real part
    #[inline(always)]
    pub fn real(&self) -> T { self.0 }

    /// Returns the dual part
    #[inline(always)]
    pub fn dual(&self) -> T { self.1 }

    /// Returns both real and dual parts as a tuple
    #[inline]
    pub fn into_tuple(self) -> (T, T) {
        (self.0, self.1)
    }

    /// Returns a mutable reference to the real part
    #[inline]
    pub fn real_mut(&mut self) -> &mut T { &mut self.0 }

    /// Returns a mutable reference to the dual part
    #[inline]
    pub fn dual_mut(&mut self) -> &mut T { &mut self.1 }

    /// Returns the conjugate of the dual number.
    pub fn conjugate(self) -> Self {
        DualNumber(self.real(), self.dual().neg())
    }
}

impl<T: Float> Display for DualNumber<T> where T: Display {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let precision = f.precision().unwrap_or(2);

        write!(f, "{:.p$} + \u{03B5}{:.p$}", self.real(), self.dual(), p = precision)
    }
}

impl<T: Float> PartialEq<Self> for DualNumber<T> {
    fn eq(&self, rhs: &Self) -> bool {
        self.0 == rhs.0
    }
}

impl<T: Float> PartialOrd<Self> for DualNumber<T> {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.0, &rhs.0)
    }

    fn lt(&self, rhs: &Self) -> bool { self.real() < rhs.real() }
    fn le(&self, rhs: &Self) -> bool { self.real() <= rhs.real() }
    fn gt(&self, rhs: &Self) -> bool { self.real() > rhs.real() }
    fn ge(&self, rhs: &Self) -> bool { self.real() >= rhs.real() }
}

impl<T: Float> PartialEq<T> for DualNumber<T> {
    fn eq(&self, rhs: &T) -> bool {
        self.0 == *rhs
    }
}

impl<T: Float> PartialOrd<T> for DualNumber<T> {
    fn partial_cmp(&self, rhs: &T) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.0, rhs)
    }

    fn lt(&self, rhs: &T) -> bool { self.real() < *rhs }
    fn le(&self, rhs: &T) -> bool { self.real() <= *rhs }
    fn gt(&self, rhs: &T) -> bool { self.real() > *rhs }
    fn ge(&self, rhs: &T) -> bool { self.real() >= *rhs }
}

macro_rules! impl_to_primitive {
    ($($name:ident, $ty:ty),*) => {
        impl<T: Float> ToPrimitive for DualNumber<T> {
            $(
                fn $name(&self) -> Option<$ty> {
                    self.real().$name()
                }
            )*
        }
    }
}

impl_to_primitive!(to_isize, isize, to_i8, i8, to_i16, i16, to_i32, i32, to_i64, i64,
                   to_usize, usize, to_u8, u8, to_u16, u16, to_u32, u32, to_u64, u64,
                   to_f32, f32, to_f64, f64);

impl<T: Float> Add<T> for DualNumber<T> {
    type Output = DualNumber<T>;

    #[inline]
    fn add(self, rhs: T) -> DualNumber<T> {
        DualNumber::new(self.real() + rhs,
                        self.dual())
    }
}

impl<T: Float> Sub<T> for DualNumber<T> {
    type Output = DualNumber<T>;

    #[inline]
    fn sub(self, rhs: T) -> DualNumber<T> {
        DualNumber::new(self.real() - rhs,
                        self.dual())
    }
}

impl<T: Float> Mul<T> for DualNumber<T> {
    type Output = DualNumber<T>;

    fn mul(self, rhs: T) -> DualNumber<T> {
        self * DualNumber::from_real(rhs)
    }
}

impl<T: Float> Div<T> for DualNumber<T> {
    type Output = DualNumber<T>;

    #[inline]
    fn div(self, rhs: T) -> DualNumber<T> {
        self / DualNumber::from_real(rhs)
    }
}

impl<T: Float> Neg for DualNumber<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        DualNumber::new(self.real().neg(),
                        self.dual().neg())
    }
}

impl<T: Float> Add<Self> for DualNumber<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        DualNumber::new(self.real() + rhs.real(),
                        self.dual() + rhs.dual())
    }
}

impl<T: Float> Sub<Self> for DualNumber<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        DualNumber::new(self.real() - rhs.real(),
                        self.dual() - rhs.dual())
    }
}

impl<T: Float> Mul<Self> for DualNumber<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        DualNumber::new(
            self.real() * rhs.real(),
            self.real() * rhs.dual() + self.dual() * rhs.real()
        )
    }
}

impl<T: Float> Div<Self> for DualNumber<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        DualNumber::new(
            self.real() / rhs.real(),
            (self.dual() * rhs.real() - self.real() * rhs.dual()) / (rhs.real() * rhs.real())
        )
    }
}

impl<T: Float> Rem<Self> for DualNumber<T> {
    type Output = Self;

    /// **UNIMPLEMENTED!!!**
    ///
    /// As far as I know, remainder is not a valid operation on dual numbers,
    /// but is required for the `Float` trait to be implemented.
    fn rem(self, _: Self) -> Self {
        unimplemented!()
    }
}

impl<T: Float> Zero for DualNumber<T> {
    #[inline]
    fn zero() -> Self {
        DualNumber::new(T::zero(), T::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.real().is_zero()
    }
}

impl<T: Float> One for DualNumber<T> {
    #[inline]
    fn one() -> Self {
        DualNumber::new(T::one(), T::zero())
    }
}

impl<T: Float> Num for DualNumber<T> {
    type FromStrRadixErr = <T as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        <T as Num>::from_str_radix(str, radix)
            .map(DualNumber::from_real)
    }
}

impl<T: Float> NumCast for DualNumber<T> {
    #[inline]
    fn from<N: ToPrimitive>(n: N) -> Option<Self> {
        <T as NumCast>::from(n)
            .map(DualNumber::from_real)
    }
}

macro_rules! impl_real_constant {
    ($($prop:ident),*) => {
        $(
            #[inline]
            fn $prop() -> Self { DualNumber::from_real(<T as Float>::$prop()) }
        )*
    }
}

macro_rules! impl_single_boolean_op {
    ($op:ident REAL) => {
        #[inline]
        fn $op(self) -> bool {self.real().$op()}
    };
    ($op:ident OR) =>   { fn $op(self) -> bool {self.real().$op() || self.dual().$op()} };
    ($op:ident AND) =>  { fn $op(self) -> bool {self.real().$op() && self.dual().$op()} };
}

macro_rules! impl_boolean_op {
    ($($op:ident $t:tt),*) => {
        $(impl_single_boolean_op!($op $t);)*
    };
}

macro_rules! impl_real_op {
    ($($op:ident),*) => {
        #[inline]
        $(fn $op(self) -> Self { DualNumber::new(self.real().$op(), T::zero()) })*
    }
}

impl<T: Float> Float for DualNumber<T> {
    impl_real_constant!(
        nan,
        infinity,
        neg_infinity,
        neg_zero,
        min_positive_value,
        epsilon,
        min_value,
        max_value
    );

    impl_boolean_op!(
        is_nan              OR,
        is_infinite         OR,
        is_finite           AND,
        is_normal           AND,
        is_sign_positive    REAL,
        is_sign_negative    REAL
    );

    fn classify(self) -> FpCategory {
        self.real().classify()
    }

    impl_real_op!(
        floor,
        ceil,
        round,
        trunc
    );

    fn fract(self) -> Self {
        DualNumber::new(self.real().fract(), self.dual())
    }

    fn signum(self) -> Self {
        DualNumber::from_real(self.real().signum())
    }

    fn abs(self) -> Self {
        DualNumber::new(self.real().abs(), self.dual() * self.real().signum())
    }

    fn max(self, other: Self) -> Self {
        if self.real() > other.real() { self } else { other }
    }

    fn min(self, other: Self) -> Self {
        if self.real() < other.real() { other } else { self }
    }

    fn abs_sub(self, rhs: Self) -> Self {
        if self.real() > rhs.real() {
            DualNumber::new(self.real() - rhs.real(), (self - rhs).dual())
        } else {
            Self::zero()
        }
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        DualNumber::new(self.real().mul_add(a.real(), b.real()),
                        self.dual() * a.real() + self.real() * a.dual() + b.dual())
    }

    #[inline]
    fn recip(self) -> Self {
        Self::one() / self
    }

    fn powi(self, n: i32) -> Self {
        let nf = <T as NumCast>::from(n).expect("Invalid value");

        DualNumber::new(self.real().powi(n),
                        nf * self.real().powi(n - 1) * self.dual())
    }

    fn powf(self, n: Self) -> Self {
        let real = self.real().powf(n.real());

        DualNumber::new(real,
                        n.real() * self.real().powf(n.real() - T::one()) * self.dual() +
                            real * self.real().ln() * n.dual())
    }

    fn exp(self) -> Self {
        let real = self.real().exp();

        DualNumber::new(real, self.dual() * real)
    }

    fn exp2(self) -> Self {
        let ln_2 = <T as NumCast>::from(::std::f64::consts::LN_2).expect("Invalid cast from f64");

        let real = self.real().exp2();

        DualNumber::new(real, self.dual() * ln_2 * real)
    }

    fn ln(self) -> Self {
        DualNumber::new(self.real().ln(), self.dual() / self.real())
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    #[inline]
    fn log2(self) -> Self {
        let ln_2 = <T as NumCast>::from(::std::f64::consts::LN_2).expect("Invalid cast from f64");

        DualNumber::new(self.real().log10(), self.dual() / (self.real() * ln_2))
    }

    #[inline]
    fn log10(self) -> Self {
        let ln_10 = <T as NumCast>::from(::std::f64::consts::LN_10).expect("Invalid cast from f64");

        DualNumber::new(self.real().log10(), self.dual() / (self.real() * ln_10))
    }

    #[inline]
    fn sqrt(self) -> Self {
        let real = self.real().sqrt();

        DualNumber::new(real, self.dual() / (T::from(2).unwrap() * real))
    }

    #[inline]
    fn cbrt(self) -> Self {
        let real = self.real().cbrt();

        DualNumber::new(real, self.dual() / (T::from(3).unwrap() * real))
    }

    fn hypot(self, other: Self) -> Self {
        let real = self.real().hypot(other.real());

        DualNumber::new(real, (self.real() * other.dual() + other.real() * self.dual()) / real)
    }

    fn sin(self) -> Self { DualNumber::new(self.real().sin(), self.dual() * self.real().cos()) }
    fn cos(self) -> Self { DualNumber::new(self.real().cos(), self.dual().neg() * self.real().sin()) }

    fn tan(self) -> Self {
        let t = self.real().tan();

        DualNumber::new(t, self.dual() * (t * t + T::one()))
    }

    fn asin(self) -> Self { DualNumber::new(self.real().asin(), self.dual() / (T::one() - self.real().powi(2)).sqrt()) }
    fn acos(self) -> Self { DualNumber::new(self.real().acos(), self.dual().neg() / (T::one() - self.real().powi(2)).sqrt()) }
    fn atan(self) -> Self { DualNumber::new(self.real().atan(), self.dual() / (self.real().powi(2) + T::one()).sqrt()) }

    fn atan2(self, other: Self) -> Self {
        DualNumber::new(
            self.real().atan2(other.real()),
            (other.real() * self.dual() - self.real() * other.dual()) /
                (self.real().powi(2) + other.real().powi(2))
        )
    }

    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.real().sin_cos();

        let sn = DualNumber::new(s, self.dual() * c);
        let cn = DualNumber::new(c, self.dual().neg() * s);

        (sn, cn)
    }

    fn exp_m1(self) -> Self { DualNumber::new(self.real().exp_m1(), self.dual() * self.real().exp()) }

    fn ln_1p(self) -> Self { DualNumber::new(self.real().ln_1p(), self.dual() / (self.real() + T::one())) }

    fn sinh(self) -> Self { DualNumber::new(self.real().sinh(), self.dual() * self.real().cosh()) }
    fn cosh(self) -> Self { DualNumber::new(self.real().cosh(), self.dual() * self.real().sinh()) }

    fn tanh(self) -> Self {
        let real = self.real().tanh();

        DualNumber::new(real, self.dual() * (T::one() - real.powi(2)))
    }

    fn asinh(self) -> Self { DualNumber::new(self.real().asinh(), self.dual() / (self.real().powi(2) + T::one()).sqrt()) }

    fn acosh(self) -> Self {
        DualNumber::new(self.real().acosh(),
                        self.dual() /
                            ((self.real() + T::one()).sqrt() *
                                (self.real() - T::one()).sqrt()))
    }

    fn atanh(self) -> Self { DualNumber::new(self.real().atanh(), self.dual() / (T::one() - self.real().powi(2))) }

    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) { self.real().integer_decode() }

    #[inline]
    fn to_degrees(self) -> Self { DualNumber::from_real(self.real().to_degrees()) }

    #[inline]
    fn to_radians(self) -> Self { DualNumber::from_real(self.real().to_radians()) }
}