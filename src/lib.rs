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
//! extern crate hyperdual;
//!
//! use hyperdual::{Dual, Float, differentiate};
//!
//! fn main() {
//!     // find partial derivative at x=4.0
//!     println!("{:.5}", differentiate(4.0f64, |x| {
//!         x.sqrt() + Dual::from_real(1.0)
//!     })); // 0.25000
//! }
//! ```

// Note that the somewhat excessive #[inline] annotations are not harmful here,
// and can improve cross-crate inlining.
//
// Also, for clarity I've avoiding using .0 and .1 outside of the struct impl.
// They're even made private to encourage using .real() and .dual() instead.

extern crate nalgebra as na;
extern crate num_traits;

use std::cmp::Ordering;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::num::FpCategory;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

pub use num_traits::{Float, FloatConst, Num, One, Zero};

// Re-export the differential functions
mod differentials;
pub use differentials::*;

// Re-export the linear algebra helpers
pub mod linalg;

use num_traits::{FromPrimitive, NumCast, Signed, ToPrimitive, Unsigned};

/// Dual Number structure
///
/// Although `Dual` does implement `PartialEq` and `PartialOrd`,
/// it only compares the real part.
///
/// Additionally, `min` and `max` only compare the real parts, and keep the dual parts.
///
/// Lastly, the `Rem` remainder operator is not correctly or fully defined for `Dual`, and will panic.
#[derive(Debug, Clone, Copy)]
pub struct Dual<T>(T, T);

impl<T> Dual<T> {
    /// Create a new dual number from its real and dual parts.
    pub fn new(real: T, dual: T) -> Dual<T> {
        Dual(real, dual)
    }

    /// Create a new dual number from a real number.
    ///
    /// The dual part is set to zero.
    pub fn from_real(real: T) -> Dual<T>
    where
        T: Zero,
    {
        Dual::new(real, T::zero())
    }

    /// Returns both real and dual parts as a tuple
    pub fn into_tuple(self) -> (T, T) {
        (self.0, self.1)
    }

    /// Returns a reference to the real part
    pub fn real_ref(&self) -> &T {
        &self.0
    }

    /// Returns a reference to the dual part
    pub fn dual_ref(&self) -> &T {
        &self.1
    }

    /// Returns a mutable reference to the real part
    pub fn real_mut(&mut self) -> &mut T {
        &mut self.0
    }

    /// Returns a mutable reference to the dual part
    pub fn dual_mut(&mut self) -> &mut T {
        &mut self.1
    }

    /// Convenience method to take a closure (or any function) that can operate on the dual number in place
    pub fn map<F>(self, mapper: F) -> Self
    where
        F: Fn(Dual<T>) -> Dual<T>,
    {
        mapper(self)
    }

    /// Convenience method to take a closure (or any function) that can operate on the dual number parts in place
    pub fn map_parts<F>(self, mapper: F) -> Self
    where
        F: Fn(T, T) -> Dual<T>,
    {
        mapper(self.0, self.1)
    }
}

impl<T: Zero> From<T> for Dual<T> {
    fn from(real: T) -> Dual<T> {
        Dual::from_real(real)
    }
}

impl<T: Copy> Dual<T> {
    /// Returns the real part
    pub fn real(&self) -> T {
        self.0
    }

    /// Returns the dual part
    pub fn dual(&self) -> T {
        self.1
    }
}

impl<T: Float> Dual<T> {
    /// Returns the conjugate of the dual number.
    pub fn conjugate(self) -> Self {
        Dual(self.real(), self.dual().neg())
    }
}

impl<T: Display> Display for Dual<T> {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let precision = f.precision().unwrap_or(2);

        write!(f, "{:.p$} + \u{03B5}{:.p$}", self.0, self.1, p = precision)
    }
}

impl<T: PartialEq> PartialEq<Self> for Dual<T> {
    fn eq(&self, rhs: &Self) -> bool {
        self.0 == rhs.0 && self.1 == rhs.1
    }
}

impl<T: PartialOrd> PartialOrd<Self> for Dual<T> {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(self.real_ref(), rhs.real_ref())
    }

    fn lt(&self, rhs: &Self) -> bool {
        self.0 < rhs.0
    }
    fn gt(&self, rhs: &Self) -> bool {
        self.0 > rhs.0
    }
}

impl<T: PartialEq> PartialEq<T> for Dual<T> {
    fn eq(&self, rhs: &T) -> bool {
        self.0 == *rhs
    }
}

impl<T: PartialOrd> PartialOrd<T> for Dual<T> {
    fn partial_cmp(&self, rhs: &T) -> Option<Ordering> {
        PartialOrd::partial_cmp(self.real_ref(), rhs)
    }

    fn lt(&self, rhs: &T) -> bool {
        self.0 < *rhs
    }
    fn gt(&self, rhs: &T) -> bool {
        self.0 > *rhs
    }
}

macro_rules! impl_to_primitive {
    ($($name:ident, $ty:ty),*) => {
        impl<T: ToPrimitive> ToPrimitive for Dual<T> {
            $(
                fn $name(&self) -> Option<$ty> {
                    (self.0).$name()
                }
            )*
        }
    }
}

macro_rules! impl_from_primitive {
    ($($name:ident, $ty:ty),*) => {
        impl<T: FromPrimitive> FromPrimitive for Dual<T> where T: Zero {
            $(
                fn $name(n: $ty) -> Option<Dual<T>> {
                    T::$name(n).map(Dual::from_real)
                }
            )*
        }
    }
}

macro_rules! impl_primitive_cast {
    ($($to:ident, $from:ident - $ty:ty),*) => {
        impl_to_primitive!($($to, $ty),*);
        impl_from_primitive!($($from, $ty),*);
    }
}

impl_primitive_cast!(
    to_isize,
    from_isize - isize,
    to_i8,
    from_i8 - i8,
    to_i16,
    from_i16 - i16,
    to_i32,
    from_i32 - i32,
    to_i64,
    from_i64 - i64,
    to_usize,
    from_usize - usize,
    to_u8,
    from_u8 - u8,
    to_u16,
    from_u16 - u16,
    to_u32,
    from_u32 - u32,
    to_u64,
    from_u64 - u64,
    to_f32,
    from_f32 - f32,
    to_f64,
    from_f64 - f64
);

impl<T: Num + Copy> Add<T> for Dual<T> {
    type Output = Dual<T>;

    fn add(self, rhs: T) -> Dual<T> {
        Dual::new(self.real() + rhs, self.dual())
    }
}

impl<T: Num + Copy> Sub<T> for Dual<T> {
    type Output = Dual<T>;

    fn sub(self, rhs: T) -> Dual<T> {
        Dual::new(self.real() - rhs, self.dual())
    }
}

impl<T: Num + Copy> Mul<T> for Dual<T> {
    type Output = Dual<T>;

    fn mul(self, rhs: T) -> Dual<T> {
        self * Dual::from_real(rhs)
    }
}

impl<T: Num + Copy> Div<T> for Dual<T> {
    type Output = Dual<T>;

    fn div(self, rhs: T) -> Dual<T> {
        self / Dual::from_real(rhs)
    }
}

impl<T: Signed + Copy> Neg for Dual<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Dual::new(self.real().neg(), self.dual().neg())
    }
}

impl<T: Num + Copy> Add<Self> for Dual<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Dual::new(self.real() + rhs.real(), self.dual() + rhs.dual())
    }
}

impl<T: Num + Copy> Sub<Self> for Dual<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Dual::new(self.real() - rhs.real(), self.dual() - rhs.dual())
    }
}

impl<T: Num + Copy> Mul<Self> for Dual<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Dual::new(
            self.real() * rhs.real(),
            self.real() * rhs.dual() + self.dual() * rhs.real(),
        )
    }
}

impl<T: Num + Copy> Div<Self> for Dual<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Dual::new(
            self.real() / rhs.real(),
            (self.dual() * rhs.real() - self.real() * rhs.dual()) / (rhs.real() * rhs.real()),
        )
    }
}

impl<T: Num + Copy> Rem<Self> for Dual<T> {
    type Output = Self;

    /// **UNIMPLEMENTED!!!**
    ///
    /// As far as I know, remainder is not a valid operation on dual numbers,
    /// but is required for the `Float` trait to be implemented.
    fn rem(self, _: Self) -> Self {
        unimplemented!()
    }
}

impl<T> Signed for Dual<T>
where
    T: Signed + Copy + PartialOrd,
{
    fn abs(&self) -> Self {
        Dual::new(self.real().abs(), self.dual() * self.real().signum())
    }

    fn abs_sub(&self, rhs: &Self) -> Self {
        if self.real() > rhs.real() {
            Dual::new(self.real() - rhs.real(), self.sub(*rhs).dual())
        } else {
            Self::zero()
        }
    }

    fn signum(&self) -> Self {
        Dual::from_real(self.real().signum())
    }

    fn is_positive(&self) -> bool {
        self.real().is_positive()
    }

    fn is_negative(&self) -> bool {
        self.real().is_negative()
    }
}

impl<T: Unsigned> Unsigned for Dual<T>
where
    Self: Num,
{
}

impl<T: Num + Zero + Copy> Zero for Dual<T> {
    fn zero() -> Dual<T> {
        Dual::new(T::zero(), T::zero())
    }

    fn is_zero(&self) -> bool {
        self.real().is_zero() && self.dual().is_zero()
    }
}

impl<T: Num + One + Copy> One for Dual<T> {
    fn one() -> Dual<T> {
        Dual::new(T::one(), T::zero())
    }
}

impl<T: Num + Copy> Num for Dual<T> {
    type FromStrRadixErr = <T as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Dual<T>, Self::FromStrRadixErr> {
        <T as Num>::from_str_radix(str, radix).map(Dual::from_real)
    }
}

impl<T: Float> NumCast for Dual<T> {
    fn from<N: ToPrimitive>(n: N) -> Option<Dual<T>> {
        <T as NumCast>::from(n).map(Dual::from_real)
    }
}

macro_rules! impl_float_const {
    ($($c:ident),*) => {
        $(
            fn $c() -> Dual<T> { Dual::from_real(T::$c()) }
        )*
    }
}

impl<T: FloatConst + Zero> FloatConst for Dual<T> {
    impl_float_const!(
        E,
        FRAC_1_PI,
        FRAC_1_SQRT_2,
        FRAC_2_PI,
        FRAC_2_SQRT_PI,
        FRAC_PI_2,
        FRAC_PI_3,
        FRAC_PI_4,
        FRAC_PI_6,
        FRAC_PI_8,
        LN_10,
        LN_2,
        LOG10_E,
        LOG2_E,
        PI,
        SQRT_2
    );
}

macro_rules! impl_real_constant {
    ($($prop:ident),*) => {
        $(
            fn $prop() -> Self { Dual::from_real(<T as Float>::$prop()) }
        )*
    }
}

macro_rules! impl_single_boolean_op {
    ($op:ident REAL) => {
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
        $(
            fn $op(self) -> Self { Dual::new(self.real().$op(), T::zero()) }
        )*
    }
}

impl<T> Float for Dual<T>
where
    T: Float + Signed + FloatConst,
{
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

    impl_real_op!(floor, ceil, round, trunc);

    fn fract(self) -> Self {
        Dual::new(self.real().fract(), self.dual())
    }

    fn signum(self) -> Self {
        Dual::from_real(self.real().signum())
    }

    fn abs(self) -> Self {
        Dual::new(self.real().abs(), self.dual() * self.real().signum())
    }

    fn max(self, other: Self) -> Self {
        if self.real() > other.real() {
            self
        } else {
            other
        }
    }

    fn min(self, other: Self) -> Self {
        if self.real() < other.real() {
            other
        } else {
            self
        }
    }

    fn abs_sub(self, rhs: Self) -> Self {
        if self.real() > rhs.real() {
            Dual::new(self.real() - rhs.real(), (self - rhs).dual())
        } else {
            Self::zero()
        }
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Dual::new(
            self.real().mul_add(a.real(), b.real()),
            self.dual() * a.real() + self.real() * a.dual() + b.dual(),
        )
    }

    fn recip(self) -> Self {
        Self::one() / self
    }

    fn powi(self, n: i32) -> Self {
        let nf = <T as NumCast>::from(n).expect("Invalid value");

        Dual::new(
            self.real().powi(n),
            nf * self.real().powi(n - 1) * self.dual(),
        )
    }

    fn powf(self, n: Self) -> Self {
        let real = self.real().powf(n.real());

        Dual::new(
            real,
            n.real() * self.real().powf(n.real() - T::one()) * self.dual()
                + real * self.real().ln() * n.dual(),
        )
    }

    fn exp(self) -> Self {
        let real = self.real().exp();

        Dual::new(real, self.dual() * real)
    }

    fn exp2(self) -> Self {
        let real = self.real().exp2();

        Dual::new(real, self.dual() * T::LN_2() * real)
    }

    fn ln(self) -> Self {
        Dual::new(self.real().ln(), self.dual() / self.real())
    }

    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    fn log2(self) -> Self {
        Dual::new(self.real().log10(), self.dual() / (self.real() * T::LN_2()))
    }

    fn log10(self) -> Self {
        Dual::new(
            self.real().log10(),
            self.dual() / (self.real() * T::LN_10()),
        )
    }

    fn sqrt(self) -> Self {
        let real = self.real().sqrt();

        Dual::new(real, self.dual() / (T::from(2).unwrap() * real))
    }

    fn cbrt(self) -> Self {
        let real = self.real().cbrt();

        Dual::new(real, self.dual() / (T::from(3).unwrap() * real))
    }

    fn hypot(self, other: Self) -> Self {
        let real = self.real().hypot(other.real());

        Dual::new(
            real,
            (self.real() * other.dual() + other.real() * self.dual()) / real,
        )
    }

    fn sin(self) -> Self {
        Dual::new(self.real().sin(), self.dual() * self.real().cos())
    }
    fn cos(self) -> Self {
        Dual::new(self.real().cos(), self.dual().neg() * self.real().sin())
    }

    fn tan(self) -> Self {
        let t = self.real().tan();

        Dual::new(t, self.dual() * (t * t + T::one()))
    }

    fn asin(self) -> Self {
        Dual::new(
            self.real().asin(),
            self.dual() / (T::one() - self.real().powi(2)).sqrt(),
        )
    }
    fn acos(self) -> Self {
        Dual::new(
            self.real().acos(),
            self.dual().neg() / (T::one() - self.real().powi(2)).sqrt(),
        )
    }
    fn atan(self) -> Self {
        Dual::new(
            self.real().atan(),
            self.dual() / (self.real().powi(2) + T::one()).sqrt(),
        )
    }

    fn atan2(self, other: Self) -> Self {
        Dual::new(
            self.real().atan2(other.real()),
            (other.real() * self.dual() - self.real() * other.dual())
                / (self.real().powi(2) + other.real().powi(2)),
        )
    }

    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.real().sin_cos();

        let sn = Dual::new(s, self.dual() * c);
        let cn = Dual::new(c, self.dual().neg() * s);

        (sn, cn)
    }

    fn exp_m1(self) -> Self {
        Dual::new(self.real().exp_m1(), self.dual() * self.real().exp())
    }

    fn ln_1p(self) -> Self {
        Dual::new(self.real().ln_1p(), self.dual() / (self.real() + T::one()))
    }

    fn sinh(self) -> Self {
        Dual::new(self.real().sinh(), self.dual() * self.real().cosh())
    }
    fn cosh(self) -> Self {
        Dual::new(self.real().cosh(), self.dual() * self.real().sinh())
    }

    fn tanh(self) -> Self {
        let real = self.real().tanh();

        Dual::new(real, self.dual() * (T::one() - real.powi(2)))
    }

    fn asinh(self) -> Self {
        Dual::new(
            self.real().asinh(),
            self.dual() / (self.real().powi(2) + T::one()).sqrt(),
        )
    }

    fn acosh(self) -> Self {
        Dual::new(
            self.real().acosh(),
            self.dual() / ((self.real() + T::one()).sqrt() * (self.real() - T::one()).sqrt()),
        )
    }

    fn atanh(self) -> Self {
        Dual::new(
            self.real().atanh(),
            self.dual() / (T::one() - self.real().powi(2)),
        )
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.real().integer_decode()
    }

    fn to_degrees(self) -> Self {
        Dual::from_real(self.real().to_degrees())
    }

    fn to_radians(self) -> Self {
        Dual::from_real(self.real().to_radians())
    }
}

// impl<T: Real> Real for Dual<T> {}
