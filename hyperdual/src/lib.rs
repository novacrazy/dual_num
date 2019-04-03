//! Dual Numbers
//!
//! Fully-featured Dual Number implementation with features for automatic differentiation of multivariate vectorial functions into gradients.
//!
//! ## Usage
//!
//! ```rust
//! extern crate hyperdual;
//!
//! use hyperdual::{Dual, Hyperdual, Float, differentiate, U3};
//!
//! fn main() {
//!     // find partial derivative at x=4.0
//!     let univariate = differentiate(4.0f64, |x| x.sqrt() + Dual::from_real(1.0));
//!     assert!((univariate - 0.25).abs() < 1e-16, "wrong derivative");
//!
//!     // find the partial derivatives of a multivariate function
//!     let x: Hyperdual<f64, U3> = Hyperdual::from_slice(&[4.0, 1.0, 0.0]);
//!     let y: Hyperdual<f64, U3> = Hyperdual::from_slice(&[5.0, 0.0, 1.0]);
//!
//!     let multivariate = x * x + (x * y).sin() + y.powi(3);
//!     assert!((multivariate[0] - 141.91294525072763).abs() < 1e-13, "f(4, 5) incorrect");
//!     assert!((multivariate[1] - 10.04041030906696).abs() < 1e-13, "df/dx(4, 5) incorrect");
//!     assert!((multivariate[2] - 76.63232824725357).abs() < 1e-13, "df/dy(4, 5) incorrect");
//! }
//! ```
//!
//! ##### Previous Work
//! * [https://github.com/FreeFull/dual_numbers](https://github.com/FreeFull/dual_numbers)
//! * [https://github.com/ibab/rust-ad](https://github.com/ibab/rust-ad)
//! * [https://github.com/tesch1/cxxduals](https://github.com/tesch1/cxxduals)

extern crate nalgebra as na;
extern crate num_traits;

use std::cmp::Ordering;
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::iter::{Product, Sum};
use std::num::FpCategory;
use std::ops::{Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

pub use num_traits::{Float, FloatConst, Num, One, Zero};

pub mod linalg;

use num_traits::{FromPrimitive, Inv, MulAdd, MulAddAssign, NumCast, Pow, Signed, ToPrimitive, Unsigned};

use na::VectorN;

// Re-export traits useful for construction and extension of duals
pub use na::allocator::Allocator;
pub use na::dimension::*;
pub use na::storage::Owned;
pub use na::{DefaultAllocator, Dim, DimName, Scalar};

/// Hyperdual Number structure
///
/// Although `Hyperdual` does implement `PartialEq` and `PartialOrd`,
/// it only compares the real part.
///
/// Additionally, `min` and `max` only compare the real parts, and keep the dual parts.
///
/// Lastly, the `Rem` remainder operator is not correctly or fully defined for `Hyperdual`, and will panic.
#[derive(Clone, Copy)]
pub struct Hyperdual<T: Scalar, N: Dim + DimName>(VectorN<T, N>)
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy;

pub type DualN<T, N> = Hyperdual<T, N>;

impl<T: Scalar, N: Dim + DimName> Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    /// Create a new dual number from its real and dual parts.
    #[inline]
    pub fn from_slice(v: &[T]) -> Hyperdual<T, N> {
        Hyperdual(VectorN::<T, N>::from_row_slice(v))
    }

    /// Create a new dual number from a real number.
    ///
    /// The dual part is set to zero.
    #[inline]
    pub fn from_real(real: T) -> Hyperdual<T, N>
    where
        T: Zero,
    {
        let mut dual = VectorN::<T, N>::zeros();
        dual[0] = real;
        Hyperdual(dual)
    }

    /// Returns the real part
    #[inline]
    pub fn real(&self) -> T {
        self[0]
    }

    /// Returns a reference to the real part
    #[inline]
    pub fn real_ref(&self) -> &T {
        &self[0]
    }

    /// Returns a mutable reference to the real part
    pub fn real_mut(&mut self) -> &mut T {
        &mut self[0]
    }

    #[inline]
    pub fn map_dual<F>(&self, r: T, f: F) -> Hyperdual<T, N>
    where
        F: Fn(&T) -> T,
    {
        // TODO: improve, so the real does not get mapped
        let mut v = self.map(|x| f(&x));
        v[0] = r;
        Hyperdual(v)
    }

    /// Create a new dual number from a function
    #[inline]
    pub fn from_fn<F>(mut f: F) -> Hyperdual<T, N>
    where
        F: FnMut(usize) -> T,
    {
        Hyperdual(VectorN::<T, N>::from_fn(|i, _| f(i)))
    }
}

pub type Dual<T> = Hyperdual<T, U2>;

impl<T: Scalar> Dual<T> {
    /// Create a new dual number from the real and dual parts
    #[inline]
    pub fn new(real: T, dual: T) -> Dual<T> {
        Dual::from_slice(&[real, dual])
    }

    /// Returns the dual part
    #[inline]
    pub fn dual(&self) -> T {
        self[1]
    }

    /// Returns a reference to the dual part
    #[inline]
    pub fn dual_ref(&self) -> &T {
        &self[1]
    }

    /// Returns a mutable reference to the dual part
    #[inline]
    pub fn dual_mut(&mut self) -> &mut T {
        &mut self[1]
    }
}

impl<T: Scalar, N: Dim + DimName> Debug for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let mut a = f.debug_tuple("Dual");
        for x in self.iter() {
            a.field(x);
        }
        a.finish()
    }
}

impl<T: Scalar + Num + Zero, N: Dim + DimName> Default for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn default() -> Hyperdual<T, N> {
        Hyperdual::zero()
    }
}

impl<T: Scalar + Zero, N: Dim + DimName> From<T> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn from(real: T) -> Hyperdual<T, N> {
        Hyperdual::from_real(real)
    }
}

impl<T: Scalar, N: Dim + DimName> Deref for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    type Target = VectorN<T, N>;

    #[inline]
    fn deref(&self) -> &VectorN<T, N> {
        &self.0
    }
}

impl<T: Scalar, N: Dim + DimName> DerefMut for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn deref_mut(&mut self) -> &mut VectorN<T, N> {
        &mut self.0
    }
}

impl<T: Scalar, N: Dim + DimName> AsRef<VectorN<T, N>> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn as_ref(&self) -> &VectorN<T, N> {
        &self.0
    }
}

impl<T: Scalar, N: Dim + DimName> AsMut<VectorN<T, N>> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn as_mut(&mut self) -> &mut VectorN<T, N> {
        &mut self.0
    }
}

impl<T: Scalar + Neg<Output = T>, N: Dim + DimName> Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    /// Returns the conjugate of the dual number.
    #[inline]
    pub fn conjugate(self) -> Self {
        self.map_dual(self.real(), |x| x.neg())
    }
}

impl<T: Scalar + Display, N: Dim + DimName> Display for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let precision = f.precision().unwrap_or(4);

        write!(f, "{:.p$}", self.real(), p = precision)?;
        for (i, x) in self.iter().skip(1).enumerate() {
            write!(f, " + {:.p$}\u{03B5}{}", x, i + 1, p = precision)?;
        }

        Ok(())
    }
}

impl<T: Scalar + PartialEq, N: Dim + DimName> PartialEq<Self> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.0 == rhs.0
    }
}

impl<T: Scalar + PartialOrd, N: Dim + DimName> PartialOrd<Self> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(self.real_ref(), rhs.real_ref())
    }

    #[inline]
    fn lt(&self, rhs: &Self) -> bool {
        self.real() < rhs.real()
    }

    #[inline]
    fn gt(&self, rhs: &Self) -> bool {
        self.real() > rhs.real()
    }
}

impl<T: Scalar + PartialEq, N: Dim + DimName> PartialEq<T> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn eq(&self, rhs: &T) -> bool {
        *self.real_ref() == *rhs
    }
}

impl<T: Scalar + PartialOrd, N: Dim + DimName> PartialOrd<T> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn partial_cmp(&self, rhs: &T) -> Option<Ordering> {
        PartialOrd::partial_cmp(self.real_ref(), rhs)
    }

    #[inline]
    fn lt(&self, rhs: &T) -> bool {
        self.real() < *rhs
    }

    #[inline]
    fn gt(&self, rhs: &T) -> bool {
        self.real() > *rhs
    }
}

macro_rules! impl_to_primitive {
    ($($name:ident, $ty:ty),*) => {
        impl<T: Scalar + ToPrimitive, N: Dim + DimName> ToPrimitive for Hyperdual<T, N>
            where
                DefaultAllocator: Allocator<T, N>,
                Owned<T, N>: Copy, {
            $(
                #[inline]
                fn $name(&self) -> Option<$ty> {
                    self.real_ref().$name()
                }
            )*
        }
    }
}

macro_rules! impl_from_primitive {
    ($($name:ident, $ty:ty),*) => {
        impl<T: Scalar + FromPrimitive, N: Dim + DimName> FromPrimitive for Hyperdual<T, N>
            where
                T: Zero,
                DefaultAllocator: Allocator<T, N>,
                Owned<T, N>: Copy, {
            $(
                #[inline]
                fn $name(n: $ty) -> Option<Hyperdual<T,N>> {
                    T::$name(n).map(Hyperdual::from_real)
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

impl_primitive_cast! {
    to_isize,   from_isize  - isize,
    to_i8,      from_i8     - i8,
    to_i16,     from_i16    - i16,
    to_i32,     from_i32    - i32,
    to_i64,     from_i64    - i64,
    to_usize,   from_usize  - usize,
    to_u8,      from_u8     - u8,
    to_u16,     from_u16    - u16,
    to_u32,     from_u32    - u32,
    to_u64,     from_u64    - u64,
    to_f32,     from_f32    - f32,
    to_f64,     from_f64    - f64
}

impl<T: Scalar + Num, N: Dim + DimName> Add<T> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    type Output = Hyperdual<T, N>;

    #[inline]
    fn add(self, rhs: T) -> Hyperdual<T, N> {
        let mut d = self.clone();
        d[0] = d[0] + rhs;
        d
    }
}

impl<T: Scalar + Num, N: Dim + DimName> AddAssign<T> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        *self = (*self) + Hyperdual::from_real(rhs)
    }
}

impl<T: Scalar + Num, N: Dim + DimName> Sub<T> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    type Output = Hyperdual<T, N>;

    #[inline]
    fn sub(self, rhs: T) -> Hyperdual<T, N> {
        let mut d = self.clone();
        d[0] = d[0] - rhs;
        d
    }
}

impl<T: Scalar + Num, N: Dim + DimName> SubAssign<T> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        *self = (*self) - Hyperdual::from_real(rhs)
    }
}

impl<T: Scalar + Num, N: Dim + DimName> Mul<T> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    type Output = Hyperdual<T, N>;

    #[inline]
    fn mul(self, rhs: T) -> Hyperdual<T, N> {
        self * Hyperdual::from_real(rhs)
    }
}

impl<T: Scalar + Num, N: Dim + DimName> MulAssign<T> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = (*self) * Hyperdual::from_real(rhs)
    }
}

impl<T: Scalar + Num, N: Dim + DimName> Div<T> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    type Output = Hyperdual<T, N>;

    #[inline]
    fn div(self, rhs: T) -> Hyperdual<T, N> {
        self / Hyperdual::from_real(rhs)
    }
}

impl<T: Scalar + Num, N: Dim + DimName> DivAssign<T> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        *self = (*self) / Self::from_real(rhs)
    }
}

impl<T: Scalar + Signed, N: Dim + DimName> Neg for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Hyperdual(self.map(|x| x.neg()))
    }
}

impl<T: Scalar + Num, N: Dim + DimName> Add<Self> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Hyperdual(self.zip_map(&rhs, |x, y| x + y))
    }
}

impl<T: Scalar + Num, N: Dim + DimName> AddAssign<Self> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = (*self) + rhs
    }
}

impl<T: Scalar + Num, N: Dim + DimName> Sub<Self> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Hyperdual(self.zip_map(&rhs, |x, y| x - y))
    }
}

impl<T: Scalar + Num, N: Dim + DimName> SubAssign<Self> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = (*self) - rhs
    }
}

impl<T: Scalar + Num, N: Dim + DimName> Mul<Self> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // TODO: skip real part
        let mut v = self.zip_map(&rhs, |x, y| rhs.real() * x + self.real() * y);
        v[0] = self.real() * rhs.real();
        Hyperdual(v)
    }
}

impl<T: Scalar + Num, N: Dim + DimName> MulAssign<Self> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = (*self) * rhs
    }
}

macro_rules! impl_mul_add {
    ($(<$a:ident, $b:ident>),*) => {
        $(
            impl<T: Scalar + Num + Mul + Add, N: Dim + DimName> MulAdd<$a, $b> for Hyperdual<T,N>where
                DefaultAllocator: Allocator<T, N>,
                Owned<T, N>: Copy,
            {
                type Output = Hyperdual<T,N>;

                #[inline]
                fn mul_add(self, a: $a, b: $b) -> Self {
                    (self * a) + b
                }
            }

            impl<T: Scalar + Num + Mul + Add, N: Dim + DimName> MulAddAssign<$a, $b> for Hyperdual<T,N>where
                DefaultAllocator: Allocator<T, N>,
                Owned<T, N>: Copy,
            {
                #[inline]
                fn mul_add_assign(&mut self, a: $a, b: $b) {
                    *self = (*self * a) + b;
                }
            }
        )*
    }
}

impl_mul_add! {
    <Self, Self>,
    <T, Self>,
    <Self, T>,
    <T, T>
}

impl<T: Scalar + Num, N: Dim + DimName> Div<Self> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        // TODO: specialize with inv so we can precompute the inverse
        let d = rhs.real() * rhs.real();

        // TODO: skip real part
        let mut v = self.zip_map(&rhs, |x, y| (rhs.real() * x - self.real() * y) / d);
        v[0] = self.real() / rhs.real();
        Hyperdual(v)
    }
}

impl<T: Scalar + Num, N: Dim + DimName> DivAssign<Self> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = (*self) / rhs
    }
}

impl<T: Scalar + Num, N: Dim + DimName> Rem<Self> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    type Output = Self;

    /// **UNIMPLEMENTED!!!**
    ///
    /// As far as I know, remainder is not a valid operation on dual numbers,
    /// but is required for the `Float` trait to be implemented.
    fn rem(self, _: Self) -> Self {
        unimplemented!()
    }
}

impl<T: Scalar, P: Into<Hyperdual<T, N>>, N: Dim + DimName> Pow<P> for Hyperdual<T, N>
where
    Hyperdual<T, N>: Float,
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    type Output = Self;

    #[inline]
    fn pow(self, rhs: P) -> Self {
        self.powf(rhs.into())
    }
}

impl<T: Scalar, N: Dim + DimName> Inv for Hyperdual<T, N>
where
    Self: One + Div<Output = Self>,
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    type Output = Self;

    #[inline]
    fn inv(self) -> Self {
        Self::one() / self
    }
}

impl<T: Scalar, N: Dim + DimName> Signed for Hyperdual<T, N>
where
    T: Signed + PartialOrd,
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn abs(&self) -> Self {
        let s = self.real().signum();
        Hyperdual(self.map(|x| x * s))
    }

    #[inline]
    fn abs_sub(&self, rhs: &Self) -> Self {
        if self.real() > rhs.real() {
            self.sub(*rhs)
        } else {
            Self::zero()
        }
    }

    #[inline]
    fn signum(&self) -> Self {
        Self::from_real(self.real().signum())
    }

    #[inline]
    fn is_positive(&self) -> bool {
        self.real().is_positive()
    }

    #[inline]
    fn is_negative(&self) -> bool {
        self.real().is_negative()
    }
}

impl<T: Scalar + Unsigned, N: Dim + DimName> Unsigned for Hyperdual<T, N>
where
    Self: Num,
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
}

impl<T: Scalar + Num + Zero, N: Dim + DimName> Zero for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn zero() -> Hyperdual<T, N> {
        Hyperdual::from_real(T::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.iter().all(|x| x.is_zero())
    }
}

impl<T: Scalar + Num + One, N: Dim + DimName> One for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn one() -> Hyperdual<T, N> {
        Hyperdual::from_real(T::one())
    }

    #[inline]
    fn is_one(&self) -> bool
    where
        Self: PartialEq,
    {
        self.real().is_one()
    }
}

impl<T: Scalar + Num, N: Dim + DimName> Num for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    type FromStrRadixErr = <T as Num>::FromStrRadixErr;

    #[inline]
    fn from_str_radix(str: &str, radix: u32) -> Result<Hyperdual<T, N>, Self::FromStrRadixErr> {
        <T as Num>::from_str_radix(str, radix).map(Self::from_real)
    }
}

impl<T: Scalar + Float, N: Dim + DimName> NumCast for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    #[inline]
    fn from<P: ToPrimitive>(n: P) -> Option<Hyperdual<T, N>> {
        <T as NumCast>::from(n).map(Self::from_real)
    }
}

macro_rules! impl_float_const {
    ($($c:ident),*) => {
        $(
            fn $c() -> Hyperdual<T, N> { Self::from_real(T::$c()) }
        )*
    }
}

impl<T: Scalar + FloatConst + Zero, N: Dim + DimName> FloatConst for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
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
            fn $prop() -> Self { Self::from_real(<T as Float>::$prop()) }
        )*
    }
}

macro_rules! impl_single_boolean_op {
    ($op:ident REAL) => {
        fn $op(self) -> bool {self.real().$op()}
    };
    ($op:ident OR) =>   { fn $op(self) -> bool {
        let mut b = self.real().$op();
        for x in self.iter().skip(1) {
            b |= x.$op();
        }
        b} };
    ($op:ident AND) =>  { fn $op(self) -> bool {
        let mut b = self.real().$op();
        for x in self.iter().skip(1) {
            b &= x.$op();
        }
        b} };
}

macro_rules! impl_boolean_op {
    ($($op:ident $t:tt),*) => {
        $(impl_single_boolean_op!($op $t);)*
    };
}

macro_rules! impl_real_op {
    ($($op:ident),*) => {
        $(
            fn $op(self) -> Self { Hyperdual::from_real(self.real().$op()) }
        )*
    }
}

impl<T: Scalar + Num + Zero, N: Dim + DimName> Sum for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    fn sum<I: Iterator<Item = Hyperdual<T, N>>>(iter: I) -> Hyperdual<T, N> {
        iter.fold(Hyperdual::zero(), |a, b| a + b)
    }
}

impl<'a, T: Scalar + Num + Zero, N: Dim + DimName> Sum<&'a Hyperdual<T, N>> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    fn sum<I: Iterator<Item = &'a Hyperdual<T, N>>>(iter: I) -> Hyperdual<T, N> {
        iter.fold(Hyperdual::zero(), |a, b| a + *b)
    }
}

impl<T: Scalar + Num + One, N: Dim + DimName> Product for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    fn product<I: Iterator<Item = Hyperdual<T, N>>>(iter: I) -> Hyperdual<T, N> {
        iter.fold(Hyperdual::one(), |a, b| a * b)
    }
}

impl<'a, T: Scalar + Num + One, N: Dim + DimName> Product<&'a Hyperdual<T, N>> for Hyperdual<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    fn product<I: Iterator<Item = &'a Hyperdual<T, N>>>(iter: I) -> Hyperdual<T, N> {
        iter.fold(Hyperdual::one(), |a, b| a * *b)
    }
}

impl<T: Scalar, N: Dim + DimName> Float for Hyperdual<T, N>
where
    T: Float + Signed + FloatConst,
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy,
{
    impl_real_constant!(nan, infinity, neg_infinity, neg_zero, min_positive_value, epsilon, min_value, max_value);

    impl_boolean_op!(
        is_nan              OR,
        is_infinite         OR,
        is_finite           AND,
        is_normal           AND,
        is_sign_positive    REAL,
        is_sign_negative    REAL
    );

    #[inline]
    fn classify(self) -> FpCategory {
        self.real().classify()
    }

    impl_real_op!(floor, ceil, round, trunc);

    #[inline]
    fn fract(self) -> Self {
        let mut v = self.clone();
        v[0] = self.real().fract();
        v
    }

    #[inline]
    fn signum(self) -> Self {
        Self::from_real(self.real().signum())
    }

    #[inline]
    fn abs(self) -> Self {
        let s = self.real().signum();
        Hyperdual(self.map(|x| x * s))
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        if self.real() > other.real() {
            self
        } else {
            other
        }
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        if self.real() < other.real() {
            other
        } else {
            self
        }
    }

    #[inline]
    fn abs_sub(self, rhs: Self) -> Self {
        if self.real() > rhs.real() {
            self.sub(rhs)
        } else {
            Self::zero()
        }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let mut dual = Hyperdual::from_real(self.real().mul_add(a.real(), b.real()));

        for x in 1..self.len() {
            dual[x] = self[x] * a.real() + self.real() * a[x] + b[x];
        }

        dual
    }

    #[inline]
    fn recip(self) -> Self {
        Self::one() / self
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        let r = self.real().powi(n - 1);
        let nf = <T as NumCast>::from(n).expect("Invalid value") * r;

        self.map_dual(self.real() * r, |x| nf * *x)
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        let c = self.real().powf(n.real());
        let a = n.real() * self.real().powf(n.real() - T::one());
        let b = c * self.real().ln();

        let mut v = self.zip_map(&n, |x, y| a * x + b * y);
        v[0] = c;
        Hyperdual(v)
    }

    #[inline]
    fn exp(self) -> Self {
        let real = self.real().exp();
        self.map_dual(real, |x| real * *x)
    }

    #[inline]
    fn exp2(self) -> Self {
        let real = self.real().exp2();
        self.map_dual(real, |x| *x * T::LN_2() * real)
    }

    #[inline]
    fn ln(self) -> Self {
        self.map_dual(self.real().ln(), |x| *x / self.real())
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    #[inline]
    fn log2(self) -> Self {
        self.map_dual(self.real().log2(), |x| *x / (self.real() * T::LN_2()))
    }

    #[inline]
    fn log10(self) -> Self {
        self.map_dual(self.real().log10(), |x| *x / (self.real() * T::LN_10()))
    }

    #[inline]
    fn sqrt(self) -> Self {
        let real = self.real().sqrt();
        let d = T::from(1).unwrap() / (T::from(2).unwrap() * real);
        self.map_dual(real, |x| *x * d)
    }

    #[inline]
    fn cbrt(self) -> Self {
        let real = self.real().cbrt();
        self.map_dual(real, |x| *x / (T::from(3).unwrap() * real))
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        let c = self.real().hypot(other.real());
        let mut v = self.zip_map(&other, |x, y| (self.real() * y + other.real() * x) / c);
        v[0] = c;
        Hyperdual(v)
    }

    #[inline]
    fn sin(self) -> Self {
        let c = self.real().cos();
        self.map_dual(self.real().sin(), |x| *x * c)
    }

    #[inline]
    fn cos(self) -> Self {
        let c = self.real().sin();
        self.map_dual(self.real().cos(), |x| x.neg() * c)
    }

    #[inline]
    fn tan(self) -> Self {
        let t = self.real().tan();
        let c = t * t + T::one();
        self.map_dual(t, |x| *x * c)
    }

    #[inline]
    fn asin(self) -> Self {
        // TODO: implement inv
        let c = (T::one() - self.real().powi(2)).sqrt();
        self.map_dual(self.real().asin(), |x| *x / c)
    }

    #[inline]
    fn acos(self) -> Self {
        // TODO: implement inv
        let c = (T::one() - self.real().powi(2)).sqrt();
        self.map_dual(self.real().acos(), |x| x.neg() / c)
    }

    #[inline]
    fn atan(self) -> Self {
        // TODO: implement inv
        let c = T::one() + self.real().powi(2);
        self.map_dual(self.real().atan(), |x| *x / c)
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        let c = self.real().powi(2) + other.real().powi(2);
        let mut v = self.zip_map(&other, |x, y| (other.real() * x - self.real() * y) / c);
        v[0] = self.real().atan2(other.real());
        Hyperdual(v)
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.real().sin_cos();
        let sn = self.map_dual(s, |x| *x * c);
        let cn = self.map_dual(c, |x| x.neg() * s);
        (sn, cn)
    }

    #[inline]
    fn exp_m1(self) -> Self {
        let c = self.real().exp();
        self.map_dual(self.real().exp_m1(), |x| *x * c)
    }

    #[inline]
    fn ln_1p(self) -> Self {
        let c = self.real() + T::one();
        self.map_dual(self.real().ln_1p(), |x| *x / c)
    }

    #[inline]
    fn sinh(self) -> Self {
        let c = self.real().cosh();
        self.map_dual(self.real().sinh(), |x| *x * c)
    }

    #[inline]
    fn cosh(self) -> Self {
        let c = self.real().sinh();
        self.map_dual(self.real().cosh(), |x| *x * c)
    }

    #[inline]
    fn tanh(self) -> Self {
        let real = self.real().tanh();
        let c = T::one() - real.powi(2);
        self.map_dual(real, |x| *x * c)
    }

    #[inline]
    fn asinh(self) -> Self {
        let c = (self.real().powi(2) + T::one()).sqrt();
        self.map_dual(self.real().asinh(), |x| *x / c)
    }

    #[inline]
    fn acosh(self) -> Self {
        let c = (self.real() + T::one()).sqrt() * (self.real() - T::one()).sqrt();
        self.map_dual(self.real().acosh(), |x| *x / c)
    }

    #[inline]
    fn atanh(self) -> Self {
        let c = T::one() - self.real().powi(2);
        self.map_dual(self.real().atanh(), |x| *x / c)
    }

    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) {
        self.real().integer_decode()
    }

    #[inline]
    fn to_degrees(self) -> Self {
        Hyperdual::from_real(self.real().to_degrees())
    }

    #[inline]
    fn to_radians(self) -> Self {
        Hyperdual::from_real(self.real().to_radians())
    }
}

// TODO
// impl<T: na::Real> na::Real for Hyperdual<T,N> {}

pub fn hyperspace_from_vector<T: Scalar + Num + Zero, D: Dim + DimName, N: Dim + DimName>(v: &VectorN<T, N>) -> VectorN<Hyperdual<T, D>, N>
where
    DefaultAllocator: Allocator<T, D> + Allocator<T, N> + Allocator<Hyperdual<T, D>, N>,
    Owned<T, D>: Copy,
{
    let mut space_slice = vec![Hyperdual::<T, D>::zero(); N::dim()];
    for i in 0..N::dim() {
        space_slice[i] = Hyperdual::<T, D>::from_fn(|j| {
            if j == 0 {
                v[i]
            } else if i + 1 == j {
                T::one()
            } else {
                T::zero()
            }
        });
    }
    VectorN::<Hyperdual<T, D>, N>::from_row_slice(&space_slice)
}

/// Evaluates the function using dual numbers to get the partial derivative at the input point
#[inline]
pub fn differentiate<T: Scalar + One, F>(x: T, f: F) -> T
where
    F: Fn(Dual<T>) -> Dual<T>,
{
    f(Dual::new(x, T::one())).dual()
}
