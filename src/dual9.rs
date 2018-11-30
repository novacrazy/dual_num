use std::cmp::Ordering;
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::iter::{Product, Sum};
use std::num::FpCategory;
use std::ops::{Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

pub use num_traits::{Float, FloatConst, Num, One, Zero};

// Re-export the differential functions
pub use differentials::*;

use num_traits::{FromPrimitive, Inv, MulAdd, MulAddAssign, NumCast, Pow, Signed, ToPrimitive, Unsigned};

pub use na::{MatrixArray, Scalar, VectorN, U1, U2, U9};

type Vector9<T> = VectorN<T, U9>;

/// Dual Number structure
///
/// Although `Dual` does implement `PartialEq` and `PartialOrd`,
/// it only compares the real part.
///
/// Additionally, `min` and `max` only compare the real parts, and keep the dual parts.
///
/// Lastly, the `Rem` remainder operator is not correctly or fully defined for `Dual`, and will panic.
#[derive(Clone, Copy)]
pub struct Dual9<T: Scalar>(Vector9<T>);

impl<T: Scalar> Dual9<T> {
    /// Create a new dual number from its real and dual parts.
    #[inline]
    pub fn new(real: T, dual1: T, dual2: T, dual3: T, dual4: T, dual5: T, dual6: T, dual7: T, dual8: T) -> Dual9<T> {
        Dual9(Vector9::from_row_slice(&[real, dual1, dual2, dual3, dual4, dual5, dual6, dual7, dual8]))
    }

    /// Create a new dual number from a real number.
    ///
    /// The dual part is set to zero.
    #[inline]
    pub fn from_real(real: T) -> Dual9<T>
    where
        T: Zero,
    {
        Dual9::new(
            real,
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
        )
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
    #[inline]
    pub fn real_mut(&mut self) -> &mut T {
        &mut self[0]
    }

    /// Returns a mutable reference to the first dual part
    #[inline]
    pub fn dual_mut(&mut self) -> &mut T {
        &mut self[1]
    }
}

impl<T: Scalar> Debug for Dual9<T> {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        f.debug_tuple("Dual")
            .field(self.real_ref())
            .field(&self[1])
            .field(&self[2])
            .field(&self[3])
            .field(&self[4])
            .field(&self[5])
            .field(&self[6])
            .field(&self[7])
            .field(&self[8])
            .finish()
    }
}

impl<T: Scalar + Zero> Default for Dual9<T> {
    #[inline]
    fn default() -> Dual9<T> {
        Dual9::new(
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
        )
    }
}

impl<T: Scalar + Zero> From<T> for Dual9<T> {
    #[inline]
    fn from(real: T) -> Dual9<T> {
        Dual9::from_real(real)
    }
}

impl<T: Scalar> Deref for Dual9<T> {
    type Target = Vector9<T>;

    #[inline]
    fn deref(&self) -> &Vector9<T> {
        &self.0
    }
}

impl<T: Scalar> DerefMut for Dual9<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Vector9<T> {
        &mut self.0
    }
}

impl<T: Scalar + Neg<Output = T>> Dual9<T> {
    /// Returns the conjugate of the dual number.
    #[inline]
    pub fn conjugate(self) -> Self {
        Dual9::new(
            self.real(),
            self[1].neg(),
            self[2].neg(),
            self[3].neg(),
            self[4].neg(),
            self[5].neg(),
            self[6].neg(),
            self[7].neg(),
            self[8].neg(),
        )
    }
}

impl<T: Scalar + Display> Display for Dual9<T> {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let precision = f.precision().unwrap_or(4);

        write!(f, "{:.p$} + \u{03B5}1{:.p$} + \u{03B5}1{:.p$} + \u{03B5}1{:.p$} + \u{03B5}1{:.p$} + \u{03B5}1{:.p$} + \u{03B5}1{:.p$} + \u{03B5}1{:.p$} + \u{03B5}1{:.p$} + \u{03B5}1{:.p$}", 
            self.real_ref(), &self[1],&self[2],&self[3],&self[4],&self[5],&self[6],&self[7],&self[8], p = precision)
    }
}

impl<T: Scalar + PartialEq> PartialEq<Self> for Dual9<T> {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.0 == rhs.0
    }
}

impl<T: Scalar + PartialOrd> PartialOrd<Self> for Dual9<T> {
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

impl<T: Scalar + PartialEq> PartialEq<T> for Dual9<T> {
    #[inline]
    fn eq(&self, rhs: &T) -> bool {
        *self.real_ref() == *rhs
    }
}

impl<T: Scalar + PartialOrd> PartialOrd<T> for Dual9<T> {
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
        impl<T: Scalar + ToPrimitive> ToPrimitive for Dual9<T> {
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
        impl<T: Scalar + FromPrimitive> FromPrimitive for Dual9<T> where T: Zero {
            $(
                #[inline]
                fn $name(n: $ty) -> Option<Dual9<T>> {
                    T::$name(n).map(Dual9::from_real)
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

impl<T: Scalar + Num> Add<T> for Dual9<T> {
    type Output = Dual9<T>;

    #[inline]
    fn add(self, rhs: T) -> Dual9<T> {
        let mut d = self.clone();
        d[0] = d[0] + rhs;
        d
    }
}

impl<T: Scalar + Num> AddAssign<T> for Dual9<T> {
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        *self = (*self) + Dual9::from_real(rhs)
    }
}

impl<T: Scalar + Num> Sub<T> for Dual9<T> {
    type Output = Dual9<T>;

    #[inline]
    fn sub(self, rhs: T) -> Dual9<T> {
        let mut d = self.clone();
        d[0] = d[0] - rhs;
        d
    }
}

impl<T: Scalar + Num> SubAssign<T> for Dual9<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        *self = (*self) - Dual9::from_real(rhs)
    }
}

impl<T: Scalar + Num> Mul<T> for Dual9<T> {
    type Output = Dual9<T>;

    #[inline]
    fn mul(self, rhs: T) -> Dual9<T> {
        self * Dual9::from_real(rhs)
    }
}

impl<T: Scalar + Num> MulAssign<T> for Dual9<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = (*self) * Dual9::from_real(rhs)
    }
}

impl<T: Scalar + Num> Div<T> for Dual9<T> {
    type Output = Dual9<T>;

    #[inline]
    fn div(self, rhs: T) -> Dual9<T> {
        self / Dual9::from_real(rhs)
    }
}

impl<T: Scalar + Num> DivAssign<T> for Dual9<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        *self = (*self) / Dual9::from_real(rhs)
    }
}

impl<T: Scalar + Signed> Neg for Dual9<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Dual9(self.map(|x| x.neg()))
    }
}

impl<T: Scalar + Num> Add<Self> for Dual9<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        // TODO: find zip-like method?
        //Dual9(self.zip(rhs).map(|x,y| x + y))
        Dual9::new(
            self[0] + rhs[0],
            self[1] + rhs[1],
            self[2] + rhs[2],
            self[3] + rhs[3],
            self[4] + rhs[4],
            self[5] + rhs[5],
            self[6] + rhs[6],
            self[7] + rhs[7],
            self[8] + rhs[8],
        )
    }
}

impl<T: Scalar + Num> AddAssign<Self> for Dual9<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = (*self) + rhs
    }
}

impl<T: Scalar + Num> Sub<Self> for Dual9<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Dual9::new(
            self[0] - rhs[0],
            self[1] - rhs[1],
            self[2] - rhs[2],
            self[3] - rhs[3],
            self[4] - rhs[4],
            self[5] - rhs[5],
            self[6] - rhs[6],
            self[7] - rhs[7],
            self[8] - rhs[8],
        )
    }
}

impl<T: Scalar + Num> SubAssign<Self> for Dual9<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = (*self) - rhs
    }
}

impl<T: Scalar + Num> Mul<Self> for Dual9<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // TODO: check
        Dual9::new(
            self.real() * rhs.real(),
            self.real() * rhs[1] + self[1] * rhs.real(),
            self.real() * rhs[2] + self[2] * rhs.real(),
            self.real() * rhs[3] + self[3] * rhs.real(),
            self.real() * rhs[4] + self[4] * rhs.real(),
            self.real() * rhs[5] + self[5] * rhs.real(),
            self.real() * rhs[6] + self[6] * rhs.real(),
            self.real() * rhs[7] + self[7] * rhs.real(),
            self.real() * rhs[8] + self[8] * rhs.real(),
        )
    }
}

impl<T: Scalar + Num> MulAssign<Self> for Dual9<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = (*self) * rhs
    }
}

macro_rules! impl_mul_add {
    ($(<$a:ident, $b:ident>),*) => {
        $(
            impl<T: Scalar + Num + Mul + Add> MulAdd<$a, $b> for Dual9<T> {
                type Output = Dual9<T>;

                #[inline]
                fn mul_add(self, a: $a, b: $b) -> Dual9<T> {
                    (self * a) + b
                }
            }

            impl<T: Scalar + Num + Mul + Add> MulAddAssign<$a, $b> for Dual9<T> {
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

impl<T: Scalar + Num> Div<Self> for Dual9<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        let d = rhs.real() * rhs.real();
        Dual9::new(
            self.real() / rhs.real(),
            (self[1] * rhs.real() - self.real() * rhs[1]) / d,
            (self[2] * rhs.real() - self.real() * rhs[2]) / d,
            (self[3] * rhs.real() - self.real() * rhs[3]) / d,
            (self[4] * rhs.real() - self.real() * rhs[4]) / d,
            (self[5] * rhs.real() - self.real() * rhs[5]) / d,
            (self[6] * rhs.real() - self.real() * rhs[6]) / d,
            (self[7] * rhs.real() - self.real() * rhs[7]) / d,
            (self[8] * rhs.real() - self.real() * rhs[8]) / d,
        )
    }
}

impl<T: Scalar + Num> DivAssign<Self> for Dual9<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = (*self) / rhs
    }
}

impl<T: Scalar + Num> Rem<Self> for Dual9<T> {
    type Output = Self;

    /// **UNIMPLEMENTED!!!**
    ///
    /// As far as I know, remainder is not a valid operation on dual numbers,
    /// but is required for the `Float` trait to be implemented.
    fn rem(self, _: Self) -> Self {
        unimplemented!()
    }
}

impl<T: Scalar, P: Into<Dual9<T>>> Pow<P> for Dual9<T>
where
    Dual9<T>: Float,
{
    type Output = Dual9<T>;

    #[inline]
    fn pow(self, rhs: P) -> Dual9<T> {
        self.powf(rhs.into())
    }
}

impl<T: Scalar> Inv for Dual9<T>
where
    Self: One + Div<Output = Self>,
{
    type Output = Dual9<T>;

    #[inline]
    fn inv(self) -> Dual9<T> {
        Dual9::one() / self
    }
}

impl<T: Scalar> Signed for Dual9<T>
where
    T: Signed + PartialOrd,
{
    #[inline]
    fn abs(&self) -> Self {
        let s = self.real().signum();
        Dual9(self.map(|x| x * s))
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
        Dual9::from_real(self.real().signum())
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

impl<T: Scalar + Unsigned> Unsigned for Dual9<T> where Self: Num {}

impl<T: Scalar + Num + Zero> Zero for Dual9<T> {
    #[inline]
    fn zero() -> Dual9<T> {
        Dual9::new(
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
        )
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.iter().all(|x| x.is_zero())
    }
}

impl<T: Scalar + Num + One> One for Dual9<T> {
    #[inline]
    fn one() -> Dual9<T> {
        Dual9::new(
            T::one(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
        )
    }

    #[inline]
    fn is_one(&self) -> bool
    where
        Self: PartialEq,
    {
        self.real().is_one()
    }
}

impl<T: Scalar + Num> Num for Dual9<T> {
    type FromStrRadixErr = <T as Num>::FromStrRadixErr;

    #[inline]
    fn from_str_radix(str: &str, radix: u32) -> Result<Dual9<T>, Self::FromStrRadixErr> {
        <T as Num>::from_str_radix(str, radix).map(Dual9::from_real)
    }
}

impl<T: Scalar + Float> NumCast for Dual9<T> {
    #[inline]
    fn from<N: ToPrimitive>(n: N) -> Option<Dual9<T>> {
        <T as NumCast>::from(n).map(Dual9::from_real)
    }
}

macro_rules! impl_float_const {
    ($($c:ident),*) => {
        $(
            fn $c() -> Dual9<T> { Dual9::from_real(T::$c()) }
        )*
    }
}

impl<T: Scalar + FloatConst + Zero> FloatConst for Dual9<T> {
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
            fn $prop() -> Self { Dual9::from_real(<T as Float>::$prop()) }
        )*
    }
}

macro_rules! impl_single_boolean_op {
    ($op:ident REAL) => {
        fn $op(self) -> bool {self.real().$op()}
    };
    ($op:ident OR) =>   { fn $op(self) -> bool {self.real().$op() || self[1].$op() || self[1].$op()
        || self[2].$op() || self[3].$op() || self[4].$op() || self[5].$op() || self[6].$op()
        || self[7].$op() || self[8].$op()} };
    ($op:ident AND) =>  { fn $op(self) -> bool {self.real().$op()  && self[1].$op() && self[1].$op()
        && self[2].$op() && self[3].$op() && self[4].$op() && self[5].$op() && self[6].$op()
        && self[7].$op() && self[8].$op()} };
}

macro_rules! impl_boolean_op {
    ($($op:ident $t:tt),*) => {
        $(impl_single_boolean_op!($op $t);)*
    };
}

macro_rules! impl_real_op {
    ($($op:ident),*) => {
        $(
            fn $op(self) -> Self { Dual9::new(self.real().$op(), T::zero(),
                    T::zero(), T::zero(), T::zero(), T::zero(), T::zero(), T::zero(), T::zero()) }
        )*
    }
}

impl<T: Scalar + Num + Zero> Sum for Dual9<T> {
    fn sum<I: Iterator<Item = Dual9<T>>>(iter: I) -> Dual9<T> {
        iter.fold(Dual9::zero(), |a, b| a + b)
    }
}

impl<'a, T: Scalar + Num + Zero> Sum<&'a Dual9<T>> for Dual9<T> {
    fn sum<I: Iterator<Item = &'a Dual9<T>>>(iter: I) -> Dual9<T> {
        iter.fold(Dual9::zero(), |a, b| a + *b)
    }
}

impl<T: Scalar + Num + One> Product for Dual9<T> {
    fn product<I: Iterator<Item = Dual9<T>>>(iter: I) -> Dual9<T> {
        iter.fold(Dual9::one(), |a, b| a * b)
    }
}

impl<'a, T: Scalar + Num + One> Product<&'a Dual9<T>> for Dual9<T> {
    fn product<I: Iterator<Item = &'a Dual9<T>>>(iter: I) -> Dual9<T> {
        iter.fold(Dual9::one(), |a, b| a * *b)
    }
}

impl<T: Scalar> Float for Dual9<T>
where
    T: Float + Signed + FloatConst,
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
        unimplemented!()
        //Dual9::new(self.real().fract(), self.Dual9())
    }

    #[inline]
    fn signum(self) -> Self {
        Dual9::from_real(self.real().signum())
    }

    #[inline]
    fn abs(self) -> Self {
        let s = self.real().signum();
        Dual9(self.map(|x| x * s))
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
        Dual9::new(
            self.real().mul_add(a.real(), b.real()),
            self[1] * a.real() + self.real() * a[1] + b[1],
            self[2] * a.real() + self.real() * a[2] + b[2],
            self[3] * a.real() + self.real() * a[3] + b[3],
            self[4] * a.real() + self.real() * a[4] + b[4],
            self[5] * a.real() + self.real() * a[5] + b[5],
            self[6] * a.real() + self.real() * a[6] + b[6],
            self[7] * a.real() + self.real() * a[7] + b[7],
            self[8] * a.real() + self.real() * a[8] + b[8],
        )
    }

    #[inline]
    fn recip(self) -> Self {
        Self::one() / self
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        let r = self.real().powi(n - 1);
        let nf = <T as NumCast>::from(n).expect("Invalid value") * r;

        Dual9::new(
            self.real() * r,
            nf * self[1],
            nf * self[2],
            nf * self[3],
            nf * self[4],
            nf * self[5],
            nf * self[6],
            nf * self[7],
            nf * self[8],
        )
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        let real = self.real().powf(n.real());

        unimplemented!()
        //let dual = n.real() * self.real().powf(n.real() - T::one()) * self.Dual9() + real * self.real().ln() * n.Dual9();
        //Dual9::new(real, dual)
    }

    #[inline]
    fn exp(self) -> Self {
        let real = self.real().exp();

        unimplemented!()
        //Dual9::new(real, self.Dual9() * real)
    }

    #[inline]
    fn exp2(self) -> Self {
        let real = self.real().exp2();

        unimplemented!()
        //Dual9::new(real, self.Dual9() * T::LN_2() * real)
    }

    #[inline]
    fn ln(self) -> Self {
        unimplemented!()
        //Dual9::new(self.real().ln(), self.Dual9() / self.real())
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    #[inline]
    fn log2(self) -> Self {
        unimplemented!()
        //Dual9::new(self.real().log10(), self.Dual9() / (self.real() * T::LN_2()))
    }

    #[inline]
    fn log10(self) -> Self {
        unimplemented!()
        //Dual9::new(self.real().log10(), self.Dual9() / (self.real() * T::LN_10()))
    }

    #[inline]
    fn sqrt(self) -> Self {
        let real = self.real().sqrt();
        let d = T::from(1).unwrap() / (T::from(2).unwrap() * real);

        Dual9::new(
            real,
            self[1] * d,
            self[2] * d,
            self[3] * d,
            self[4] * d,
            self[5] * d,
            self[6] * d,
            self[7] * d,
            self[8] * d,
        )
    }

    #[inline]
    fn cbrt(self) -> Self {
        let real = self.real().cbrt();

        unimplemented!()
        //Dual9::new(real, self.Dual9() / (T::from(3).unwrap() * real))
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        let real = self.real().hypot(other.real());

        unimplemented!()
        //Dual9::new(real, (self.real() * other.Dual9() + other.real() * self.Dual9()) / real)
    }

    #[inline]
    fn sin(self) -> Self {
        unimplemented!()
        //Dual9::new(self.real().sin(), self.Dual9() * self.real().cos())
    }

    #[inline]
    fn cos(self) -> Self {
        unimplemented!()
        //Dual9::new(self.real().cos(), self.Dual9().neg() * self.real().sin())
    }

    #[inline]
    fn tan(self) -> Self {
        let t = self.real().tan();

        unimplemented!()
        //Dual9::new(t, self.Dual9() * (t * t + T::one()))
    }

    #[inline]
    fn asin(self) -> Self {
        unimplemented!()
        //Dual9::new(self.real().asin(), self.Dual9() / (T::one() - self.real().powi(2)).sqrt())
    }

    #[inline]
    fn acos(self) -> Self {
        unimplemented!()
        //Dual9::new(self.real().acos(), self.Dual9().neg() / (T::one() - self.real().powi(2)).sqrt())
    }

    #[inline]
    fn atan(self) -> Self {
        unimplemented!()
        //Dual9::new(self.real().atan(), self.Dual9() / (self.real().powi(2) + T::one()).sqrt())
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        unimplemented!()
        /*Dual9::new(
            self.real().atan2(other.real()),
            (other.real() * self.Dual9() - self.real() * other.Dual9()) / (self.real().powi(2) + other.real().powi(2)),
        )*/
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.real().sin_cos();

        unimplemented!()
        /*let sn = Dual9::new(s, self.Dual9() * c);
        let cn = Dual9::new(c, self.Dual9().neg() * s);
        
        (sn, cn)*/
    }

    #[inline]
    fn exp_m1(self) -> Self {
        unimplemented!()
        //Dual9::new(self.real().exp_m1(), self.Dual9() * self.real().exp())
    }

    #[inline]
    fn ln_1p(self) -> Self {
        unimplemented!()
        //Dual9::new(self.real().ln_1p(), self.Dual9() / (self.real() + T::one()))
    }

    #[inline]
    fn sinh(self) -> Self {
        unimplemented!()
        //Dual9::new(self.real().sinh(), self.Dual9() * self.real().cosh())
    }

    #[inline]
    fn cosh(self) -> Self {
        unimplemented!()
        //Dual9::new(self.real().cosh(), self.Dual9() * self.real().sinh())
    }

    #[inline]
    fn tanh(self) -> Self {
        let real = self.real().tanh();

        unimplemented!()
        //Dual9::new(real, self.Dual9() * (T::one() - real.powi(2)))
    }

    #[inline]
    fn asinh(self) -> Self {
        unimplemented!()
        //Dual9::new(self.real().asinh(), self.Dual9() / (self.real().powi(2) + T::one()).sqrt())
    }

    #[inline]
    fn acosh(self) -> Self {
        unimplemented!()
        /*Dual9::new(
            self.real().acosh(),
            self.Dual9() / ((self.real() + T::one()).sqrt() * (self.real() - T::one()).sqrt()),
        )*/
    }

    #[inline]
    fn atanh(self) -> Self {
        unimplemented!()
        //Dual9::new(self.real().atanh(), self.Dual9() / (T::one() - self.real().powi(2)))
    }

    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) {
        self.real().integer_decode()
    }

    #[inline]
    fn to_degrees(self) -> Self {
        unimplemented!()
        //Dual9::from_real(self.real().to_degrees())
    }

    #[inline]
    fn to_radians(self) -> Self {
        unimplemented!()
        //Dual9::from_real(self.real().to_radians())
    }
}

// TODO
// impl<T: na::Real> na::Real for Dual9<T> {}
