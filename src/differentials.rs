use super::na::{allocator::Allocator, DefaultAllocator, DimName, MatrixMN, Real, VectorN};

use super::{Dual, Num, One, Scalar};

/// Evaluates the function using dual numbers to get the partial derivative at the input point
#[inline]
pub fn differentiate<T: Scalar + One, F>(x: T, f: F) -> T
where
    F: Fn(Dual<T>) -> Dual<T>,
{
    f(Dual::new(x, T::one())).dual()
}

/// Computes the gradiant of the provided function with a preliminary time parameter.
///
/// # How to read this signature:
/// Let `t` be a parameter, `x` be a state vector and `f` be a function whose gradient is seeked.
/// Calling `nabla` will return ∇f(t, x).
pub fn nabla_t<T: Real + Num, N: DimName, F>(t: T, x: VectorN<T, N>, f: F) -> MatrixMN<T, N, N>
where
    F: Fn(T, &VectorN<Dual<T>, N>) -> VectorN<Dual<T>, N>,
    DefaultAllocator: Allocator<Dual<T>, N> + Allocator<Dual<T>, N, N> + Allocator<usize, N> + Allocator<T, N> + Allocator<T, N, N>,
{
    let mut grad_as_slice = Vec::with_capacity(N::dim() * N::dim());

    // "Simulate" a hyperdual space of size N::dim()
    for v_i in 0..N::dim() {
        let mut dual_x = VectorN::<Dual<T>, N>::zeros();

        for i in 0..N::dim() {
            dual_x[(i, 0)] = Dual::new(x[(i, 0)], if v_i == i { T::one() } else { T::zero() });
        }

        let df_di = f(t, &dual_x);

        for i in 0..N::dim() {
            grad_as_slice.push(df_di[(i, 0)].dual());
        }
    }

    MatrixMN::<T, N, N>::from_column_slice(&grad_as_slice)
}

/// Computes the gradiant of the provided function.
///
/// # How to read this signature:
/// Let `x` be a state vector and `f` be a function whose gradient is seeked.
/// Calling `nabla` will return ∇f(x).
#[inline]
pub fn nabla<T: Real + Num, N: DimName, F>(x: VectorN<T, N>, f: F) -> MatrixMN<T, N, N>
where
    F: Fn(&VectorN<Dual<T>, N>) -> VectorN<Dual<T>, N>,
    DefaultAllocator: Allocator<Dual<T>, N> + Allocator<Dual<T>, N, N> + Allocator<usize, N> + Allocator<T, N> + Allocator<T, N, N>,
{
    nabla_t(T::zero(), x, |_, x| f(x))
}
