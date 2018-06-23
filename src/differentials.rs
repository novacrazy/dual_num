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

/// Computes the state and the partials matrix of the provided function with a preliminary time parameter.
///
/// # How to read this signature:
/// Let `t` be a parameter, `x` be a state vector and `f` be a function whose gradient is seeked.
/// Calling `partials_t` will return a tuple of ( f(t, x), ∂f(t, x)/∂x ).
/// If the function `f` is defined as f: Y^{n} → Y^{n}, where Y is a given group,
/// than `partials` returns the evaluation and gradient of `f` at position `x`, i.e.
/// a tuple of ( f(t, x), ∇f(t, x) ).
pub fn partials_t<T: Real + Num, M: DimName, N: DimName, F>(t: T, x: VectorN<T, M>, f: F) -> (VectorN<T, N>, MatrixMN<T, N, M>)
where
    F: Fn(T, &MatrixMN<Dual<T>, M, M>) -> MatrixMN<Dual<T>, N, M>,
    DefaultAllocator: Allocator<Dual<T>, M>
        + Allocator<Dual<T>, M>
        + Allocator<Dual<T>, N, M>
        + Allocator<Dual<T>, M, M>
        + Allocator<usize, N>
        + Allocator<T, N>
        + Allocator<T, M>
        + Allocator<T, N, M>,
{
    // Create a Matrix for the hyperdual space
    let mut hyperdual_space = MatrixMN::<Dual<T>, M, M>::zeros();

    for i in 0..M::dim() {
        let mut v_i = VectorN::<Dual<T>, M>::zeros();
        for j in 0..M::dim() {
            v_i[(j, 0)] = Dual::new(x[(j, 0)], if i == j { T::one() } else { T::zero() });
        }
        hyperdual_space.set_column(i, &v_i);
    }

    let state_n_grad = f(t, &hyperdual_space);
    // Extract the dual part
    let mut state = VectorN::<T, N>::zeros();
    let mut grad = MatrixMN::<T, N, M>::zeros();
    for i in 0..N::dim() {
        for j in 0..M::dim() {
            if j == 0 {
                // The state is duplicated in every column
                state[(i, 0)] = state_n_grad[(i, 0)].real();
            }
            grad[(i, j)] = state_n_grad[(i, j)].dual();
        }
    }
    (state, grad)
}

/// Computes the state and partials of the provided function.
///
/// # How to read this signature:
/// Let `x` be a state vector and `f` be a function whose gradient is seeked.
/// Calling `partials` will return a tuple of ( f(x), ∂f(x)/∂x ).
/// If the function `f` is defined as f: Y^{n} → Y^{n}, where Y is a given group,
/// than `partials` returns the evaluation and gradient of `f` at position `x`, i.e.
/// a tuple of ( f(x), ∇f(x) ).
#[inline]
pub fn partials<T: Real + Num, M: DimName, N: DimName, F>(x: VectorN<T, M>, f: F) -> (VectorN<T, N>, MatrixMN<T, N, M>)
where
    F: Fn(&MatrixMN<Dual<T>, M, M>) -> MatrixMN<Dual<T>, N, M>,
    DefaultAllocator: Allocator<Dual<T>, M>
        + Allocator<Dual<T>, M>
        + Allocator<Dual<T>, N, M>
        + Allocator<Dual<T>, M, M>
        + Allocator<usize, N>
        + Allocator<T, N>
        + Allocator<T, M>
        + Allocator<T, N, M>,
{
    partials_t(T::zero(), x, |_, x| f(x))
}
