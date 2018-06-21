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

/// Computes the state and the gradiant of the provided function with a preliminary time parameter.
///
/// # How to read this signature:
/// Let `t` be a parameter, `x` be a state vector and `f` be a function whose gradient is seeked.
/// Calling `nabla_t` will return a tuple of ( f(t, x), ∇f(t, x) ).
pub fn nabla_t<T: Real + Num, N: DimName, F>(t: T, x: VectorN<T, N>, f: F) -> (VectorN<T, N>, MatrixMN<T, N, N>)
where
    F: Fn(T, &MatrixMN<Dual<T>, N, N>) -> MatrixMN<Dual<T>, N, N>,
    DefaultAllocator: Allocator<Dual<T>, N> + Allocator<Dual<T>, N, N> + Allocator<usize, N> + Allocator<T, N> + Allocator<T, N, N>,
{
    // Create a Matrix for the hyperdual space
    let mut hyperdual_space = MatrixMN::<Dual<T>, N, N>::zeros();

    for i in 0..N::dim() {
        let mut v_i = VectorN::<Dual<T>, N>::zeros();
        for j in 0..N::dim() {
            v_i[(j, 0)] = Dual::new(x[(j, 0)], if i == j { T::one() } else { T::zero() });
        }
        hyperdual_space.set_column(i, &v_i);
    }

    let state_n_grad = f(t, &hyperdual_space);
    // Extract the dual part
    let mut state = VectorN::<T, N>::zeros();
    let mut grad = MatrixMN::<T, N, N>::zeros();
    for i in 0..N::dim() {
        for j in 0..N::dim() {
            if j == 0 {
                // The state is duplicated in every column
                state[(i, j)] = state_n_grad[(i, j)].real();
            }
            grad[(i, j)] = state_n_grad[(i, j)].dual();
        }
    }
    (state, grad)
}

/// Computes the state and gradiant of the provided function.
///
/// # How to read this signature:
/// Let `x` be a state vector and `f` be a function whose gradient is seeked.
/// Calling `nabla` will return a tuple of ( f(x), ∇f(x) ).
#[inline]
pub fn nabla<T: Real + Num, N: DimName, F>(x: VectorN<T, N>, f: F) -> (VectorN<T, N>, MatrixMN<T, N, N>)
where
    F: Fn(&MatrixMN<Dual<T>, N, N>) -> MatrixMN<Dual<T>, N, N>,
    DefaultAllocator: Allocator<Dual<T>, N> + Allocator<Dual<T>, N, N> + Allocator<usize, N> + Allocator<T, N> + Allocator<T, N, N>,
{
    nabla_t(T::zero(), x, |_, x| f(x))
}
