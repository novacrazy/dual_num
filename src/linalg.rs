use na::allocator::Allocator;
use na::{DefaultAllocator, DimName, MatrixMN, Scalar, VectorN};

use {Dual, Float, Hyperdual, Zero};

pub fn norm<T: Scalar + Float, M: DimName, N: DimName>(v: &MatrixMN<Dual<T>, M, N>) -> Dual<T>
where
    Dual<T>: Float,
    DefaultAllocator: Allocator<Dual<T>, N> + Allocator<Dual<T>, M, N>,
{
    let mut val = Dual::zero();

    for i in 0..M::dim() {
        for j in 0..N::dim() {
            val = val + v[(i, j)].powi(2);
        }
    }

    val.sqrt()
}

// TODO: Replace all ` norm(` calls with hnorm, and then rename this function.
pub fn hnorm<T: Scalar + Float, M: DimName, N: DimName>(v: &VectorN<Hyperdual<T, N>, M>) -> Hyperdual<T, N>
where
    Hyperdual<T, N>: Float,
    DefaultAllocator: Allocator<Hyperdual<T, N>, M> + Allocator<T, N>,
    <DefaultAllocator as Allocator<T, N>>::Buffer: Copy,
{
    let mut val = Hyperdual::<T, N>::zero();

    for i in 0..M::dim() {
        val = val + v[i].powi(2);
    }

    val.sqrt()
}
