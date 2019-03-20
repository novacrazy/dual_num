use na::allocator::Allocator;
use na::{DefaultAllocator, DimName, Scalar, VectorN};

use {Float, Hyperdual, Zero};

/// Computes the norm of a vector of Hyperdual.
pub fn norm<T: Scalar + Float, M: DimName, N: DimName>(v: &VectorN<Hyperdual<T, N>, M>) -> Hyperdual<T, N>
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
