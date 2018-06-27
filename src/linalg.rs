use na::allocator::Allocator;
use na::{DefaultAllocator, DimName, MatrixMN, Scalar};

use {Dual, Float, Zero};

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
