use super::na::allocator::Allocator;
use super::na::{DefaultAllocator, DimName, MatrixMN, Scalar};
use super::num_traits::Zero;
use super::{Dual, Float, FloatConst, Num, Signed};
use std::fmt::Debug;

pub fn norm<
    T: 'static + FloatConst + Scalar + Num + Signed + Float + Copy + PartialEq + Debug + Zero,
    M: DimName,
    N: DimName,
>(
    v: &MatrixMN<Dual<T>, M, N>,
) -> Dual<T>
where
    DefaultAllocator: Allocator<Dual<T>, N> + Allocator<Dual<T>, M, N>,
{
    let mut val = Dual::<T>::zero();
    for i in 0..M::dim() {
        for j in 0..N::dim() {
            val = val + v[(i, j)].powi(2);
        }
    }
    val.sqrt()
}
