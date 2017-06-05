extern crate dual_num;

use std::io::Read;

use dual_num::{DualNumber, Float, FloatConst, differentiate};

fn test<F: Float + FloatConst>(x: F) -> F {
    x.sqrt() + F::LN_2()
}

fn main() {
    let result = differentiate(4.0f64, test);

    println!("{:.5}", result);
}