extern crate dual_num;

use std::io::Read;

use dual_num::{DualNumber, Float, differentiate};

fn test<F: Float>(x: F) -> F {
    x.sqrt() + F::from(1.0).unwrap()
}

fn main() {
    let result = differentiate(4.0f64, test);

    println!("{:.5}", result);
}