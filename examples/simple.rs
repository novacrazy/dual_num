extern crate dual_num;

use dual_num::{DualNumber, Float, FloatConst, differentiate};

fn main() {
    println!("{:.5}", differentiate(4.0f64, |x| {
        x.sqrt() + DualNumber::from_real(1.0)
    }));

    println!("{:.5}", differentiate(1.0f64, |x| {
        let one = DualNumber::from_real(1.0); // Or use the One trait

        one / (one + DualNumber::E().powf(-x))
    }));

    println!("{:.5}", DualNumber::new(0.25f32, 1.0).map(|x| {
        (x * DualNumber::PI()).sin()
    }));

    println!("{:.5}", DualNumber::new(2i32, 1).map(|x| {
        x * x + x
    }));
}