extern crate dual_num;
extern crate nalgebra as na;

use na::dimension::U3;
use dual_num::{DualN, Float};

fn main() {
    // find partial derivative at x=4.0, y=5.0 for f(x,y)=x^2+sin(x*y)+y^3
    let x: DualN<f64,U3> = DualN::from_slice(&[4.0, 1.0, 0.0]);
    let y: DualN<f64,U3> = DualN::from_slice(&[5.0, 0.0, 1.0]);

    let res = x*x + (x*y).sin() + y.powi(3);
    println!("Dual output={}", res);
    println!("f(4,5)={}, df/dx_4,5={}, df/dy_4,5={}", res[0], res[1], res[2]);
}