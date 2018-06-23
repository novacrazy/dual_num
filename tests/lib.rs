extern crate dual_num;
use dual_num::{differentiate, Dual, Float, FloatConst};

extern crate nalgebra as na;

use dual_num::linalg::norm;
use dual_num::{partials, partials_t};
use na::{Matrix2x6, Matrix3x6, Matrix6, Matrix6x2, U3, U6, Vector3, Vector6};

macro_rules! abs_within {
    ($x:expr, $val:expr, $eps:expr, $msg:expr) => {
        assert!(($x - $val).abs() < $eps, $msg)
    };
}

macro_rules! zero_within {
    ($x:expr, $eps:expr, $msg:expr) => {
        assert!($x.abs() < $eps, $msg)
    };
}

#[test]
fn derive() {
    abs_within!(
        differentiate(4.0f64, |x| x.sqrt() + Dual::from_real(1.0)),
        1.0 / 4.0,
        std::f64::EPSILON,
        "incorrect norm"
    );

    println!(
        "{:.16}",
        differentiate(1.0f64, |x| {
            let one = Dual::from_real(1.0); // Or use the One trait

            one / (one + Dual::E().powf(-x))
        })
    );

    println!("{:.5}", (Dual::new(0.25f32, 1.0) * Dual::PI()).sin());

    let mut x = Dual::new(2i32, 1);

    x = x * x + x;

    assert_eq!(x.real(), 6i32, "incorrect real");
    assert_eq!(x.dual(), 5i32, "incorrect real");
}

#[test]
fn test_norm() {
    let vec = Vector3::new(Dual::from_real(1.0), Dual::from_real(1.0), Dual::from_real(1.0));
    let this_norm = norm(&vec);
    abs_within!(this_norm.real(), 3.0f64.sqrt(), std::f64::EPSILON, "incorrect real part of the norm");
    zero_within!(this_norm.dual(), std::f64::EPSILON, "incorrect dual part of the norm");
}

#[test]
fn square_gradient_no_param() {
    // This is an example of the equation of motion gradient for a spacecrate in a two body acceleration.
    fn eom(state: &Matrix6<Dual<f64>>) -> Matrix6<Dual<f64>> {
        let radius = state.fixed_slice::<U3, U6>(0, 0).into_owned();
        let velocity = state.fixed_slice::<U3, U6>(3, 0).into_owned();
        let mut body_acceleration = Matrix3x6::zeros();
        for i in 0..3 {
            let this_norm = norm(&Vector3::new(radius[(0, i)], radius[(1, i)], radius[(2, i)]));
            let body_acceleration_f = Dual::from_real(-398_600.4415) / this_norm.powi(3);
            let this_body_acceleration = Vector3::new(
                radius[(0, i)] * body_acceleration_f,
                radius[(1, i)] * body_acceleration_f,
                radius[(2, i)] * body_acceleration_f,
            );
            body_acceleration.set_column(i, &this_body_acceleration);
        }
        let mut rtn = Matrix6::zeros();
        for i in 0..6 {
            if i < 3 {
                rtn.set_row(i, &velocity.row(i));
            } else {
                rtn.set_row(i, &body_acceleration.row(i - 3));
            }
        }
        rtn
    }

    let state = Vector6::new(
        -9042.862233600335,
        18536.333069123244,
        6999.9570694864115,
        -3.28878900377057,
        -2.226285193102822,
        1.6467383807226765,
    );
    let (fx, grad) = partials(state, eom);

    let expected_fx = Vector6::new(
        -3.28878900377057,
        -2.226285193102822,
        1.6467383807226765,
        0.0003488751720191492,
        -0.0007151349009902908,
        -0.00027005954128877916,
    );

    zero_within!((fx - expected_fx).norm(), 1e-16, "f(x) computation is incorrect");

    let mut expected = Matrix6::zeros();
    expected[(0, 3)] = 1.0;
    expected[(1, 4)] = 1.0;
    expected[(2, 5)] = 1.0;
    expected[(3, 0)] = -0.000000018628398676538285;
    expected[(4, 0)] = -0.00000004089774775108092;
    expected[(5, 0)] = -0.0000000154443965496673;
    expected[(3, 1)] = -0.00000004089774775108092;
    expected[(4, 1)] = 0.000000045253271751873843;
    expected[(5, 1)] = 0.00000003165839212196757;
    expected[(3, 2)] = -0.0000000154443965496673;
    expected[(4, 2)] = 0.00000003165839212196757;
    expected[(5, 2)] = -0.000000026624873075335538;

    zero_within!((grad - expected).norm(), 1e-16, "gradient computation is incorrect");
}

#[test]
fn square_gradient_with_param() {
    // This is an example of the equation of motion gradient for a spacecrate in a two body acceleration.
    fn eom(_t: f64, state: &Matrix6<Dual<f64>>) -> Matrix6<Dual<f64>> {
        let radius = state.fixed_slice::<U3, U6>(0, 0).into_owned();
        let velocity = state.fixed_slice::<U3, U6>(3, 0).into_owned();
        let mut body_acceleration = Matrix3x6::zeros();
        for i in 0..3 {
            let this_norm = norm(&Vector3::new(radius[(0, i)], radius[(1, i)], radius[(2, i)]));
            let body_acceleration_f = Dual::from_real(-398_600.4415) / this_norm.powi(3);
            let this_body_acceleration = Vector3::new(
                radius[(0, i)] * body_acceleration_f,
                radius[(1, i)] * body_acceleration_f,
                radius[(2, i)] * body_acceleration_f,
            );
            body_acceleration.set_column(i, &this_body_acceleration);
        }
        let mut rtn = Matrix6::zeros();
        for i in 0..6 {
            if i < 3 {
                rtn.set_row(i, &velocity.row(i));
            } else {
                rtn.set_row(i, &body_acceleration.row(i - 3));
            }
        }
        rtn
    }

    let state = Vector6::new(
        -9042.862233600335,
        18536.333069123244,
        6999.9570694864115,
        -3.28878900377057,
        -2.226285193102822,
        1.6467383807226765,
    );

    let (fx, grad) = partials_t(0.0, state, eom);

    let expected_fx = Vector6::new(
        -3.28878900377057,
        -2.226285193102822,
        1.6467383807226765,
        0.0003488751720191492,
        -0.0007151349009902908,
        -0.00027005954128877916,
    );

    zero_within!((fx - expected_fx).norm(), 1e-16, "f(x) computation is incorrect");

    let mut expected = Matrix6::zeros();

    expected[(0, 3)] = 1.0;
    expected[(1, 4)] = 1.0;
    expected[(2, 5)] = 1.0;
    expected[(3, 0)] = -0.000000018628398676538285;
    expected[(4, 0)] = -0.00000004089774775108092;
    expected[(5, 0)] = -0.0000000154443965496673;
    expected[(3, 1)] = -0.00000004089774775108092;
    expected[(4, 1)] = 0.000000045253271751873843;
    expected[(5, 1)] = 0.00000003165839212196757;
    expected[(3, 2)] = -0.0000000154443965496673;
    expected[(4, 2)] = 0.00000003165839212196757;
    expected[(5, 2)] = -0.000000026624873075335538;

    zero_within!((grad - expected).norm(), 1e-16, "gradient computation is incorrect");
}
/*
#[test]
fn nonsquare_gradient_no_param() {
    // This is an example of the equation of motion gradient for a spacecrate in a two body acceleration.
    fn sensitivity(state: &Matrix2x6<Dual<f64>>) -> Matrix2x6<Dual<f64>> {
        panic!("{}", state);
        let radius = state.fixed_slice::<U3, U6>(0, 0).into_owned();
        let velocity = state.fixed_slice::<U3, U6>(3, 0).into_owned();
        let mut body_acceleration = Matrix3x6::zeros();
        for i in 0..3 {
            let this_norm = norm(&Vector3::new(radius[(0, i)], radius[(1, i)], radius[(2, i)]));
            let body_acceleration_f = Dual::from_real(-398_600.4415) / this_norm.powi(3);
            let this_body_acceleration = Vector3::new(
                radius[(0, i)] * body_acceleration_f,
                radius[(1, i)] * body_acceleration_f,
                radius[(2, i)] * body_acceleration_f,
            );
            body_acceleration.set_column(i, &this_body_acceleration);
        }
        let mut rtn = Matrix2x6::zeros();
        // for i in 0..6 {
        //     if i < 3 {
        //         rtn.set_row(i, &velocity.row(i));
        //     } else {
        //         rtn.set_row(i, &body_acceleration.row(i - 3));
        //     }
        // }
        rtn
    }

    let rx = Vector6::new(9203.993716, 18450.606914, 7016.410940, -3.769098, 1.255100, 1.644035);
    let tx = Vector6::new(4849.340233, 360.415547, 4114.752758, -0.026282, 0.353619, 0.000000);

    let (_, grad) = partials(rx - tx, sensitivity);

    /*
┌                                                                                                                                                 ┐
│     0.23123905265689662      0.9606180445702461        0.15408268225981                       0                       0                       0 │
│ -0.00019563292172470923 0.000060817042613472664  0.00008937755133596409     0.23123905265689662      0.9606180445702461        0.15408268225981 │
└                                                                                                                                                 ┘
*/

    let mut expected_grad = Matrix2x6::zeros();
    expected_grad[(0, 0)] = 0.23123905265689662;
    expected_grad[(0, 1)] = 0.9606180445702461;
    expected_grad[(0, 2)] = 0.15408268225981;
    expected_grad[(1, 3)] = -0.00019563292172470923;
    expected_grad[(1, 4)] = 0.000060817042613472664;
    expected_grad[(1, 5)] = 0.00008937755133596409;
    expected_grad[(1, 3)] = 0.23123905265689662;
    expected_grad[(1, 4)] = 0.9606180445702461;
    expected_grad[(1, 5)] = 0.15408268225981;

    zero_within!((grad - expected_grad).norm(), 1e-16, "gradient computation is incorrect");
}
*/
