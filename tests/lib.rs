extern crate dual_num;
use dual_num::{differentiate, Dual, Float, FloatConst};

extern crate nalgebra as na;

use dual_num::linalg::norm;
use dual_num::{partials, partials_t};
use na::{Matrix2x6, Matrix3x6, Matrix6, U3, U6, Vector2, Vector3, Vector6};

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
fn type_operations() {
    let mut x = Dual::new(1.0, 2.0);
    let val = 3.0f64;
    abs_within!((x + val).real(), 4.0, std::f64::EPSILON, "add failed on real part");
    abs_within!((x + val).dual(), 2.0, std::f64::EPSILON, "add failed on dual part");
    x += val;
    abs_within!(x.real(), 4.0, std::f64::EPSILON, "add_assign failed on real part");
    abs_within!(x.dual(), 2.0, std::f64::EPSILON, "add_assign failed on dual part");
    abs_within!((x - val).real(), 1.0, std::f64::EPSILON, "sub failed on real part");
    abs_within!((x - val).dual(), 2.0, std::f64::EPSILON, "sub failed on dual part");
    x -= val;
    abs_within!(x.real(), 1.0, std::f64::EPSILON, "sub_assign failed on real part");
    abs_within!(x.dual(), 2.0, std::f64::EPSILON, "sub_assign failed on dual part");
    abs_within!((x * val).real(), 3.0, std::f64::EPSILON, "mul failed on real part");
    abs_within!((x * val).dual(), 6.0, std::f64::EPSILON, "mul failed on dual part");
    x *= val;
    abs_within!(x.real(), 3.0, std::f64::EPSILON, "mul_assign failed on real part");
    abs_within!(x.dual(), 6.0, std::f64::EPSILON, "mul_assign failed on dual part");
    abs_within!((x / val).real(), 1.0, std::f64::EPSILON, "div failed on real part");
    abs_within!((x / val).dual(), 2.0, std::f64::EPSILON, "div failed on dual part");
    x /= val;
    abs_within!(x.real(), 1.0, std::f64::EPSILON, "div_assign failed on real part");
    abs_within!(x.dual(), 2.0, std::f64::EPSILON, "div_assign failed on dual part");
}

#[test]
fn dual_operations() {
    let mut x = Dual::new(1.0, 2.0);
    let y = Dual::new(3.0, 4.0);
    abs_within!((x + y).real(), 4.0, std::f64::EPSILON, "add failed");
    abs_within!((x + y).dual(), 6.0, std::f64::EPSILON, "add failed");
    x += y;
    abs_within!(x.real(), 4.0, std::f64::EPSILON, "add_assign failed");
    abs_within!(x.dual(), 6.0, std::f64::EPSILON, "add_assign failed");
    abs_within!((x - y).real(), 1.0, std::f64::EPSILON, "sub failed");
    abs_within!((x - y).dual(), 2.0, std::f64::EPSILON, "sub failed");
    x -= y;
    abs_within!(x.real(), 1.0, std::f64::EPSILON, "sub_assign failed");
    abs_within!(x.dual(), 2.0, std::f64::EPSILON, "sub_assign failed");
    abs_within!((x * y).real(), 3.0, std::f64::EPSILON, "mul failed");
    abs_within!((x * y).dual(), 10.0, std::f64::EPSILON, "mul failed");
    x *= y;
    abs_within!(x.real(), 3.0, std::f64::EPSILON, "mul_assign failed");
    abs_within!(x.dual(), 10.0, std::f64::EPSILON, "mul_assign failed");
    abs_within!((x / y).real(), 1.0, std::f64::EPSILON, "div failed");
    abs_within!((x / y).dual(), 2.0, std::f64::EPSILON, "div failed");
    x /= y;
    abs_within!(x.real(), 1.0, std::f64::EPSILON, "div_assign failed");
    abs_within!(x.dual(), 2.0, std::f64::EPSILON, "div_assign failed");
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

#[test]
fn linalg() {
    let x = Vector2::new(Dual::from(1.0f64), Dual::new(-2.0f64, 3.5f64));
    let val = Dual::new(2.0f64, 0.5f64);
    let computed = x * val;
    let expected = Vector2::new(Dual::new(2.0f64, 0.5f64), Dual::new(-4.0f64, 6.0f64));
    for i in 0..2 {
        zero_within!(
            (expected - computed)[(i, 0)].real(),
            1e-16,
            format!("Vector2 multiplication incorrect (i={})", i)
        );
        zero_within!(
            (expected - computed)[(i, 0)].dual(),
            1e-16,
            format!("Vector2 multiplication incorrect (i={})", i)
        );
    }
}

#[test]
fn partials_no_param() {
    // This is an example of the sensitivity matrix (H tilde) of a ranging method.
    fn sensitivity(state: &Matrix6<Dual<f64>>) -> Matrix2x6<Dual<f64>> {
        let range_mat = state.fixed_slice::<U3, U6>(0, 0).into_owned();
        let velocity_mat = state.fixed_slice::<U3, U6>(3, 0).into_owned();
        let mut range_slice = Vec::with_capacity(6);
        let mut range_rate_slice = Vec::with_capacity(6);
        for j in 0..6 {
            let range = norm(&Vector3::new(range_mat[(0, j)], range_mat[(1, j)], range_mat[(2, j)]));
            range_slice.push(range);
            let mut range_rate = Dual::from(0f64);
            for i in 0..3 {
                range_rate += range_mat[(i, j)] * velocity_mat[(i, j)] / range;
                // println!(
                //     "range_mat[({0}, {1})] ({2}) * velocity_mat[({0}, {1})] {3} = {4}",
                //     i,
                //     j,
                //     range_mat[(i, j)].real(),
                //     (velocity_mat[(i, j)] / range).real(),
                //     (range_mat[(i, j)] * velocity_mat[(i, j)] / range).real()
                // );
            }
            println!("{} => {:.10}", j, range_rate);
            range_rate_slice.push(range_rate);
        }
        let mut rtn = Matrix2x6::zeros();
        rtn.set_row(0, &Vector6::from_row_slice(&range_slice).transpose());
        rtn.set_row(1, &Vector6::from_row_slice(&range_rate_slice).transpose());
        rtn
    }

    let rx = Vector6::new(
        9203.99371643433,
        18450.60691397028,
        7016.41093973247,
        -3.769097909034852,
        1.2551002399532656,
        1.6440346105270638,
    );
    let tx = Vector6::new(
        4849.340232977386,
        360.4155470897664,
        4114.752758150833,
        -0.026281916700417594,
        0.35361947364427143,
        0.000000,
    );

    let (fx, dfdx) = partials(rx - tx, sensitivity);

    let expected_fx = Vector2::new(18831.82547853717, 0.2538107291309079);
    zero_within!(
        (fx - expected_fx).norm(),
        1e-16,
        format!("f(x) computation is incorrect -- here comes the delta: {}", fx - expected_fx)
    );

    let mut expected_dfdx = Matrix2x6::zeros();
    expected_dfdx[(0, 0)] = 0.23123905265689662;
    expected_dfdx[(0, 1)] = 0.9606180445702461;
    expected_dfdx[(0, 2)] = 0.15408268225981;
    expected_dfdx[(1, 3)] = -0.00019563292172470923;
    expected_dfdx[(1, 4)] = 0.000060817042613472664;
    expected_dfdx[(1, 5)] = 0.00008937755133596409;
    expected_dfdx[(1, 3)] = 0.23123905265689662;
    expected_dfdx[(1, 4)] = 0.9606180445702461;
    expected_dfdx[(1, 5)] = 0.15408268225981;

    zero_within!(
        (dfdx - expected_dfdx).norm(),
        1e-16,
        format!("partial computation is incorrect -- here comes the delta: {}", dfdx - expected_dfdx)
    );
}
