extern crate dual_num;
extern crate nalgebra as na;

use dual_num::{differentiate, nabla, nabla_t, Dual, Float, FloatConst};
use na::{Matrix6, U3, Vector3, Vector6};

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

    println!(
        "{:.5}",
        Dual::new(0.25f32, 1.0).map(|x| (x * Dual::PI()).sin())
    );

    let x = Dual::new(2i32, 1).map(|x| x * x + x);
    assert_eq!(x.real(), 6i32, "incorrect real");
    assert_eq!(x.dual(), 5i32, "incorrect real");
}

#[test]
fn norms() {
    let vec = Vector3::new(
        Dual::from_real(1.0),
        Dual::from_real(1.0),
        Dual::from_real(1.0),
    );
    // abs_within!(vec.norm(), 1.0, std::f64::EPSILON, "incorrect norm");
}

#[test]
fn gradient_no_param() {
    // This is an example of the equation of motion gradient for a spacecrate in a two body acceleration.
    fn eom(state: &Vector6<Dual<f64>>) -> Vector6<Dual<f64>> {
        let radius = state.fixed_rows::<U3>(0).into_owned();
        let velocity = state.fixed_rows::<U3>(3).into_owned();
        let norm =
            (radius[(0, 0)].powi(2) + radius[(1, 0)].powi(2) + radius[(2, 0)].powi(2)).sqrt();
        let body_acceleration_f = Dual::from_real(-398_600.4415) / norm.powi(3);
        let body_acceleration = Vector3::new(
            radius[(0, 0)] * body_acceleration_f,
            radius[(1, 0)] * body_acceleration_f,
            radius[(2, 0)] * body_acceleration_f,
        );
        Vector6::from_iterator(velocity.iter().chain(body_acceleration.iter()).cloned())
    }

    let state = Vector6::new(
        -9042.862233600335,
        18536.333069123244,
        6999.9570694864115,
        -3.28878900377057,
        -2.226285193102822,
        1.6467383807226765,
    );
    let grad = nabla(state, eom);

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

    zero_within!(
        (grad - expected).norm(),
        1e-16,
        "gradient computation is incorrect"
    );
}

#[test]
fn gradient_with_param() {
    // This is an example of the equation of motion gradient for a spacecrate in a two body acceleration.
    fn eom(_t: f64, state: &Vector6<Dual<f64>>) -> Vector6<Dual<f64>> {
        let radius = state.fixed_rows::<U3>(0).into_owned();
        let velocity = state.fixed_rows::<U3>(3).into_owned();
        let norm =
            (radius[(0, 0)].powi(2) + radius[(1, 0)].powi(2) + radius[(2, 0)].powi(2)).sqrt();
        let body_acceleration_f = Dual::from_real(-398_600.4415) / norm.powi(3);
        let body_acceleration = Vector3::new(
            radius[(0, 0)] * body_acceleration_f,
            radius[(1, 0)] * body_acceleration_f,
            radius[(2, 0)] * body_acceleration_f,
        );
        Vector6::from_iterator(velocity.iter().chain(body_acceleration.iter()).cloned())
    }

    let state = Vector6::new(
        -9042.862233600335,
        18536.333069123244,
        6999.9570694864115,
        -3.28878900377057,
        -2.226285193102822,
        1.6467383807226765,
    );
    let grad = nabla_t(0.0, state, eom);

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

    zero_within!(
        (grad - expected).norm(),
        1e-16,
        "gradient computation is incorrect"
    );
}
