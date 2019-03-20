extern crate dual_num;
extern crate nalgebra as na;

use na::{Matrix2x6, Matrix6, Vector2, Vector3, Vector6, VectorN, U2, U3, U6, U7};

use dual_num::linalg::{hnorm, norm};
use dual_num::{differentiate, Dual, DualN, Float, FloatConst, Hyperdual};
use dual_num::{partials, DimName};

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
fn default() {
    assert_eq!(Dual::<f64>::default(), Dual::new(0., 0.));
}

#[test]
fn sum_product() {
    let a = [Dual::new(1.0, 1.0), Dual::new(0.5, 0.5)];
    assert_eq!(a.iter().cloned().sum::<Dual<f64>>(), Dual::new(1.5, 1.5));
    assert_eq!(a.iter().cloned().product::<Dual<f64>>(), Dual::new(0.5, 1.0));
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

    let c = Dual::new(1.0 / 2f64.sqrt(), 1.0).asin();
    abs_within!(c.dual(), 1.414213562373095, std::f64::EPSILON, "incorrect d/dx arcsin");

    let c = Dual::new(1.0 / 2f64.sqrt(), 1.0).acos();
    abs_within!(c.dual(), -1.414213562373095, std::f64::EPSILON, "incorrect d/dx arccos");

    let c = Dual::new(1.0 / 2f64.sqrt(), 1.0).atan();
    abs_within!(c.dual(), 2.0f64 / 3.0f64, std::f64::EPSILON, "incorrect d/dx arctan");
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
fn linalg() {
    // NOTE: Due to the implementation of std::ops::Mul in nalgebra, the syntax _must_ be vec * x
    // where x is the scalar and vec the vector.
    // Quote from the author, sebcrozet
    // > The thing is that nalgebra cannot define the multiplication of a scalar by a vector
    // > (where the scalar is on the left hand side) because such an implementation would look like
    // > this: `impl<T: Scalar> Mul<Vector<T>> for T` which is forbidden by the compiler. That's
    // > why the only multiplication automatically provided by nalgebra is when the scalar is on
    // > the right-hand-side. When `T` here is `f32` or `f64` both multiplication orders work.
    let vec = Vector2::new(Dual::from(1.0f64), Dual::new(-2.0f64, 3.5f64));
    let x = Dual::new(2.0f64, 0.5f64);
    let computed = vec * x;
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

    // Checking the dot product
    // NOTE: The tolerance is relatively high because of some rounding error probably due to the powi call.
    let delta = computed.dot(&expected) - norm(&computed).powi(2);
    zero_within!(delta.real(), 1e-12, "real part of the dot product is incorrect");
    zero_within!(delta.dual(), 1e-12, "dual part of the dot product is incorrect");

    let vec = Vector3::new(Dual::from_real(1.0), Dual::from_real(1.0), Dual::from_real(1.0));
    let this_norm = norm(&vec);
    abs_within!(this_norm.real(), 3.0f64.sqrt(), std::f64::EPSILON, "incorrect real part of the norm");
    zero_within!(this_norm.dual(), std::f64::EPSILON, "incorrect dual part of the norm");
}

#[test]
fn multivariate() {
    // find partial derivative at x=4.0, y=5.0 for f(x,y)=x^2+sin(x*y)+y^3
    let x: Hyperdual<f64, U3> = Hyperdual::from_slice(&[4.0, 1.0, 0.0]);
    // DualN and Hyperdual are interchangeable aliases. Hyperdual is the name from Fike 2012
    // whereas multi-dual is from Revel et al. 2016.
    let y: DualN<f64, U3> = Hyperdual::from_slice(&[5.0, 0.0, 1.0]);

    let res = x * x + (x * y).sin() + y.powi(3);
    zero_within!((res[0] - 141.91294525072763), 1e-13, format!("f(4, 5) incorrect"));
    zero_within!((res[1] - 10.04041030906696), 1e-13, format!("df/dx(4, 5) incorrect"));
    zero_within!((res[2] - 76.63232824725357), 1e-13, format!("df/dy(4, 5) incorrect"));
}

#[test]
fn state_gradient() {
    // This is an example of the equation of motion gradient for a spacecrate in a two body acceleration.
    fn eom(_t: f64, state: &VectorN<Hyperdual<f64, U7>, U6>) -> (Vector6<f64>, Matrix6<f64>) {
        // Extract data from hyperspace
        let radius = state.fixed_rows::<U3>(0).into_owned();
        let velocity = state.fixed_rows::<U3>(3).into_owned();

        // Code up math as usual
        let rmag = hnorm(&radius);
        let body_acceleration = radius * (Hyperdual::<f64, U7>::from_real(-398_600.4415) / rmag.powi(3));

        // Added for inspection only
        dbg!(velocity);
        dbg!(body_acceleration);

        // Extract result into Vector6 and Matrix6
        let mut fx = Vector6::zeros();
        let mut grad = Matrix6::zeros();
        for i in 0..U6::dim() {
            fx[i] = if i < 3 { velocity[i].real() } else { body_acceleration[i - 3].real() };
            for j in 1..U7::dim() {
                grad[(i, j - 1)] = if i < 3 { velocity[i][j] } else { body_acceleration[i - 3][j] };
            }
        }

        (fx, grad)
    }

    let state = Vector6::new(
        -9042.862233600335,
        18536.333069123244,
        6999.9570694864115,
        -3.28878900377057,
        -2.226285193102822,
        1.6467383807226765,
    );

    let hyperstate = VectorN::<Hyperdual<f64, U7>, U6>::from_row_slice(&[
        Hyperdual::<f64, U7>::from_slice(&[state[0], 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        Hyperdual::<f64, U7>::from_slice(&[state[1], 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        Hyperdual::<f64, U7>::from_slice(&[state[2], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        Hyperdual::<f64, U7>::from_slice(&[state[3], 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        Hyperdual::<f64, U7>::from_slice(&[state[4], 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        Hyperdual::<f64, U7>::from_slice(&[state[5], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    ]);

    let (fx, grad) = eom(0.0, &hyperstate);

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
fn state_partials() {
    // This is an example of the sensitivity matrix (H tilde) of a ranging method.
    fn sensitivity(state: &VectorN<Hyperdual<f64, U7>, U6>) -> (Vector2<f64>, Matrix2x6<f64>) {
        // Extract data from hyperspace
        let range_vec = state.fixed_rows::<U3>(0).into_owned();
        let velocity_vec = state.fixed_rows::<U3>(3).into_owned();

        // Code up math as usual
        let delta_v_vec = velocity_vec / hnorm(&range_vec);
        let range = hnorm(&range_vec);
        let range_rate = range_vec.dot(&delta_v_vec);

        // Added for inspection only
        dbg!(range);
        dbg!(range_rate);

        // Extract result into Vector2 and Matrix2x6
        let mut fx = Vector2::zeros();
        let mut pmat = Matrix2x6::zeros();
        for i in 0..U2::dim() {
            fx[i] = if i == 0 { range.real() } else { range_rate.real() };
            for j in 1..U7::dim() {
                pmat[(i, j - 1)] = if i == 0 { range[j] } else { range_rate[j] };
            }
        }

        (fx, pmat)
    }

    let vec = Vector6::new(
        4354.65348345694383169757202267646790,
        18090.19136688051366945728659629821777,
        2901.65818158163710904773324728012085,
        -3.74281599233443440510882282978855,
        0.90148076630899409700248270382872,
        1.64403461052706378886512084136484,
    );

    let hyperstate = VectorN::<Hyperdual<f64, U7>, U6>::from_row_slice(&[
        Hyperdual::<f64, U7>::from_slice(&[vec[0], 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        Hyperdual::<f64, U7>::from_slice(&[vec[1], 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        Hyperdual::<f64, U7>::from_slice(&[vec[2], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        Hyperdual::<f64, U7>::from_slice(&[vec[3], 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        Hyperdual::<f64, U7>::from_slice(&[vec[4], 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        Hyperdual::<f64, U7>::from_slice(&[vec[5], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    ]);

    let (fx, dfdx) = sensitivity(&hyperstate);

    let expected_fx = Vector2::new(18831.82547853717, 0.2538107291309079);

    zero_within!(
        (fx - expected_fx).norm(),
        1e-20,
        format!("f(x) computation is incorrect -- here comes the delta: {}", fx - expected_fx)
    );

    let mut expected_dfdx = Matrix2x6::zeros();
    expected_dfdx[(0, 0)] = 0.23123905265689662091;
    expected_dfdx[(0, 1)] = 0.96061804457024613235;
    expected_dfdx[(0, 2)] = 0.15408268225981000543;
    expected_dfdx[(1, 0)] = -0.00020186608829958833;
    expected_dfdx[(1, 1)] = 0.00003492309339579752;
    expected_dfdx[(1, 2)] = 0.00008522417406774546;
    expected_dfdx[(1, 3)] = 0.23123905265689662091;
    expected_dfdx[(1, 4)] = 0.96061804457024613235;
    expected_dfdx[(1, 5)] = 0.15408268225981000543;

    zero_within!(
        (dfdx - expected_dfdx).norm(),
        1e-20,
        format!("partial computation is incorrect -- here comes the delta: {}", dfdx - expected_dfdx)
    );
}
