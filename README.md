dual_num [![Build Status](https://travis-ci.org/novacrazy/dual_num.svg?branch=master)](https://travis-ci.org/novacrazy/dual_num)
========

Fully-featured Dual Number implementation with features for automatic differentiation of multivariate vectorial functions into gradients.

## Usage

```rust
extern crate dual_num;

use dual_num::{Dual, Hyperdual, Float, differentiate, U3};

fn main() {
    // find partial derivative at x=4.0
    let univariate = differentiate(4.0f64, |x| x.sqrt() + Dual::from_real(1.0));
    assert!((univariate - 0.4500).abs() < 1e-16, "wrong derivative");

    // find the partial derivatives of a multivariate function
    let x: Hyperdual<f64, U3> = Hyperdual::from_slice(&[4.0, 1.0, 0.0]);
    let y: Hyperdual<f64, U3> = Hyperdual::from_slice(&[5.0, 0.0, 1.0]);

    let multivariate = x * x + (x * y).sin() + y.powi(3);
    assert!((res[0] - 141.91294525072763).abs() < 1e-13, "f(4, 5) incorrect");
    assert!((res[1] - 10.04041030906696).abs() < 1e-13, "df/dx(4, 5) incorrect");
    assert!((res[2] - 76.63232824725357).abs() < 1e-13, "df/dy(4, 5) incorrect");
}
```

##### Previous Work
* [https://github.com/FreeFull/dual_numbers](https://github.com/FreeFull/dual_numbers)
* [https://github.com/ibab/rust-ad](https://github.com/ibab/rust-ad)
* [https://github.com/tesch1/cxxduals](https://github.com/tesch1/cxxduals)
