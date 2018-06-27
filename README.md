dual_num [![Build Status](https://travis-ci.org/novacrazy/dual_num.svg?branch=master)](https://travis-ci.org/novacrazy/dual_num)
========

Fully-featured Dual Number implementation with features for automatic differentiation of multivariate vectorial functions into gradients.

## Usage

```rust
extern crate dual_num;

use dual_num::{Dual, Float, differentiate};

fn main() {
    // find partial derivative at x=4.0
    println!("{:.5}", differentiate(4.0f64, |x| {
        x.sqrt() + Dual::from_real(1.0)
    })); // 0.25000
}
```

##### Previous Work
* [https://github.com/FreeFull/dual_numbers](https://github.com/FreeFull/dual_numbers)
* [https://github.com/ibab/rust-ad](https://github.com/ibab/rust-ad)
* [https://github.com/tesch1/cxxduals](https://github.com/tesch1/cxxduals)