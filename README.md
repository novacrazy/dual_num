dual_num
========

This is a dual number implementation scavenged from other dual number libraries and articles around the web, including:

* [https://github.com/FreeFull/dual_numbers](https://github.com/FreeFull/dual_numbers)
* [https://github.com/ibab/rust-ad](https://github.com/ibab/rust-ad)
* [https://github.com/tesch1/cxxduals](https://github.com/tesch1/cxxduals)

The difference being is that I have checked each method against Wolfram Alpha for correctness and will
keep this implementation up to date and working with the latest stable Rust and `num-traits` crate.

## Usage

```rust
extern crate dual_num;

use dual_num::{DualNumber, Float, differentiate};

fn main() {
    // find partial derivative at x=4.0
    println!("{:.5}", differentiate(4.0f64, |x| {
        x.sqrt() + Dual::from_real(1.0)
    })); // 0.25000
}
```