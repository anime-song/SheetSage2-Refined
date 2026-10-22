// erf as Eigen's float approximation (the CPU kernels' `math::erf`), for the same GELU on both.
fn erf_approx(x0: f32) -> f32 {
    let x = clamp(x0, -4.0, 4.0);
    let x2 = x * x;
    var p = x2 * -2.7261423e-10 + 2.7706814e-08;
    p = x2 * p + -2.101024e-06;
    p = x2 * p + -5.6925064e-05;
    p = x2 * p + -7.3499063e-04;
    p = x2 * p + -2.9546e-3;
    p = x2 * p + -1.6096033e-02;
    p = p * x;
    var q = x2 * -1.4566072e-05 + -2.1337406e-04;
    q = x2 * q + -1.682827e-03;
    q = x2 * q + -7.3733293e-03;
    q = x2 * q + -1.426474e-02;
    return p / q;
}

fn gelu(x: f32) -> f32 {
    return 0.5 * x * (1.0 + erf_approx(x * 0.70710678118654752));
}
