mod bifurcation_point;
mod dynamical_system;
mod particle;
mod search_result;
extern crate nalgebra as na;
use crate::bifurcation_point::bifurcation_point;
use crate::dynamical_system::DiscreteTimeSystem;
use std::time::{Duration, Instant};

fn henon_f1(x: &[f64], l: &[f64]) -> f64 {
    1.0 - l[0] * x[0] * x[0] + x[1]
}

fn henon_f2(x: &[f64], l: &[f64]) -> f64 {
    l[1] * x[0]
}

fn main() {
    // 系の定義
    let maps: Vec<fn(&[f64], &[f64]) -> f64> = vec![henon_f1, henon_f2];

    // 探索空間
    let param_lower_bound: Vec<f64> = vec![1.4, 0.08];
    let param_upper_bound: Vec<f64> = vec![1.6, 0.28];
    let state_lower_bound: Vec<f64> = vec![-2., -2.];
    let state_upper_bound: Vec<f64> = vec![2., 2.];

    // 周期数・特性乗数
    let period = 5;
    let mu = -1;

    let start_time_1: Instant = Instant::now();

    for _ in 0..100 {
        let henon_map = DiscreteTimeSystem::new(
            &maps,
            &param_lower_bound,
            &param_upper_bound,
            &state_lower_bound,
            &state_upper_bound,
        );

        let (bif, pp) = bifurcation_point(henon_map, period, mu);

        println!(
            "{},{},{:.6e},{},{},{:.6e},{:?}",
            bif.iter,
            bif.value
                .iter()
                .map(|&x| format!("{:.6}", x))
                .collect::<Vec<String>>()
                .join(","),
            bif.fitness,
            pp.iter,
            pp.value
                .iter()
                .map(|&x| format!("{:.6}", x))
                .collect::<Vec<String>>()
                .join(","),
            pp.fitness,
            start_time_1.elapsed()
        );
    }

    let elapsed_time_1: Duration = start_time_1.elapsed();
    eprintln!("INFO: done - elapsed time: {:?}", elapsed_time_1);
}
