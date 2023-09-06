use particle::Particle;
use rand::rngs::ThreadRng;
use rand::Rng;
use search_result::SearchResult;
use std::time::{Duration, Instant};

mod particle;
mod search_result;

extern crate nalgebra as na;

const PERIOD: i32 = 5;
const MU: i32 = -1;
const CB: f64 = 1e-3;
const CP: f64 = 1e-10;
const DB: usize = 2;
const DP: usize = 2;
const MB: usize = 30;
const MP: usize = 20;
const TB: i32 = 600;
const TP: i32 = 300;
const PARAM_LOWER_BOUND: [f64; 2] = [1.4, 0.08];
const PARAM_UPPER_BOUND: [f64; 2] = [1.6, 0.28];
const STATE_LOWER_BOUND: [f64; 2] = [-2., -2.];
const STATE_UPPER_BOUND: [f64; 2] = [2., 2.];

fn main() {
    let start_time_1: Instant = Instant::now();

    for _ in 0..100 {
        let (bif, pp) = bifurcation_point();

        println!(
            "{},{},{:.6e},{},{},{:.6e}",
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
        );
    }

    let elapsed_time_1: Duration = start_time_1.elapsed();
    println!("First Run - Elapsed Time: {:?}", elapsed_time_1);
}

fn next(x: &mut Vec<f64>, l: &Vec<f64>) {
    let px = x[0];
    x[0] = 1.0 - l[0] * x[0] * x[0] + x[1];
    x[1] = l[1] * px;
}

fn jacobian_determinant(x: &Vec<f64>, l: &Vec<f64>) -> f64 {
    // numerical diff
    let h: f64 = 1e-4;

    // let mut m: na::Matrix<f64, na::Dyn, na::Dyn, na::VecStorage<_, na::Dyn, na::Dyn>> =
    //     DMatrix::zeros(n, n);
    let mut m: Vec<na::Matrix<f64, na::Dyn, na::Dyn, na::VecStorage<_, na::Dyn, na::Dyn>>> =
        (0..PERIOD).map(|_| na::DMatrix::zeros(DP, DP)).collect();
    let mut xk: Vec<f64> = x.clone();

    for k in 0..PERIOD as usize {
        let prevx: Vec<f64> = xk.clone();

        // TODO: try
        next(&mut xk, &l);

        for i in 0..DP {
            let mut qx: Vec<f64> = prevx.clone();
            qx[i] += h;

            // TODO: try
            next(&mut qx, &l);

            for j in 0..DP {
                m[k][(j, i)] = (qx[j] - xk[j]) / h;
            }
        }
    }

    let mut jacobian = m[(PERIOD - 1) as usize].clone();
    for k in 1..PERIOD as usize {
        jacobian = jacobian * m[PERIOD as usize - 1 - k].clone()
    }

    for i in 0..DP {
        jacobian[(i, i)] -= MU as f64;
    }

    jacobian.determinant().abs()
}

fn bifurcation_point() -> (SearchResult, SearchResult) {
    // init
    let mut rng: ThreadRng = rand::thread_rng();
    let mut swarm: Vec<Particle> = (0..MB)
        .map(|_| {
            let initial_position: Vec<f64> = (0..DB)
                .map(|i| rng.gen_range(PARAM_LOWER_BOUND[i]..PARAM_UPPER_BOUND[i]))
                .collect();
            Particle::new(&initial_position)
        })
        .collect();
    let mut global_best: Vec<f64> = vec![0.0; DB];
    let mut global_best_fitness: f64 = f64::MAX;
    let mut global_best_child: SearchResult = SearchResult::new(&vec![0.0], 1.0 / CP, -1);

    let mut iter: i32 = -1;
    // main loop
    for t in 0..TB {
        for particle in &mut swarm {
            let result = periodic_point(&particle.position, &mut rng);
            let mut new_error: f64 = result.fitness / CP / CP;
            if result.fitness < CP {
                new_error = jacobian_determinant(&result.value, &particle.position);
            }

            if new_error < particle.best_fitness {
                particle.best_position = particle.position.clone();
                particle.best_fitness = new_error;
            }

            if particle.best_fitness < global_best_fitness {
                // debug
                // println!("\r{}:{:.6e}     ", t, global_best_fitness);
                global_best = particle.best_position.clone();
                global_best_fitness = new_error;
                global_best_child = result;
            }
        }

        if global_best_fitness < CB {
            // Early termination if the threshold is reached
            iter = t;
            break;
        }

        for particle in &mut swarm {
            // move
            for i in 0..DB {
                let inertia: f64 = 0.7 * particle.velocity[i];
                let r1: f64 = rng.gen_range(0.0..1.49445);
                let r2: f64 = rng.gen_range(0.0..1.49445);
                let cognitive: f64 = r1 * (particle.best_position[i] - particle.position[i]);
                let social: f64 = r2 * (global_best[i] - particle.position[i]);
                particle.velocity[i] = inertia + cognitive + social;
                particle.position[i] = (particle.position[i] + particle.velocity[i])
                    .max(PARAM_LOWER_BOUND[i])
                    .min(PARAM_UPPER_BOUND[i]);
            }
        }
    }

    (
        SearchResult {
            value: global_best,
            fitness: global_best_fitness,
            iter,
        },
        global_best_child,
    )
}

fn periodic_point(param: &Vec<f64>, rng: &mut ThreadRng) -> SearchResult {
    // init
    let mut swarm: Vec<Particle> = (0..MP)
        .map(|_| {
            let initial_position: Vec<f64> = (0..DP)
                .map(|i| rng.gen_range(STATE_LOWER_BOUND[i]..STATE_UPPER_BOUND[i]))
                .collect();
            Particle::new(&initial_position)
        })
        .collect();
    let mut global_best: Vec<f64> = vec![0.0; DP];
    let mut global_best_fitness: f64 = f64::MAX;

    let mut iter: i32 = -1;
    // main loop
    for t in 0..TP {
        for particle in &mut swarm {
            let new_error: f64 = periodic_error(&particle.position, &param);

            if new_error < particle.best_fitness {
                particle.best_position = particle.position.clone();
                particle.best_fitness = new_error;
            }

            if particle.best_fitness < global_best_fitness {
                global_best = particle.best_position.clone();
                global_best_fitness = new_error;
            }
        }

        if global_best_fitness < CP {
            // Early termination if the threshold is reached
            iter = t;
            break;
        }

        for particle in &mut swarm {
            // move
            for i in 0..DP {
                let inertia: f64 = 0.7 * particle.velocity[i];
                let r1: f64 = rng.gen_range(0.0..1.49445);
                let r2: f64 = rng.gen_range(0.0..1.49445);
                let cognitive: f64 = r1 * (particle.best_position[i] - particle.position[i]);
                let social: f64 = r2 * (global_best[i] - particle.position[i]);
                particle.velocity[i] = inertia + cognitive + social;
                particle.position[i] = (particle.position[i] + particle.velocity[i])
                    .max(STATE_LOWER_BOUND[i])
                    .min(STATE_UPPER_BOUND[i]);
            }
        }
    }

    SearchResult {
        value: global_best,
        fitness: global_best_fitness,
        iter,
    }
}

fn periodic_error(z: &Vec<f64>, l: &Vec<f64>) -> f64 {
    let mut nz: Vec<f64> = z.clone();
    for _ in 0..PERIOD {
        next(&mut nz, &l)
    }

    let mut dx: f64 = 0.0;
    for i in 0..z.len() {
        let diff = nz[i] - z[i];
        dx += diff * diff;
    }

    dx
}
