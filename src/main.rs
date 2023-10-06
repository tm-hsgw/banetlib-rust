use particle::Particle;
use rand::{rngs::ThreadRng, Rng};
use search_result::SearchResult;
use std::{
    sync::{Arc, Mutex, MutexGuard},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

mod particle;
mod search_result;

extern crate nalgebra as na;

const PERIOD: i32 = 5;
const MU: i32 = 1;
const CB: f64 = 1e-3;
const CP: f64 = 1e-10;
const DB: usize = 2;
const DP: usize = 2;
const MB: usize = 30;
const MP: usize = 20;
const TB: i32 = 600;
const TP: i32 = 300;
// henon map
const PARAM_LOWER_BOUND: [f64; 2] = [1.4, 0.08];
const PARAM_UPPER_BOUND: [f64; 2] = [1.6, 0.28];
const STATE_LOWER_BOUND: [f64; 2] = [-2., -2.];
const STATE_UPPER_BOUND: [f64; 2] = [2., 2.];

// chens eq.
// const PARAM_LOWER_BOUND: [f64; 2] = [42., 2.8];
// const PARAM_UPPER_BOUND: [f64; 2] = [47., 3.3];
// const STATE_LOWER_BOUND: [f64; 3] = [-20., -0.00001, -20.];
// const STATE_UPPER_BOUND: [f64; 3] = [20., 0.00001, 20.];
// const POINCARE_INDEX: usize = 1;
// const POINCARE_VALUE: f64 = 0.;

fn main() {
    let start_time_1: Instant = Instant::now();

    for _ in 0..100 {
        let (bif, pp) = bifurcation_point();

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
    println!("First Run - Elapsed Time: {:?}", elapsed_time_1);
}

fn next(x: &mut Vec<f64>, l: &Vec<f64>) -> bool {
    // henon map
    let px = x[0];
    x[0] = 1.0 - l[0] * x[0] * x[0] + x[1];
    x[1] = l[1] * px;
    true
    // chens
    // let maps: Vec<Box<dyn Fn(&Vec<f64>, &Vec<f64>) -> f64>> = vec![
    //     Box::new(|x, l| f0(x, l)),
    //     Box::new(|x, l| f1(x, l)),
    //     Box::new(|x, l| f2(x, l)),
    // ];

    // let mut nx: Vec<f64> = x.clone();
    // let mut h: f64 = 1e-3;
    // rk(&mut nx, &l, h, &maps);
    // let sign: bool = nx[POINCARE_INDEX] > POINCARE_VALUE;

    // let mut px: Vec<f64>;

    // while h > 1e-9 {
    //     for _ in 0..2000 {
    //         px = nx.clone();
    //         rk(&mut nx, &l, h, &maps);

    //         if (nx[POINCARE_INDEX] - POINCARE_VALUE).abs() < 1e-6 {
    //             for i in 0..DP {
    //                 x[i] = nx[i];
    //             }

    //             return true;
    //         }

    //         if (nx[POINCARE_INDEX] > POINCARE_VALUE) != sign {
    //             nx = px.clone();
    //             break;
    //         }
    //     }

    //     h /= 2.;
    // }

    // false
}

// chens
// fn f0(x: &Vec<f64>, l: &Vec<f64>) -> f64 {
//     l[0] * (x[1] - x[0])
// }

// fn f1(x: &Vec<f64>, l: &Vec<f64>) -> f64 {
//     (28. - l[0]) * x[0] - x[0] * x[2] + 28. * x[1]
// }

// fn f2(x: &Vec<f64>, l: &Vec<f64>) -> f64 {
//     x[0] * x[1] - l[1] * x[2]
// }

// continuous
// fn rk(
//     state: &mut Vec<f64>,
//     param: &Vec<f64>,
//     h: f64,
//     f: &Vec<Box<dyn Fn(&Vec<f64>, &Vec<f64>) -> f64>>,
// ) {
//     let mut z: Vec<f64> = vec![0.0; DP];
//     let mut k1: Vec<f64> = vec![0.0; DP];
//     for i in 0..DP {
//         let x: f64 = f[i](&state, &param);
//         k1[i] = x;
//     }

//     for i in 0..DP {
//         z[i] = state[i] + h / 2.0 * k1[i];
//     }

//     let mut k2: Vec<f64> = vec![0.0; DP];
//     for i in 0..DP {
//         let x: f64 = f[i](&z, &param);
//         k2[i] = x;
//     }

//     for i in 0..DP {
//         z[i] = state[i] + h / 2.0 * k2[i];
//     }

//     let mut k3: Vec<f64> = vec![0.0; DP];
//     for i in 0..DP {
//         let x: f64 = f[i](&z, &param);
//         k3[i] = x;
//     }

//     for i in 0..DP {
//         z[i] = state[i] + h * k3[i];
//     }

//     let mut k4: Vec<f64> = vec![0.0; DP];
//     for i in 0..DP {
//         let x: f64 = f[i](&z, &param);
//         k4[i] = x;
//         state[i] = state[i] + h / 6. * (k1[i] + 2. * k2[i] + 2. * k3[i] + k4[i]);
//     }
// }

fn jacobian_determinant(x: &Vec<f64>, l: &Vec<f64>) -> f64 {
    // numerical diff
    let h: f64 = 1e-4;

    let mut m: Vec<na::Matrix<f64, na::Dyn, na::Dyn, na::VecStorage<_, na::Dyn, na::Dyn>>> =
        (0..PERIOD).map(|_| na::DMatrix::zeros(DP, DP)).collect();
    let mut xk: Vec<f64> = x.clone();

    for k in 0..PERIOD as usize {
        let prevx: Vec<f64> = xk.clone();

        let res: bool = next(&mut xk, &l);
        if !res {
            return 1e+20;
        }

        for i in 0..DP {
            let mut qx: Vec<f64> = prevx.clone();
            qx[i] += h;

            let res: bool = next(&mut qx, &l);
            if !res {
                return 1e+15;
            }

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
    let global_best: Arc<Mutex<SearchResult>> = Arc::new(Mutex::new(SearchResult::new(
        &PARAM_LOWER_BOUND.to_vec(),
        f64::MAX,
        -1,
    )));

    let global_best_child: Arc<Mutex<SearchResult>> =
        Arc::new(Mutex::new(SearchResult::new(&vec![0.0], 1.0 / CP, -1)));

    let mut threads: Vec<JoinHandle<()>> = Vec::new();

    // main loop
    for _ in 0..MB {
        let global_best: Arc<Mutex<SearchResult>> = Arc::clone(&global_best);
        let global_best_child: Arc<Mutex<SearchResult>> = Arc::clone(&global_best_child);

        threads.push(thread::spawn(move || {
            // init
            let mut rng: ThreadRng = rand::thread_rng();
            let initial_position: Vec<f64> = (0..DB)
                .map(|i| rng.gen_range(PARAM_LOWER_BOUND[i]..PARAM_UPPER_BOUND[i]))
                .collect();
            let mut particle: Particle = Particle::new(&initial_position);
            let mut global_best_clone: SearchResult = SearchResult {
                value: vec![0.0; DB],
                fitness: f64::MAX,
                iter: -1,
            };

            for t in 0..TB {
                // search periodic point
                let child_result = periodic_point(&particle.position, &mut rng);
                let mut new_error: f64 = child_result.fitness / CP / CP;
                if child_result.fitness < CP {
                    new_error = jacobian_determinant(&child_result.value, &particle.position);
                }

                // update bests
                if new_error < particle.best_fitness {
                    particle.best_position = particle.position.clone();
                    particle.best_fitness = new_error;
                }

                let global_best_: MutexGuard<'_, SearchResult> = global_best.lock().unwrap();
                global_best_clone.value = global_best_.value.clone();
                global_best_clone.fitness = global_best_.fitness;
                drop(global_best_);
                if new_error < global_best_clone.fitness {
                    let mut global_best: MutexGuard<'_, SearchResult> = global_best.lock().unwrap();
                    global_best.value = particle.best_position.clone();
                    global_best.fitness = new_error;
                    global_best.iter = t;
                    // drop(global_best);

                    let mut global_best_child: MutexGuard<'_, SearchResult> =
                        global_best_child.lock().unwrap();
                    global_best_child.value = child_result.value.clone();
                    global_best_child.fitness = child_result.fitness;
                    global_best_child.iter = child_result.iter;
                    // drop(global_best_child);

                    // debug
                    // eprintln!("{}, {:.6e}", t, new_error);
                }

                if global_best_clone.fitness < CB {
                    // let mut global_best: MutexGuard<'_, SearchResult> = global_best.lock().unwrap();
                    // global_best.iter = t;
                    break;
                }

                // update position
                for i in 0..DB {
                    let inertia: f64 = 0.7 * particle.velocity[i];
                    let r1: f64 = rng.gen_range(0.0..1.49445);
                    let r2: f64 = rng.gen_range(0.0..1.49445);
                    let cognitive: f64 = r1 * (particle.best_position[i] - particle.position[i]);
                    let social: f64 = r2 * (global_best_clone.value[i] - particle.position[i]);
                    particle.velocity[i] = inertia + cognitive + social;
                    let mut nx: f64 = particle.position[i] + particle.velocity[i];
                    if nx < PARAM_LOWER_BOUND[i] || nx > PARAM_UPPER_BOUND[i] {
                        particle.velocity[i] /= 10.;
                        nx = rng.gen_range(PARAM_LOWER_BOUND[i]..PARAM_UPPER_BOUND[i]);
                    }
                    particle.position[i] = nx;
                }
            }
        }));
    }

    for thread in threads {
        thread.join().unwrap();
    }

    let global_best: MutexGuard<'_, SearchResult> = global_best.lock().unwrap();
    let global_best_child: MutexGuard<'_, SearchResult> = global_best_child.lock().unwrap();

    (
        SearchResult {
            value: global_best.value.clone(),
            fitness: global_best.fitness,
            iter: global_best.iter,
        },
        SearchResult {
            value: global_best_child.value.clone(),
            fitness: global_best_child.fitness,
            iter: global_best_child.iter,
        },
    )
}

fn bifurcation_point_sel() -> (SearchResult, SearchResult) {
    let mut rng: ThreadRng = rand::thread_rng();
    let mut swarm: Vec<Particle> = (0..MB)
        .map(|_| {
            let initial_position: Vec<f64> = (0..DB)
                .map(|i| rng.gen_range(PARAM_LOWER_BOUND[i]..PARAM_UPPER_BOUND[i]))
                .collect();
            Particle::new(&initial_position)
        })
        .collect();
    let mut global_best: SearchResult =
        SearchResult::new(&PARAM_LOWER_BOUND.to_vec(), f64::MAX, -1);
    let mut global_best_child: SearchResult = SearchResult::new(&vec![0.0], f64::MAX, -1);

    for t in 0..TB {
        for particle in &mut swarm {
            // search periodic point
            let child_result = periodic_point(&particle.position, &mut rng);
            let mut new_error: f64 = child_result.fitness / CP / CP;
            if child_result.fitness < CP {
                new_error = jacobian_determinant(&child_result.value, &particle.position);
            }

            if new_error < particle.best_fitness {
                particle.best_position = particle.position.clone();
                particle.best_fitness = new_error;
            }

            if particle.best_fitness < global_best.fitness {
                global_best.value = particle.best_position.clone();
                global_best.fitness = new_error;
                global_best.iter = t;

                global_best_child.value = child_result.value.clone();
                global_best_child.fitness = child_result.fitness;
                global_best_child.iter = child_result.iter;

                // debug
                // eprintln!("{}, {:.6e}", t, new_error);
            }
        }

        if global_best.fitness < CB {
            // Early termination if the threshold is reached
            break;
        }

        for particle in &mut swarm {
            // update position
            for i in 0..DB {
                let inertia: f64 = 0.7 * particle.velocity[i];
                let r1: f64 = rng.gen_range(0.0..1.49445);
                let r2: f64 = rng.gen_range(0.0..1.49445);
                let cognitive: f64 = r1 * (particle.best_position[i] - particle.position[i]);
                let social: f64 = r2 * (global_best.value[i] - particle.position[i]);
                particle.velocity[i] = inertia + cognitive + social;
                let mut nx: f64 = particle.position[i] + particle.velocity[i];
                if nx < PARAM_LOWER_BOUND[i] || nx > PARAM_UPPER_BOUND[i] {
                    particle.velocity[i] /= 10.;
                    nx = rng.gen_range(PARAM_LOWER_BOUND[i]..PARAM_UPPER_BOUND[i]);
                }
                particle.position[i] = nx;
            }
        }
    }

    (
        SearchResult {
            value: global_best.value.clone(),
            fitness: global_best.fitness,
            iter: global_best.iter,
        },
        SearchResult {
            value: global_best_child.value.clone(),
            fitness: global_best_child.fitness,
            iter: global_best_child.iter,
        },
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
                let mut nx: f64 = particle.position[i] + particle.velocity[i];
                if nx < STATE_LOWER_BOUND[i] || nx > STATE_UPPER_BOUND[i] {
                    particle.velocity[i] /= 10.;
                    nx = rng.gen_range(STATE_LOWER_BOUND[i]..STATE_UPPER_BOUND[i]);
                }
                particle.position[i] = nx;
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
        let res = next(&mut nz, &l);
        if !res {
            return 1e+10;
        }
    }

    let mut dx: f64 = 0.0;
    for i in 0..z.len() {
        let diff = nz[i] - z[i];
        dx += diff * diff;
    }

    dx
}
