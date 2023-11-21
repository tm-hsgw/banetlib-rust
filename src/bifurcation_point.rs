extern crate rand;
use crate::particle::Particle;
use crate::search_result::SearchResult;
use dynamical_system::DynamicalSystem;
use rand::rngs::ThreadRng;
use rand::Rng;
use std::{
    sync::{Arc, Mutex, MutexGuard},
    thread::{self, JoinHandle},
};

use crate::dynamical_system::{self};

extern crate nalgebra as na;

const CB: f64 = 1e-3;
const CP: f64 = 1e-10;
const DB: usize = 2;
const DP: usize = 2;
const MB: usize = 30;
const MP: usize = 20;
const TB: i32 = 600;
const TP: i32 = 300;

fn jacobian_determinant<T: DynamicalSystem>(
    x: &Vec<f64>,
    l: &Vec<f64>,
    sys: &T,
    period: u32,
    mu: i32,
) -> f64 {
    let d = x.len();

    // numerical diff
    let h: f64 = 1e-4;

    let mut m: Vec<na::Matrix<f64, na::Dyn, na::Dyn, na::VecStorage<_, na::Dyn, na::Dyn>>> =
        (0..period).map(|_| na::DMatrix::zeros(d, d)).collect();
    let mut xk: Vec<f64> = x.clone();

    for k in 0..period as usize {
        let prevx: Vec<f64> = xk.clone();

        let res: bool = sys.next(&mut xk, &l);
        if !res {
            return 1e+20;
        }

        for i in 0..d {
            let mut qx: Vec<f64> = prevx.clone();
            qx[i] += h;

            let res: bool = sys.next(&mut qx, &l);
            if !res {
                return 1e+15;
            }

            for j in 0..d {
                m[k][(j, i)] = (qx[j] - xk[j]) / h;
            }
        }
    }

    let mut jacobian = m[(period - 1) as usize].clone();
    for k in 1..period as usize {
        jacobian = jacobian * m[period as usize - 1 - k].clone()
    }

    for i in 0..d {
        jacobian[(i, i)] -= mu as f64;
    }

    jacobian.determinant().abs()
}

pub fn bifurcation_point<T: DynamicalSystem + Clone + Send + 'static>(
    sys: T,
    period: u32,
    mu: i32,
) -> (SearchResult, SearchResult) {
    let global_best: Arc<Mutex<SearchResult>> =
        Arc::new(Mutex::new(SearchResult::new(&vec![0.0; DP], f64::MAX, -1)));

    let global_best_child: Arc<Mutex<SearchResult>> =
        Arc::new(Mutex::new(SearchResult::new(&vec![0.0], 1.0 / CP, -1)));

    let mut threads: Vec<JoinHandle<()>> = Vec::new();

    // main loop
    for _ in 0..MB {
        let sys_clone = sys.clone();
        let global_best: Arc<Mutex<SearchResult>> = Arc::clone(&global_best);
        let global_best_child: Arc<Mutex<SearchResult>> = Arc::clone(&global_best_child);

        threads.push(thread::spawn(move || {
            // init
            let mut rng: ThreadRng = rand::thread_rng();
            // xorshift 置き換えただけだと速くならなかった
            // let mut rng = rand_xorshift::XorShiftRng::seed_from_u64(
            //     SystemTime::now()
            //         .duration_since(UNIX_EPOCH)
            //         .expect("Time went backwards")
            //         .as_nanos()
            //         .try_into()
            //         .unwrap(),
            // );

            let initial_position: Vec<f64> = (0..DB)
                .map(|i| {
                    rng.gen_range(sys_clone.param_lower_bound(i)..sys_clone.param_upper_bound(i))
                })
                .collect();
            let mut particle: Particle = Particle::new(&initial_position);
            let mut global_best_clone: SearchResult = SearchResult {
                value: vec![0.0; DB],
                fitness: f64::MAX,
                iter: -1,
            };

            for t in 0..TB {
                // search periodic point
                let child_result = periodic_point(&particle.position, &mut rng, &sys_clone, period);
                let mut new_error: f64 = child_result.fitness / CP / CP;
                if child_result.fitness < CP {
                    new_error = jacobian_determinant(
                        &child_result.value,
                        &particle.position,
                        &sys_clone,
                        period,
                        mu,
                    );
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
                    if nx < sys_clone.param_lower_bound(i) || nx > sys_clone.param_upper_bound(i) {
                        particle.velocity[i] /= 10.;
                        nx = rng.gen_range(
                            sys_clone.param_lower_bound(i)..sys_clone.param_upper_bound(i),
                        );
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

fn periodic_point<T: DynamicalSystem>(
    param: &Vec<f64>,
    rng: &mut ThreadRng,
    sys: &T,
    period: u32,
) -> SearchResult {
    // init
    let mut swarm: Vec<Particle> = (0..MP)
        .map(|_| {
            let initial_position: Vec<f64> = (0..DP)
                .map(|i| rng.gen_range(sys.state_lower_bound(i)..sys.state_upper_bound(i)))
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
            let new_error: f64 = periodic_error(&particle.position, &param, sys, period);

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
                    .max(sys.state_lower_bound(i))
                    .min(sys.state_upper_bound(i));
            }
        }
    }

    SearchResult {
        value: global_best,
        fitness: global_best_fitness,
        iter,
    }
}

fn periodic_error<T: DynamicalSystem>(z: &Vec<f64>, l: &Vec<f64>, sys: &T, period: u32) -> f64 {
    let mut nz: Vec<f64> = z.clone();
    for _ in 0..period {
        let res = sys.next(&mut nz, &l);
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
