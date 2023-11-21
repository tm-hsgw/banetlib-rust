use std::usize;
pub trait DynamicalSystem: Send + Clone {
    fn next(&self, state: &mut Vec<f64>, param: &[f64]) -> bool;
    fn param_lower_bound(&self, i: usize) -> f64;
    fn param_upper_bound(&self, i: usize) -> f64;
    fn state_lower_bound(&self, i: usize) -> f64;
    fn state_upper_bound(&self, i: usize) -> f64;
}

#[derive(Clone)]
pub struct DiscreteTimeSystem {
    dim: usize,
    maps: Vec<fn(&[f64], &[f64]) -> f64>,
    lmax: Vec<f64>,
    lmin: Vec<f64>,
    xmax: Vec<f64>,
    xmin: Vec<f64>,
}

#[derive(Clone)]
pub struct AutonomousSystem {
    dim: usize,
    maps: Vec<fn(&[f64], &[f64]) -> f64>,
    poincare_index: usize,
    poincare_value: f64,
    lmax: Vec<f64>,
    lmin: Vec<f64>,
    xmax: Vec<f64>,
    xmin: Vec<f64>,
}

impl DiscreteTimeSystem {
    pub fn new(
        maps: &[fn(&[f64], &[f64]) -> f64],
        param_min: &[f64],
        param_max: &[f64],
        state_min: &[f64],
        state_max: &[f64],
    ) -> DiscreteTimeSystem {
        let m = maps.to_vec();
        DiscreteTimeSystem {
            dim: m.len(),
            maps: m,
            lmin: param_min.to_vec(),
            lmax: param_max.to_vec(),
            xmin: state_min.to_vec(),
            xmax: state_max.to_vec(),
        }
    }
}

impl DynamicalSystem for DiscreteTimeSystem {
    fn next(&self, state: &mut Vec<f64>, param: &[f64]) -> bool {
        let mut nx: Vec<f64> = vec![0.0; self.dim];
        for i in 0..self.dim {
            nx[i] = self.maps[i](&state, &param);
        }

        for i in 0..self.dim {
            state[i] = nx[i];
        }

        true
    }

    fn param_lower_bound(&self, i: usize) -> f64 {
        self.lmin[i]
    }

    fn param_upper_bound(&self, i: usize) -> f64 {
        self.lmax[i]
    }

    fn state_lower_bound(&self, i: usize) -> f64 {
        self.xmin[i]
    }

    fn state_upper_bound(&self, i: usize) -> f64 {
        self.xmax[i]
    }
}

impl AutonomousSystem {
    pub fn new(
        maps: &[fn(&[f64], &[f64]) -> f64],
        poincare_index: usize,
        poincare_value: f64,
        param_min: &[f64],
        param_max: &[f64],
        state_min: &[f64],
        state_max: &[f64],
    ) -> AutonomousSystem {
        let m = maps.to_vec();
        AutonomousSystem {
            dim: m.len(),
            maps: m,
            poincare_index,
            poincare_value,
            lmin: param_min.to_vec(),
            lmax: param_max.to_vec(),
            xmin: state_min.to_vec(),
            xmax: state_max.to_vec(),
        }
    }
}

impl DynamicalSystem for AutonomousSystem {
    fn next(&self, state: &mut Vec<f64>, param: &[f64]) -> bool {
        let mut nx: Vec<f64> = state.clone();
        let mut h: f64 = 1e-3;
        rk(&mut nx, &param, h, &self.maps);
        let sign: bool = nx[self.poincare_index] > self.poincare_value;

        let mut px: Vec<f64>;

        while h > 1e-9 {
            for _ in 0..2000 {
                px = nx.clone();
                rk(&mut nx, &param, h, &self.maps);

                if (nx[self.poincare_index] - self.poincare_value).abs() < 1e-6 {
                    for i in 0..self.dim {
                        state[i] = nx[i];
                    }

                    return true;
                }

                if (nx[self.poincare_index] > self.poincare_value) != sign {
                    nx = px.clone();
                    break;
                }
            }

            h /= 2.;
        }

        false
    }

    fn param_lower_bound(&self, i: usize) -> f64 {
        self.lmin[i]
    }

    fn param_upper_bound(&self, i: usize) -> f64 {
        self.lmax[i]
    }

    fn state_lower_bound(&self, i: usize) -> f64 {
        self.xmin[i]
    }

    fn state_upper_bound(&self, i: usize) -> f64 {
        self.xmax[i]
    }
}

fn rk(state: &mut Vec<f64>, param: &[f64], h: f64, f: &[fn(&[f64], &[f64]) -> f64]) {
    let d: usize = state.len();
    let mut z: Vec<f64> = vec![0.0; d];
    let mut k1: Vec<f64> = vec![0.0; d];
    for i in 0..d {
        let x: f64 = f[i](&state, &param);
        k1[i] = x;
    }

    for i in 0..d {
        z[i] = state[i] + h / 2.0 * k1[i];
    }

    let mut k2: Vec<f64> = vec![0.0; d];
    for i in 0..d {
        let x: f64 = f[i](&z, &param);
        k2[i] = x;
    }

    for i in 0..d {
        z[i] = state[i] + h / 2.0 * k2[i];
    }

    let mut k3: Vec<f64> = vec![0.0; d];
    for i in 0..d {
        let x: f64 = f[i](&z, &param);
        k3[i] = x;
    }

    for i in 0..d {
        z[i] = state[i] + h * k3[i];
    }

    let mut k4: Vec<f64> = vec![0.0; d];
    for i in 0..d {
        let x: f64 = f[i](&z, &param);
        k4[i] = x;
        state[i] = state[i] + h / 6. * (k1[i] + 2. * k2[i] + 2. * k3[i] + k4[i]);
    }
}
