pub struct Particle {
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
    pub best_position: Vec<f64>,
    pub best_fitness: f64,
}

impl Particle {
    pub fn new(position: &Vec<f64>) -> Self {
        Particle {
            position: position.clone(),
            velocity: vec![0.0; position.len()],
            best_position: position.clone(),
            best_fitness: f64::MAX,
        }
    }
}
