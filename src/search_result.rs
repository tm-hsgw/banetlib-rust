pub struct SearchResult {
    pub value: Vec<f64>,
    pub fitness: f64,
    pub iter: i32,
}

impl SearchResult {
    pub fn new(value: &Vec<f64>, fitness: f64, iter: i32) -> Self {
        SearchResult {
            value: value.clone(),
            fitness,
            iter,
        }
    }
}
