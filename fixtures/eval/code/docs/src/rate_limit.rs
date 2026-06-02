use std::collections::HashMap;

pub struct RateLimiter {
    counts: HashMap<String, usize>,
    max_per_path: usize,
}

impl RateLimiter {
    pub fn new(max_per_path: usize) -> Self {
        Self {
            counts: HashMap::new(),
            max_per_path,
        }
    }

    pub fn allow(&mut self, path: &str) -> bool {
        let count = self.counts.entry(path.to_string()).or_insert(0);
        if *count >= self.max_per_path {
            return false;
        }
        *count += 1;
        true
    }
}

