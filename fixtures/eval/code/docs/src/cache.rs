use std::collections::HashMap;

pub struct MemoryCache {
    values: HashMap<String, String>,
}

impl MemoryCache {
    pub fn get(&self, key: &str) -> Option<&String> {
        self.values.get(key)
    }
}
