pub fn json_ok(body: &str) -> String {
    format!(r#"{{"ok":true,"body":"{}"}}"#, body)
}

pub fn json_error(status: u16, message: &str) -> String {
    format!(r#"{{"ok":false,"status":{},"error":"{}"}}"#, status, message)
}

