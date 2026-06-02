pub fn parse_bearer_token(header: &str) -> Option<String> {
    let trimmed = header.trim();
    let token = trimmed.strip_prefix("Bearer ")?;
    if token.is_empty() {
        None
    } else {
        Some(token.to_string())
    }
}

