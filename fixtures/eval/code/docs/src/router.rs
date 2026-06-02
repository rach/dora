use crate::auth::parse_bearer_token;
use crate::rate_limit::RateLimiter;
use crate::response::{json_error, json_ok};

pub fn processRequest(path: &str, authorization: &str, limiter: &mut RateLimiter) -> String {
    if !limiter.allow(path) {
        return json_error(429, "too many requests");
    }
    match parse_bearer_token(authorization) {
        Some(token) => dispatch(path, &token),
        None => json_error(401, "missing bearer token"),
    }
}

fn dispatch(path: &str, token: &str) -> String {
    match path {
        "/health" => json_ok("healthy"),
        "/account" => json_ok(token),
        _ => json_error(404, "not found"),
    }
}

