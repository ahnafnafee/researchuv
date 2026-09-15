//! The editor server — a dependency-free HTTP/1.1 endpoint that serves the
//! atlas editor page and a JSON gateway over [`HostLink`].
//!
//! Requests: `GET /` (the page), `GET /health`, `POST /api` with
//! `{"entry": "<name>", "params": {...}}`. The reply is the link response
//! envelope as JSON: `{"Ok": bool, "Error": str, "Payload": {...}}`.

use crate::json::{from_json, to_json};
use researchuv_api::ApiHandler;
use researchuv_core::val::{CRef, Val};
use researchuv_link::HostLink;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

/// A bound editor server. Call [`Server::run`] to accept connections
/// (blocks; run it on its own thread).
pub struct Server {
    listener: TcpListener,
    host: std::sync::Mutex<HostLink>,
}

impl Server {
    /// Bind on `addr` (use port 0 for an ephemeral port).
    pub fn bind(addr: &str, host: HostLink) -> std::io::Result<Self> {
        Ok(Self {
            listener: TcpListener::bind(addr)?,
            host: std::sync::Mutex::new(host),
        })
    }

    /// The port actually bound.
    pub fn port(&self) -> u16 {
        self.listener.local_addr().map(|a| a.port()).unwrap_or(0)
    }

    /// Accept loop; connections are handled one at a time (the editor makes
    /// one request at a time, and the engine state is single-threaded).
    pub fn run(self) -> std::io::Result<()> {
        for stream in self.listener.incoming() {
            let Ok(stream) = stream else { continue };
            let _ = handle(stream, &self.host);
        }
        Ok(())
    }
}

fn handle(mut stream: TcpStream, host: &std::sync::Mutex<HostLink>) -> std::io::Result<()> {
    let req = read_request(&mut stream)?;
    let (status, ctype, body) = route(&req, host);
    let text = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(text.as_bytes())?;
    stream.write_all(&body)?;
    stream.flush()
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> std::io::Result<Request> {
    let mut buf: Vec<u8> = Vec::with_capacity(1024);
    let mut chunk = [0u8; 4096];
    // Read until the header terminator, then Content-Length more.
    let header_end = loop {
        if let Some(pos) = find(&buf, b"\r\n\r\n") {
            break pos + 4;
        }
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "connection closed before headers finished",
            ));
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or_default().to_string();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
    let mut content_length = 0usize;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }
    while buf.len() < header_end + content_length {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body = buf[header_end..(header_end + content_length).min(buf.len())].to_vec();
    Ok(Request { method, path, body })
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}

fn route(req: &Request, host: &std::sync::Mutex<HostLink>) -> (&'static str, &'static str, Vec<u8>) {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") => ("200 OK", "text/html; charset=utf-8", crate::page::INDEX.as_bytes().to_vec()),
        ("GET", "/health") => ("200 OK", "text/plain; charset=utf-8", b"ok".to_vec()),
        ("POST", "/api") => {
            let text = String::from_utf8_lossy(&req.body).to_string();
            let mut host = host.lock().expect("host lock");
            match api_call(&mut host, &text) {
                Ok(v) => ("200 OK", "application/json", to_json(&v).into_bytes()),
                Err(e) => (
                    "400 Bad Request",
                    "application/json",
                    to_json(&Val::Str(e)).into_bytes(),
                ),
            }
        }
        _ => (
            "404 Not Found",
            "text/plain; charset=utf-8",
            b"not found".to_vec(),
        ),
    }
}

/// Run one `{"entry": ..., "params": {...}}` call and wrap it in the
/// response envelope.
fn api_call(host: &mut HostLink, body: &str) -> Result<Val, String> {
    let v = from_json(body)?;
    let Val::Object(req) = v else {
        return Err("request body must be a JSON object".into());
    };
    let entry = match req.get("entry") {
        Some(Val::Str(s)) => s.clone(),
        _ => return Err("request needs an \"entry\" string".into()),
    };
    let params = match req.get("params") {
        Some(Val::Object(o)) => o.clone(),
        Some(Val::Null) | None => CRef::new("Params"),
        _ => return Err("\"params\" must be an object".into()),
    };
    let mut envelope = CRef::new("Json");
    match host.invoke(&entry, &params) {
        Ok(payload) => {
            envelope.values.push(("Ok".into(), Val::Bool(true)));
            envelope.values.push(("Error".into(), Val::Str(String::new())));
            envelope.values.push(("Payload".into(), Val::Object(payload)));
        }
        Err(e) => {
            envelope.values.push(("Ok".into(), Val::Bool(false)));
            envelope.values.push(("Error".into(), Val::Str(e.to_string())));
            envelope.values.push(("Payload".into(), Val::Object(CRef::new("Json"))));
        }
    }
    Ok(Val::Object(envelope))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read one full HTTP response (Connection: close ends the stream).
    fn read_response(stream: &mut TcpStream) -> String {
        let mut out = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            let n = match stream.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            out.extend_from_slice(&chunk[..n]);
        }
        String::from_utf8_lossy(&out).to_string()
    }

    fn get(port: u16, path: &str) -> String {
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        write!(
            s,
            "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        .unwrap();
        read_response(&mut s)
    }

    fn post(port: u16, body: &str) -> String {
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        write!(
            s,
            "POST /api HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )
        .unwrap();
        read_response(&mut s)
    }

    fn spawn_server() -> u16 {
        // The server (and its engine state) lives entirely on its own
        // thread; only the port crosses back.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let server = Server::bind("127.0.0.1:0", HostLink::new()).unwrap();
            tx.send(server.port()).unwrap();
            let _ = server.run();
        });
        rx.recv().unwrap()
    }

    #[test]
    fn serves_the_page_and_health() {
        let port = spawn_server();
        let page = get(port, "/");
        assert!(page.starts_with("HTTP/1.1 200 OK"));
        assert!(page.contains("<canvas"), "the page embeds the atlas canvas");
        assert!(page.contains("</html>"));
        assert!(get(port, "/health").ends_with("ok"));
        assert!(get(port, "/nope").starts_with("HTTP/1.1 404"));
    }

    #[test]
    fn api_gateway_round_trips_an_unwrap() {
        let port = spawn_server();
        let info = post(port, r#"{"entry": "Api.Info"}"#);
        assert!(info.contains("\"Ok\":true"), "{info}");
        assert!(info.contains("Version"));

        let fix = post(port, r#"{"entry": "Doc.Fixture", "params": {"Fixture": "cube", "N": 4}}"#);
        assert!(fix.contains("\"Ok\":true"), "{fix}");
        let run = post(
            port,
            r#"{"entry": "Unwrap.Run", "params": {"Packer": "islands"}}"#,
        );
        assert!(run.contains("\"Ok\":true"), "{run}");
        assert!(run.contains("\"Charts\":6"));
        let atlas = post(port, r#"{"entry": "Atlas.Get"}"#);
        assert!(atlas.contains("\"Ok\":true"), "{atlas}");
        assert!(atlas.contains("\"Uv\""), "the atlas carries island UVs");
        // Unknown entries come back as a failed envelope, not a transport error.
        let bad = post(port, r#"{"entry": "Nope.Nothing"}"#);
        assert!(bad.contains("\"Ok\":false"), "{bad}");
        assert!(bad.contains("unknown api entry"));
        // Malformed JSON is a 400.
        assert!(post(port, "{not json").starts_with("HTTP/1.1 400"));
    }
}
