//! # researchuv-editor — the browser-served atlas editor
//!
//! A dependency-free local web app over the engine: [`Server`] serves the
//! canvas editor page and a JSON gateway (`POST /api`) mapped 1:1 onto the
//! [`researchuv_api`] catalog, dispatched through [`researchuv_link`]'s
//! [`HostLink`]. The page re-runs the whole unwrap pipeline live (fixture,
//! cut angles, packer, seam-cut, distortion-driven re-cutting, thread count)
//! and renders the packed islands with per-chart distortion metrics.
//!
//! ```no_run
//! # fn main() -> std::io::Result<()> {
//! let server = researchuv_editor::Server::bind("127.0.0.1:7899", researchuv_link::HostLink::new())?;
//! println!("http://127.0.0.1:{}", server.port());
//! server.run()
//! # }
//! ```

pub mod json;
pub mod page;
pub mod server;

pub use json::{from_json, to_json};
pub use server::Server;

#[cfg(test)]
mod tests {
    #[test]
    fn page_is_a_complete_document() {
        assert!(crate::page::INDEX.starts_with("<!DOCTYPE html>"));
        assert!(crate::page::INDEX.contains("</html>"));
        assert!(crate::page::INDEX.contains("/api"));
    }
}
