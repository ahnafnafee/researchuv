//! The `researchuv-editor` binary: serve the atlas editor and (with `--open`)
//! launch the default browser.

use researchuv_editor::Server;
use researchuv_link::HostLink;

const USAGE: &str = "usage: researchuv-editor [ADDR] [--open]\n  ADDR  bind address (default 127.0.0.1:7899)\n  --open  launch the default browser\n";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return;
    }
    let open = args.iter().any(|a| a == "--open");
    let addr = args
        .iter()
        .find(|a| !a.starts_with('-'))
        .cloned()
        .unwrap_or_else(|| "127.0.0.1:7899".to_string());
    let server = match Server::bind(&addr, HostLink::new()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    let url = format!("http://127.0.0.1:{}", server.port());
    println!("researchuv editor listening on {url} (Ctrl+C to stop)");
    if open {
        #[cfg(windows)]
        let spawned = std::process::Command::new("cmd")
            .args(["/C", "start", "", &url])
            .spawn()
            .map(|_| ());
        #[cfg(not(windows))]
        let spawned = std::process::Command::new("xdg-open")
            .arg(&url)
            .spawn()
            .map(|_| ());
        if let Err(e) = spawned {
            eprintln!("warning: could not launch a browser ({e}); open {url} manually");
        }
    }
    if let Err(e) = server.run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
