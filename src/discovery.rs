use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};

use anyhow::{anyhow, Result};
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use tokio::sync::mpsc;
use tracing::{debug, warn};

pub const SERVICE_TYPE: &str = "_hailort._tcp.local.";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WorkerEndpoint {
    pub instance_name: String,
    pub host: String,
    pub ip: Ipv4Addr,
    pub port: u16,
}

#[derive(Debug)]
pub enum DiscoveryEvent {
    Appeared(WorkerEndpoint),
    Disappeared(String), // instance_name
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Registers this process as a HailoRT worker via mDNS-SD.
///
/// **The returned `ServiceDaemon` must be kept alive** — dropping it
/// immediately de-registers the service. Bind it to a long-lived variable
/// with a leading underscore to make the intent obvious without triggering
/// the unused-variable lint, e.g.:
/// ```
/// let _mdns = discovery::register_worker("worker-1", 50051)?;
/// ```
pub fn register_worker(instance_name: &str, grpc_port: u16) -> Result<ServiceDaemon> {
    let ip = local_ip()?;
    let host = local_hostname();

    let mut props = std::collections::HashMap::new();
    props.insert("version".to_string(), "1".to_string());
    props.insert("instance".to_string(), instance_name.to_string());

    let service = ServiceInfo::new(
        SERVICE_TYPE,
        instance_name,
        &host,
        IpAddr::V4(ip),
        grpc_port,
        Some(props),
    )
    .map_err(|e| anyhow!("ServiceInfo::new failed: {}", e))?;

    let daemon = ServiceDaemon::new().map_err(|e| anyhow!("ServiceDaemon::new failed: {}", e))?;
    daemon
        .register(service)
        .map_err(|e| anyhow!("mDNS register failed: {}", e))?;

    tracing::info!(
        instance = instance_name,
        ip = %ip,
        port = grpc_port,
        "Worker registered via mDNS"
    );
    Ok(daemon)
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Browses the LAN for HailoRT workers.
///
/// Returns a channel that yields [`DiscoveryEvent`]s and the [`ServiceDaemon`]
/// that drives the browse loop. **Both must be kept alive** by the caller.
///
/// `buffer` controls the tokio mpsc channel capacity.
pub fn browse_workers(
    buffer: usize,
) -> Result<(mpsc::Receiver<DiscoveryEvent>, ServiceDaemon)> {
    let daemon =
        ServiceDaemon::new().map_err(|e| anyhow!("ServiceDaemon::new failed: {}", e))?;

    let browse_rx = daemon
        .browse(SERVICE_TYPE)
        .map_err(|e| anyhow!("mDNS browse failed: {}", e))?;

    let (tx, rx) = mpsc::channel::<DiscoveryEvent>(buffer);

    // mdns-sd's receiver may be a std or flume channel — use spawn_blocking
    // + blocking_send so the exact receiver type doesn't matter.
    tokio::task::spawn_blocking(move || {
        loop {
            match browse_rx.recv() {
                Ok(event) => {
                    let discovery = match event {
                        ServiceEvent::ServiceResolved(info) => {
                            let ip = info
                                .get_addresses()
                                .iter()
                                .find_map(|addr| {
                                    if let IpAddr::V4(v4) = addr {
                                        Some(*v4)
                                    } else {
                                        None
                                    }
                                });

                            match ip {
                                Some(ip) => {
                                    debug!(
                                        instance = info.get_fullname(),
                                        %ip,
                                        port = info.get_port(),
                                        "Worker appeared"
                                    );
                                    Some(DiscoveryEvent::Appeared(WorkerEndpoint {
                                        instance_name: info.get_fullname().to_string(),
                                        host: info.get_hostname().to_string(),
                                        ip,
                                        port: info.get_port(),
                                    }))
                                }
                                None => {
                                    warn!(
                                        instance = info.get_fullname(),
                                        "Resolved service has no IPv4 address, skipping"
                                    );
                                    None
                                }
                            }
                        }
                        ServiceEvent::ServiceRemoved(_type, fullname) => {
                            debug!(instance = %fullname, "Worker disappeared");
                            Some(DiscoveryEvent::Disappeared(fullname))
                        }
                        // SearchStarted, SearchStopped, etc.
                        _ => None,
                    };

                    if let Some(ev) = discovery {
                        if tx.blocking_send(ev).is_err() {
                            // Receiver dropped — stop the browse loop.
                            break;
                        }
                    }
                }
                Err(_) => {
                    // Channel closed on the mdns-sd side.
                    break;
                }
            }
        }
    });

    Ok((rx, daemon))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Determines the local LAN IPv4 address via a no-op UDP connect trick.
/// No packet is actually sent.
pub fn local_ip() -> Result<Ipv4Addr> {
    let socket = UdpSocket::bind("0.0.0.0:0")?;
    socket.connect(SocketAddr::from(([8, 8, 8, 8], 80)))?;
    let addr = socket.local_addr()?;
    match addr.ip() {
        IpAddr::V4(v4) => Ok(v4),
        other => Err(anyhow!("local address is not IPv4: {}", other)),
    }
}

/// Returns the machine hostname, falling back to `"localhost"`.
pub fn local_hostname() -> String {
    hostname::get()
        .ok()
        .and_then(|s| s.into_string().ok())
        .unwrap_or_else(|| "localhost".to_string())
}
