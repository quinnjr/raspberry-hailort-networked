use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use dashmap::DashMap;
use tokio::time::sleep;
use tonic::transport::Channel;
use tracing::{info, warn};

use raspberry_hailort_networked::{
    discovery::{self, DiscoveryEvent, WorkerEndpoint},
    proto::{
        inference_service_client::InferenceServiceClient, GetStatusRequest, LoadModelRequest,
        RunInferenceRequest, UnloadModelRequest,
    },
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "master", about = "HailoRT distributed inference master")]
struct Cli {
    /// How long to wait for mDNS discovery before running commands (ms)
    #[arg(long, default_value_t = 3000)]
    discovery_wait_ms: u64,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Query status of all discovered workers
    Status,

    /// Load a HEF model onto worker(s)
    LoadModel {
        /// Path to the .hef file
        #[arg(long)]
        file: std::path::PathBuf,

        /// Logical name to assign to the model
        #[arg(long)]
        name: String,

        /// Target a specific worker instance (broadcasts to all if omitted)
        #[arg(long)]
        worker: Option<String>,
    },

    /// Unload a model from worker(s)
    UnloadModel {
        /// Logical model name
        #[arg(long)]
        name: String,

        /// Target a specific worker instance (broadcasts to all if omitted)
        #[arg(long)]
        worker: Option<String>,
    },

    /// Run inference on a worker
    Infer {
        /// Logical model name
        #[arg(long)]
        model: String,

        /// Path to binary input file
        #[arg(long)]
        input: std::path::PathBuf,

        /// Target a specific worker instance (picks first available if omitted)
        #[arg(long)]
        worker: Option<String>,

        /// Expected output byte count (0 = let worker decide)
        #[arg(long, default_value_t = 0)]
        expected_output_size: u32,
    },
}

// ---------------------------------------------------------------------------
// Worker pool
// ---------------------------------------------------------------------------

type WorkerPool = Arc<DashMap<String, WorkerEndpoint>>;

fn grpc_addr(ep: &WorkerEndpoint) -> String {
    format!("http://{}:{}", ep.ip, ep.port)
}

async fn connect_client(ep: &WorkerEndpoint) -> Result<InferenceServiceClient<Channel>> {
    let addr = grpc_addr(ep);
    let client = InferenceServiceClient::connect(addr).await?;
    Ok(client)
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

fn start_discovery(pool: WorkerPool) {
    match discovery::browse_workers(64) {
        Ok((mut rx, daemon)) => {
            tokio::spawn(async move {
                // Keep the daemon alive for the lifetime of this task.
                let _keep_alive = daemon;

                while let Some(event) = rx.recv().await {
                    match event {
                        DiscoveryEvent::Appeared(ep) => {
                            info!(
                                instance = %ep.instance_name,
                                ip = %ep.ip,
                                port = ep.port,
                                "Worker discovered"
                            );
                            pool.insert(ep.instance_name.clone(), ep);
                        }
                        DiscoveryEvent::Disappeared(name) => {
                            info!(instance = %name, "Worker lost");
                            pool.remove(&name);
                        }
                    }
                }
            });
        }
        Err(e) => warn!(error = %e, "mDNS browse failed; continuing without discovery"),
    }
}

// ---------------------------------------------------------------------------
// Subcommand handlers
// ---------------------------------------------------------------------------

async fn cmd_status(pool: &WorkerPool) -> Result<()> {
    if pool.is_empty() {
        println!("No workers discovered.");
        return Ok(());
    }

    for entry in pool.iter() {
        let ep = entry.value();
        match connect_client(ep).await {
            Ok(mut client) => match client.get_status(GetStatusRequest {}).await {
                Ok(resp) => {
                    let s = resp.into_inner();
                    println!(
                        "Worker: {}\n  Hostname:          {}\n  Loaded models:     {:?}\n  Total inferences:  {}\n  Hardware active:   {}\n",
                        s.worker_id,
                        s.hostname,
                        s.loaded_models,
                        s.total_inferences,
                        s.hardware_active,
                    );
                }
                Err(e) => warn!(worker = %ep.instance_name, error = %e, "GetStatus RPC failed"),
            },
            Err(e) => warn!(worker = %ep.instance_name, error = %e, "Connect failed"),
        }
    }
    Ok(())
}

async fn cmd_load_model(
    pool: &WorkerPool,
    file: &std::path::Path,
    name: &str,
    worker_filter: Option<&str>,
) -> Result<()> {
    let hef_data = tokio::fs::read(file).await?;

    let targets: Vec<WorkerEndpoint> = pool
        .iter()
        .filter(|e| {
            worker_filter
                .map(|f| e.instance_name.contains(f))
                .unwrap_or(true)
        })
        .map(|e| e.value().clone())
        .collect();

    if targets.is_empty() {
        return Err(anyhow!("No matching workers found"));
    }

    for ep in &targets {
        match connect_client(ep).await {
            Ok(mut client) => {
                let resp = client
                    .load_model(LoadModelRequest {
                        hef_data: hef_data.clone().into(),
                        model_name: name.to_string(),
                    })
                    .await?
                    .into_inner();

                if resp.success {
                    println!(
                        "Worker {}: model '{}' loaded (inputs={}, outputs={})",
                        ep.instance_name, name, resp.input_count, resp.output_count
                    );
                } else {
                    warn!(
                        worker = %ep.instance_name,
                        model = name,
                        error = %resp.error_msg,
                        "LoadModel failed"
                    );
                }
            }
            Err(e) => warn!(worker = %ep.instance_name, error = %e, "Connect failed"),
        }
    }
    Ok(())
}

async fn cmd_unload_model(
    pool: &WorkerPool,
    name: &str,
    worker_filter: Option<&str>,
) -> Result<()> {
    let targets: Vec<WorkerEndpoint> = pool
        .iter()
        .filter(|e| {
            worker_filter
                .map(|f| e.instance_name.contains(f))
                .unwrap_or(true)
        })
        .map(|e| e.value().clone())
        .collect();

    if targets.is_empty() {
        return Err(anyhow!("No matching workers found"));
    }

    for ep in &targets {
        match connect_client(ep).await {
            Ok(mut client) => {
                let resp = client
                    .unload_model(UnloadModelRequest {
                        model_name: name.to_string(),
                    })
                    .await?
                    .into_inner();

                if resp.success {
                    println!("Worker {}: model '{}' unloaded", ep.instance_name, name);
                } else {
                    warn!(
                        worker = %ep.instance_name,
                        model = name,
                        error = %resp.error_msg,
                        "UnloadModel failed"
                    );
                }
            }
            Err(e) => warn!(worker = %ep.instance_name, error = %e, "Connect failed"),
        }
    }
    Ok(())
}

async fn cmd_infer(
    pool: &WorkerPool,
    model: &str,
    input_path: &std::path::Path,
    worker_filter: Option<&str>,
    expected_output_size: u32,
) -> Result<()> {
    let input_data = tokio::fs::read(input_path).await?;

    // Pick the first matching worker (or a specific one).
    let ep = pool
        .iter()
        .find(|e| {
            worker_filter
                .map(|f| e.instance_name.contains(f))
                .unwrap_or(true)
        })
        .map(|e| e.value().clone())
        .ok_or_else(|| anyhow!("No matching workers found"))?;

    let mut client = connect_client(&ep).await?;
    let resp = client
        .run_inference(RunInferenceRequest {
            model_name: model.to_string(),
            input_data: input_data.into(),
            expected_output_size,
        })
        .await?
        .into_inner();

    if resp.success {
        println!(
            "Worker {}: inference OK — {} output bytes, {}ms latency",
            ep.instance_name,
            resp.output_data.len(),
            resp.latency_ms
        );
        // Print a hex preview of the first 32 bytes.
        let preview: Vec<String> = resp.output_data.iter().take(32).map(|b| format!("{:02x}", b)).collect();
        println!("Output preview: {}", preview.join(" "));
    } else {
        return Err(anyhow!("Inference failed: {}", resp.error_msg));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let pool: WorkerPool = Arc::new(DashMap::new());

    start_discovery(Arc::clone(&pool));

    info!(wait_ms = cli.discovery_wait_ms, "Waiting for worker discovery");
    sleep(Duration::from_millis(cli.discovery_wait_ms)).await;
    info!(found = pool.len(), "Discovery window closed");

    match &cli.command {
        Command::Status => cmd_status(&pool).await?,

        Command::LoadModel { file, name, worker } => {
            cmd_load_model(&pool, file, name, worker.as_deref()).await?
        }

        Command::UnloadModel { name, worker } => {
            cmd_unload_model(&pool, name, worker.as_deref()).await?
        }

        Command::Infer {
            model,
            input,
            worker,
            expected_output_size,
        } => cmd_infer(&pool, model, input, worker.as_deref(), *expected_output_size).await?,
    }

    Ok(())
}
