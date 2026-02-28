use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use clap::Parser;
use tokio::sync::Mutex;
use tonic::{transport::Server, Request, Response, Status};
use tracing::info;
use uuid::Uuid;

use raspberry_hailort_networked::{
    discovery,
    hailo::HailoBackend,
    proto::{
        inference_service_server::{InferenceService, InferenceServiceServer},
        GetStatusRequest, GetStatusResponse, LoadModelRequest, LoadModelResponse,
        RunInferenceRequest, RunInferenceResponse, UnloadModelRequest, UnloadModelResponse,
    },
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "worker", about = "HailoRT distributed inference worker")]
struct Cli {
    /// gRPC listen port
    #[arg(long, default_value_t = 50051)]
    port: u16,

    /// mDNS instance name (defaults to a new UUID if omitted)
    #[arg(long)]
    instance_name: Option<String>,
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

struct WorkerState {
    worker_id: String,
    hostname: String,
    /// Serialises access to the Hailo device — the chip is single-stream.
    backend: Mutex<HailoBackend>,
    total_inferences: AtomicU64,
}

// ---------------------------------------------------------------------------
// gRPC service implementation
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct WorkerService {
    state: Arc<WorkerState>,
}

#[tonic::async_trait]
impl InferenceService for WorkerService {
    async fn load_model(
        &self,
        request: Request<LoadModelRequest>,
    ) -> Result<Response<LoadModelResponse>, Status> {
        let req = request.into_inner();
        let state = Arc::clone(&self.state);
        let model_name = req.model_name.clone();
        let hef_data: Vec<u8> = req.hef_data.into();

        // HailoRT's configure path is synchronous — run on the blocking pool.
        let result = tokio::task::spawn_blocking(move || {
            let mut backend = state.backend.blocking_lock();
            backend.load_model(&model_name, &hef_data)
        })
        .await
        .map_err(|e| anyhow!("spawn_blocking panicked: {}", e))
        .and_then(|r| r);

        match result {
            Ok((input_count, output_count)) => {
                info!(model = %req.model_name, "Model loaded");
                Ok(Response::new(LoadModelResponse {
                    success: true,
                    error_msg: String::new(),
                    input_count,
                    output_count,
                }))
            }
            Err(e) => Ok(Response::new(LoadModelResponse {
                success: false,
                error_msg: e.to_string(),
                input_count: 0,
                output_count: 0,
            })),
        }
    }

    async fn unload_model(
        &self,
        request: Request<UnloadModelRequest>,
    ) -> Result<Response<UnloadModelResponse>, Status> {
        let req = request.into_inner();
        let state = Arc::clone(&self.state);
        let model_name = req.model_name.clone();

        let result = tokio::task::spawn_blocking(move || {
            let mut backend = state.backend.blocking_lock();
            backend.unload_model(&model_name)
        })
        .await
        .map_err(|e| anyhow!("spawn_blocking panicked: {}", e))
        .and_then(|r| r);

        match result {
            Ok(()) => {
                info!(model = %req.model_name, "Model unloaded");
                Ok(Response::new(UnloadModelResponse {
                    success: true,
                    error_msg: String::new(),
                }))
            }
            Err(e) => Ok(Response::new(UnloadModelResponse {
                success: false,
                error_msg: e.to_string(),
            })),
        }
    }

    async fn run_inference(
        &self,
        request: Request<RunInferenceRequest>,
    ) -> Result<Response<RunInferenceResponse>, Status> {
        let req = request.into_inner();
        let state = Arc::clone(&self.state);
        let model_name = req.model_name.clone();
        let input_data: Vec<u8> = req.input_data.into();
        let expected = req.expected_output_size as usize;

        // run_inference_sync blocks on HailoRT DMA — always spawn_blocking.
        let result = tokio::task::spawn_blocking(move || {
            let backend = state.backend.blocking_lock();
            backend.run_inference_sync(&model_name, &input_data, expected)
        })
        .await
        .map_err(|e| anyhow!("spawn_blocking panicked: {}", e))
        .and_then(|r| r);

        match result {
            Ok((output, latency_ms)) => {
                self.state.total_inferences.fetch_add(1, Ordering::Relaxed);
                info!(
                    model = %req.model_name,
                    latency_ms,
                    "Inference complete"
                );
                Ok(Response::new(RunInferenceResponse {
                    success: true,
                    error_msg: String::new(),
                    output_data: output.into(),
                    latency_ms,
                }))
            }
            Err(e) => Ok(Response::new(RunInferenceResponse {
                success: false,
                error_msg: e.to_string(),
                output_data: vec![].into(),
                latency_ms: 0,
            })),
        }
    }

    async fn get_status(
        &self,
        _request: Request<GetStatusRequest>,
    ) -> Result<Response<GetStatusResponse>, Status> {
        let backend = self.state.backend.lock().await;
        Ok(Response::new(GetStatusResponse {
            worker_id: self.state.worker_id.clone(),
            hostname: self.state.hostname.clone(),
            loaded_models: backend.loaded_models(),
            total_inferences: self.state.total_inferences.load(Ordering::Relaxed),
            hardware_active: true,
        }))
    }
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

    let worker_id = cli
        .instance_name
        .clone()
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let hostname = discovery::local_hostname();

    let backend = HailoBackend::try_new()?;

    let state = Arc::new(WorkerState {
        worker_id: worker_id.clone(),
        hostname: hostname.clone(),
        backend: Mutex::new(backend),
        total_inferences: AtomicU64::new(0),
    });

    let svc = WorkerService {
        state: Arc::clone(&state),
    };

    // Keep the mDNS daemon alive for the process lifetime.
    let _mdns_daemon = discovery::register_worker(&worker_id, cli.port)?;

    let addr = format!("0.0.0.0:{}", cli.port).parse()?;
    info!(worker_id = %worker_id, hostname = %hostname, %addr, "Worker starting");

    Server::builder()
        .add_service(InferenceServiceServer::new(svc))
        .serve(addr)
        .await?;

    Ok(())
}
