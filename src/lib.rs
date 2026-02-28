pub mod hailo;
pub mod discovery;

pub mod proto {
    tonic::include_proto!("inference");
}
