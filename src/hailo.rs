//! HailoRT backend via the `hailort-sys` FFI crate.
//!
//! One `HailoBackend` owns a single vdevice. Each loaded model occupies one
//! configured network group with pre-created vstreams.  All FFI calls are
//! synchronous; callers must dispatch them through
//! `tokio::task::spawn_blocking`.

use std::collections::HashMap;
use std::ffi::CStr;
use std::ptr;
use std::time::Instant;

use anyhow::{anyhow, Result};
use hailort_sys::*;
use tracing::info;

// ---------------------------------------------------------------------------
// Safety declaration
// ---------------------------------------------------------------------------
//
// HailoBackend holds raw C pointers. We serialise every access through a
// `tokio::sync::Mutex<HailoBackend>` in WorkerState, which guarantees that
// no two threads touch the pointers concurrently.
unsafe impl Send for HailoBackend {}

// ---------------------------------------------------------------------------
// Per-model state
// ---------------------------------------------------------------------------

struct LoadedModel {
    network_group: hailo_configured_network_group,
    input_vstreams: Vec<hailo_input_vstream>,
    output_vstreams: Vec<hailo_output_vstream>,
    /// Total expected output buffer size in bytes across all output vstreams.
    output_frame_size: usize,
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

pub struct HailoBackend {
    vdevice: hailo_vdevice,
    models: HashMap<String, LoadedModel>,
}

impl HailoBackend {
    /// Opens the Hailo vdevice with default parameters (round-robin scheduler,
    /// one device, no multi-process service).  Returns an error if no Hailo
    /// device is present on the system.
    pub fn try_new() -> Result<Self> {
        let mut params: hailo_vdevice_params_t = unsafe { std::mem::zeroed() };
        let status = unsafe { hailo_init_vdevice_params(&mut params) };
        check_status(status, "hailo_init_vdevice_params")?;

        let mut vdevice: hailo_vdevice = ptr::null_mut();
        let status = unsafe { hailo_create_vdevice(&mut params, &mut vdevice) };
        check_status(status, "hailo_create_vdevice")?;

        info!("Hailo vdevice opened");
        Ok(Self {
            vdevice,
            models: HashMap::new(),
        })
    }

    /// Loads a HEF from a byte buffer and configures input/output vstreams
    /// ready for inference.  Returns `(input_count, output_count)`.
    pub fn load_model(&mut self, model_name: &str, hef_data: &[u8]) -> Result<(u32, u32)> {
        // 1. Parse the HEF buffer.
        let mut hef: hailo_hef = ptr::null_mut();
        let status = unsafe {
            hailo_create_hef_buffer(&mut hef, hef_data.as_ptr() as *const _, hef_data.len())
        };
        check_status(status, "hailo_create_hef_buffer")?;

        // From here on, every early-return path must call hailo_release_hef.
        let result = self.load_model_inner(hef, model_name);
        unsafe { hailo_release_hef(hef) };
        result
    }

    fn load_model_inner(
        &mut self,
        hef: hailo_hef,
        model_name: &str,
    ) -> Result<(u32, u32)> {
        // 2. Query vstream infos while the HEF is still alive, so we know the
        //    stream names, directions, formats, and shapes.
        let (input_params, output_params, output_frame_size, input_count, output_count) =
            build_vstream_params(hef, model_name)?;

        // 3. Initialise configure params from the vdevice + HEF combination.
        //    hailo_configure_params_t is large (~64 KB); create on the heap.
        let mut configure_params = Box::new(unsafe {
            std::mem::zeroed::<hailo_configure_params_t>()
        });
        let status = unsafe {
            hailo_init_configure_params_by_vdevice(
                self.vdevice,
                hef,
                configure_params.as_mut(),
            )
        };
        check_status(status, "hailo_init_configure_params_by_vdevice")?;

        // 4. Configure vdevice — produces one or more configured network groups.
        let mut network_groups: [hailo_configured_network_group; HAILO_MAX_NETWORK_GROUPS] =
            [ptr::null_mut(); HAILO_MAX_NETWORK_GROUPS];
        let mut ng_count: usize = HAILO_MAX_NETWORK_GROUPS;
        let status = unsafe {
            hailo_configure_vdevice(
                self.vdevice,
                hef,
                configure_params.as_mut(),
                network_groups.as_mut_ptr(),
                &mut ng_count,
            )
        };
        check_status(status, "hailo_configure_vdevice")?;

        if ng_count == 0 {
            return Err(anyhow!("HEF '{}' contains no network groups", model_name));
        }

        // Warn about unused network groups — we only use the first.
        if ng_count > 1 {
            tracing::warn!(
                model = model_name,
                ng_count,
                "HEF has multiple network groups; using the first"
            );
            for &ng in &network_groups[1..ng_count] {
                unsafe { hailo_release_network_group(ng) };
            }
        }
        let network_group = network_groups[0];

        // 5. Create input vstreams.
        let mut raw_inputs: Vec<hailo_input_vstream> =
            vec![ptr::null_mut(); input_params.len()];
        let status = unsafe {
            hailo_create_input_vstreams(
                network_group,
                input_params.as_ptr(),
                input_params.len(),
                raw_inputs.as_mut_ptr(),
            )
        };
        if let Err(e) = check_status(status, "hailo_create_input_vstreams") {
            unsafe { hailo_release_network_group(network_group) };
            return Err(e);
        }

        // 6. Create output vstreams.
        let mut raw_outputs: Vec<hailo_output_vstream> =
            vec![ptr::null_mut(); output_params.len()];
        let status = unsafe {
            hailo_create_output_vstreams(
                network_group,
                output_params.as_ptr(),
                output_params.len(),
                raw_outputs.as_mut_ptr(),
            )
        };
        if let Err(e) = check_status(status, "hailo_create_output_vstreams") {
            for &vs in &raw_inputs {
                unsafe { hailo_release_input_vstream(vs) };
            }
            unsafe { hailo_release_network_group(network_group) };
            return Err(e);
        }

        info!(
            model = model_name,
            inputs = input_count,
            outputs = output_count,
            output_bytes = output_frame_size,
            "Model loaded"
        );
        self.models.insert(
            model_name.to_string(),
            LoadedModel {
                network_group,
                input_vstreams: raw_inputs,
                output_vstreams: raw_outputs,
                output_frame_size,
            },
        );
        Ok((input_count, output_count))
    }

    /// Runs inference synchronously.
    ///
    /// **Must be called from a blocking thread** (e.g. via
    /// `tokio::task::spawn_blocking`) — the HailoRT C API does not have async
    /// variants and would otherwise block the tokio executor.
    pub fn run_inference_sync(
        &self,
        model_name: &str,
        input_data: &[u8],
        expected_output_size: usize,
    ) -> Result<(Vec<u8>, u32)> {
        let model = self
            .models
            .get(model_name)
            .ok_or_else(|| anyhow!("model '{}' not loaded", model_name))?;

        let out_size = if expected_output_size > 0 {
            expected_output_size
        } else {
            model.output_frame_size
        };

        let start = Instant::now();

        // Write input data, splitting evenly across all input vstreams.
        if !model.input_vstreams.is_empty() {
            let n = model.input_vstreams.len();
            let chunk_size = input_data.len() / n;
            for (i, &vstream) in model.input_vstreams.iter().enumerate() {
                let begin = i * chunk_size;
                let end = if i + 1 == n { input_data.len() } else { begin + chunk_size };
                let chunk = &input_data[begin..end];
                let status = unsafe {
                    hailo_input_vstream_write(
                        vstream,
                        chunk.as_ptr() as *const _,
                        chunk.len(),
                    )
                };
                check_status(status, "hailo_input_vstream_write")?;
            }
        }

        // Collect output data, splitting the output buffer across all output vstreams.
        let mut output = vec![0u8; out_size];
        if !model.output_vstreams.is_empty() {
            let n = model.output_vstreams.len();
            let chunk_size = out_size / n;
            for (i, &vstream) in model.output_vstreams.iter().enumerate() {
                let begin = i * chunk_size;
                let end = if i + 1 == n { out_size } else { begin + chunk_size };
                let chunk = &mut output[begin..end];
                let status = unsafe {
                    hailo_output_vstream_read(
                        vstream,
                        chunk.as_mut_ptr() as *mut _,
                        chunk.len(),
                    )
                };
                check_status(status, "hailo_output_vstream_read")?;
            }
        }

        let latency_ms = start.elapsed().as_millis() as u32;
        Ok((output, latency_ms))
    }

    /// Releases all vstreams and the configured network group for `model_name`.
    pub fn unload_model(&mut self, model_name: &str) -> Result<()> {
        let model = self
            .models
            .remove(model_name)
            .ok_or_else(|| anyhow!("model '{}' not loaded", model_name))?;
        release_model(model);
        info!(model = model_name, "Model unloaded");
        Ok(())
    }

    pub fn loaded_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
}

impl Drop for HailoBackend {
    fn drop(&mut self) {
        // Release models in insertion order; drain so we own each LoadedModel.
        for (name, model) in self.models.drain() {
            release_model(model);
            info!(model = name, "Model released on drop");
        }
        if !self.vdevice.is_null() {
            unsafe { hailo_release_vdevice(self.vdevice) };
            info!("Hailo vdevice closed");
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Releases vstreams and the network group in the correct order.
fn release_model(model: LoadedModel) {
    for vstream in model.input_vstreams {
        unsafe { hailo_release_input_vstream(vstream) };
    }
    for vstream in model.output_vstreams {
        unsafe { hailo_release_output_vstream(vstream) };
    }
    unsafe { hailo_release_network_group(model.network_group) };
}

/// Queries vstream info from a live HEF handle and builds the param arrays
/// needed by `hailo_create_input_vstreams` / `hailo_create_output_vstreams`.
///
/// Returns `(input_params, output_params, total_output_frame_size,
///           input_count, output_count)`.
fn build_vstream_params(
    hef: hailo_hef,
    model_name: &str,
) -> Result<(
    Vec<hailo_input_vstream_params_by_name_t>,
    Vec<hailo_output_vstream_params_by_name_t>,
    usize,
    u32,
    u32,
)> {
    let mut infos: Vec<hailo_vstream_info_t> = (0..HAILO_MAX_STREAMS_COUNT)
        .map(|_| unsafe { std::mem::zeroed() })
        .collect();
    let mut count: usize = HAILO_MAX_STREAMS_COUNT;

    let status = unsafe {
        hailo_hef_get_vstream_infos(hef, ptr::null(), infos.as_mut_ptr(), &mut count)
    };
    check_status(status, "hailo_hef_get_vstream_infos")?;

    tracing::debug!(model = model_name, vstream_count = count, "Queried vstream infos");

    let mut input_params: Vec<hailo_input_vstream_params_by_name_t> = Vec::new();
    let mut output_params: Vec<hailo_output_vstream_params_by_name_t> = Vec::new();
    let mut output_frame_size: usize = 0;

    for info in &infos[..count] {
        if info.direction == HAILO_H2D_STREAM {
            input_params.push(hailo_input_vstream_params_by_name_t {
                name: info.name,
                params: default_vstream_params(),
            });
        } else if info.direction == HAILO_D2H_STREAM {
            output_params.push(hailo_output_vstream_params_by_name_t {
                name: info.name,
                params: default_vstream_params(),
            });
            output_frame_size += frame_size_of(info);
        }
    }

    let input_count = input_params.len() as u32;
    let output_count = output_params.len() as u32;
    Ok((input_params, output_params, output_frame_size, input_count, output_count))
}

/// Constructs default vstream params: AUTO format, library-default timeout and
/// queue size, no statistics collection overhead.
fn default_vstream_params() -> hailo_vstream_params_t {
    hailo_vstream_params_t {
        user_buffer_format: hailo_format_t {
            type_: HAILO_FORMAT_TYPE_AUTO,
            order: HAILO_FORMAT_ORDER_AUTO,
            flags: HAILO_FORMAT_FLAGS_NONE,
        },
        timeout_ms: HAILO_DEFAULT_VSTREAM_TIMEOUT_MS as u32,
        queue_size: HAILO_DEFAULT_VSTREAM_QUEUE_SIZE as u32,
        vstream_stats_flags: HAILO_VSTREAM_STATS_NONE,
        pipeline_elements_stats_flags: HAILO_PIPELINE_ELEM_STATS_NONE,
    }
}

/// Estimates the output frame size in bytes from a vstream info struct.
///
/// For NMS outputs, the calculation is approximate (class × max_bbox × 7
/// elements, 2 bytes each for u16 format); for 3D tensor outputs it is exact.
/// Callers may override this via `expected_output_size` in the RPC request.
fn frame_size_of(info: &hailo_vstream_info_t) -> usize {
    let bpe: usize = match info.format.type_ {
        HAILO_FORMAT_TYPE_FLOAT32 => 4,
        HAILO_FORMAT_TYPE_UINT16  => 2,
        _                         => 1, // UINT8 or AUTO → quantised UINT8
    };
    let is_nms = matches!(
        info.format.order,
        HAILO_FORMAT_ORDER_HAILO_NMS
        | HAILO_FORMAT_ORDER_HAILO_NMS_WITH_BYTE_MASK
        | HAILO_FORMAT_ORDER_HAILO_NMS_ON_CHIP
        | HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS
        | HAILO_FORMAT_ORDER_HAILO_NMS_BY_SCORE
    );
    if is_nms {
        // Each detection: class_id (u16) + score (u16) + bbox 4×(u16) = 7 × u16
        let nms = unsafe { info.shape.nms_shape };
        nms.number_of_classes as usize * nms.max_bboxes_per_class as usize * 7 * bpe
    } else {
        let shape = unsafe { info.shape.shape };
        shape.height as usize * shape.width as usize * shape.features as usize * bpe
    }
}

/// Converts a `hailo_status` into a `Result`, embedding the status code and
/// the human-readable message from the HailoRT library.
pub fn check_status(status: hailo_status, context: &str) -> Result<()> {
    if status == HAILO_SUCCESS {
        return Ok(());
    }
    let msg = unsafe {
        let ptr = hailo_get_status_message(status);
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    };
    Err(anyhow!("{} failed (status {}): {}", context, status, msg))
}
