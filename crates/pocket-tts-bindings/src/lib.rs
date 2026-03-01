use candle_core::Device;
use pocket_tts::{ModelState, TTSModel};
use pyo3::prelude::*;
use std::path::Path;

/// Python wrapper for the Rust TTSModel
#[pyclass]
struct PyTTSModel {
    inner: TTSModel,
}

#[pymethods]
impl PyTTSModel {
    /// Load the model from a specific checkpoint variant
    #[staticmethod]
    fn load(variant: &str) -> PyResult<Self> {
        let model = TTSModel::load(variant)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyTTSModel { inner: model })
    }

    /// Load the model with custom parameters
    #[staticmethod]
    fn load_with_params(
        variant: &str,
        temp: f32,
        lsd_decode_steps: usize,
        eos_threshold: f32,
    ) -> PyResult<Self> {
        let model = TTSModel::load_with_params(variant, temp, lsd_decode_steps, eos_threshold)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyTTSModel { inner: model })
    }

    /// Load model from custom file paths
    ///
    /// This function allows loading a TTS model from custom paths for the config file,
    /// weights file, and tokenizer file, instead of relying on the HuggingFace Hub.
    /// The config and tokenizer files are bundled with the wheel, so only the weights
    /// file path is required.
    ///
    /// # Arguments
    /// * `weights_path` - Path to the safetensors weights file
    /// * `temp` - Generation temperature (default: 0.6)
    /// * `lsd_decode_steps` - Number of LSD decode steps (default: 10)
    /// * `eos_threshold` - End-of-sequence threshold (default: 0.2)
    /// * `noise_clamp` - Optional noise clamping value
    /// * `device` - Device to load the model on
    ///
    /// # Returns
    /// PyTTSModel instance ready for generation
    #[staticmethod]
    #[pyo3(signature = (weights_path, temp=0.6, lsd_decode_steps=10, eos_threshold=0.2, noise_clamp=None, device="cpu"))]
    fn load_from_paths(
        weights_path: &str,
        temp: f32,
        lsd_decode_steps: usize,
        eos_threshold: f32,
        noise_clamp: Option<f32>,
        device: &str,
    ) -> PyResult<Self> {
        let device = match device {
            "cpu" => Device::Cpu,
            "cuda" => Device::new_cuda(0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?,
            "metal" => Device::new_metal(0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid device. Use 'cpu', 'cuda', or 'metal'",
                ));
            }
        };

        // Use the bundled config and tokenizer files
        let config_path = "crates/pocket-tts/config/b6369a24.yaml";
        let tokenizer_path = "crates/pocket-tts/assets/tokenizer.json";

        let model = TTSModel::load_from_paths(
            config_path,
            weights_path,
            tokenizer_path,
            temp,
            lsd_decode_steps,
            eos_threshold,
            noise_clamp,
            &device,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(PyTTSModel { inner: model })
    }

    /// Generate audio from text
    ///
    /// Returns:
    ///     One-dimensional list of floats representing the audio samples.
    #[pyo3(signature = (text, valid_voice_state=None))]
    fn generate(&self, text: &str, valid_voice_state: Option<&str>) -> PyResult<Vec<f32>> {
        // Create a default voice state or load one if provided
        // Ideally we should expose a VoiceState object to Python too, but for now
        // let's just make it simple or require a path to a voice prompt file?

        if let Some(path) = valid_voice_state {
            let state = self.get_voice_state(path)?;

            let audio_tensor = self
                .inner
                .generate(text, &state.inner)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            println!(
                "DEBUG: Output tensor shape before flatten: {:?}",
                audio_tensor.shape()
            );
            let audio_data = audio_tensor
                .flatten_all()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
                .to_vec1::<f32>()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            Ok(audio_data)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Voice state path must be provided for now",
            ))
        }
    }

    /// Create a voice state from an audio file path
    /// Create a voice state from an audio file path or safetensors file
    fn get_voice_state(&self, path: &str) -> PyResult<PyModelState> {
        let path_obj = Path::new(path);
        let ext = path_obj
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let state = if ext == "safetensors" {
            self.inner
                .get_voice_state_from_prompt_file(path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        } else {
            self.inner
                .get_voice_state(path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        };

        Ok(PyModelState { inner: state })
    }

    /// Convert a WAV audio file to a safetensors voice prompt file
    ///
    /// This allows pre-computing voice prompts for faster loading during generation.
    /// The output safetensors file can be passed to `get_voice_state` or `generate`.
    ///
    /// Args:
    ///     audio_path: Path to the input WAV file
    ///     safetensors_path: Path where the output safetensors file will be saved
    ///
    /// Example:
    ///     model.save_audio_as_voice_prompt("my_voice.wav", "my_voice.safetensors")
    ///     model.generate("Hello!", "my_voice.safetensors")
    fn save_audio_as_voice_prompt(&self, audio_path: &str, safetensors_path: &str) -> PyResult<()> {
        self.inner
            .save_audio_as_voice_prompt(audio_path, safetensors_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }

    /// Generate using the voice state
    fn generate_audio(&self, text: &str, voice_state: &PyModelState) -> PyResult<Vec<f32>> {
        let audio_tensor = self
            .inner
            .generate(text, &voice_state.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        println!(
            "DEBUG: Output tensor shape before flatten: {:?}",
            audio_tensor.shape()
        );
        let audio_data = audio_tensor
            .flatten_all()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(audio_data)
    }
}

/// Python wrapper for ModelState
#[pyclass]
#[derive(Clone)]
struct PyModelState {
    inner: ModelState,
}

/// The main module exposed to Python
#[pymodule]
fn pocket_tts_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTTSModel>()?;
    m.add_class::<PyModelState>()?;
    Ok(())
}
