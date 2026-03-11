use candle_core::Device;
use pocket_tts::{ModelState, TTSModel};
use pyo3::prelude::*;
use std::path::Path;
use std::sync::{Arc, Mutex};

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
    /// This function allows loading a TTS model from only the weights file path,
    /// with the config and tokenizer files embedded directly in the library.
    /// This eliminates the need to manage separate config and tokenizer files.
    ///
    /// # Arguments
    /// * `weights_path` - Path to the safetensors weights file
    /// * `temp` - Generation temperature (default: 0.7)
    /// * `lsd_decode_steps` - Number of LSD decode steps (default: 1)
    /// * `eos_threshold` - End-of-sequence threshold (default: -4.0)
    /// * `noise_clamp` - Optional noise clamping value (default: None)
    /// * `device` - Device to load the model on
    ///
    /// # Returns
    /// PyTTSModel instance ready for generation
    #[staticmethod]
    #[pyo3(signature = (weights_path, temp=0.7, lsd_decode_steps=1, eos_threshold=-4.0, noise_clamp=None, device="cpu"))]
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

        // Embed config and tokenizer content directly in the library
        // This eliminates the need to manage separate files
        let config_content = include_str!("../../pocket-tts/config/b6369a24.yaml");
        let tokenizer_content = include_bytes!("../../pocket-tts/assets/tokenizer.model");

        let model = TTSModel::load_from_strings(
            weights_path,
            config_content,
            tokenizer_content,
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

            let audio_tensor = Python::attach(|py| {
                py.detach(|| {
                    self.inner
                        .generate(text, &state.inner)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
                })
            })?;

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
        let audio_tensor = Python::attach(|py| {
            py.detach(|| {
                self.inner
                    .generate(text, &voice_state.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            })
        })?;

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

    /// Generate audio from text in chunks (streaming)
    ///
    /// Returns an iterator that yields chunks of audio samples.
    /// Each chunk contains the samples for one Mimi frame.
    ///
    /// Args:
    ///     text: Text to generate speech from
    ///     valid_voice_state: Path to voice state file (.wav or .safetensors)
    ///
    /// Returns:
    ///     Iterator yielding lists of floats (audio samples per frame)
    #[pyo3(signature = (text, valid_voice_state=None))]
    fn generate_chunked(&self, text: &str, valid_voice_state: Option<&str>) -> PyResult<PyAudioChunkIterator> {
        if let Some(path) = valid_voice_state {
            let state = self.get_voice_state(path)?;
            let iter = self.inner.generate_stream_owned(text, &state.inner);
            Ok(PyAudioChunkIterator {
                inner: Arc::new(Mutex::new(iter)),
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Voice state path must be provided for now",
            ))
        }
    }

    /// Generate audio using a voice state in chunks (streaming)
    ///
    /// Returns an iterator that yields chunks of audio samples.
    /// Each chunk contains the samples for one Mimi frame.
    ///
    /// Args:
    ///     text: Text to generate speech from
    ///     voice_state: Voice state object
    ///
    /// Returns:
    ///     Iterator yielding lists of floats (audio samples per frame)
    fn generate_audio_chunked(&self, text: &str, voice_state: &PyModelState) -> PyResult<PyAudioChunkIterator> {
        let iter = self.inner.generate_stream_owned(text, &voice_state.inner);
        Ok(PyAudioChunkIterator {
            inner: Arc::new(Mutex::new(iter)),
        })
    }
}

/// Python iterator for audio chunks
#[pyclass]
struct PyAudioChunkIterator {
    inner: Arc<Mutex<Box<dyn Iterator<Item = Result<candle_core::Tensor, anyhow::Error>> + Send>>>,
}

#[pymethods]
impl PyAudioChunkIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRefMut<'_, Self>) -> Option<Result<Vec<f32>, PyErr>> {
        let iter = slf.inner.clone();
        Python::attach(|py| {
            py.detach(|| {
                let mut guard = iter.lock().unwrap();
                guard.next().map(|result| {
                    result
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
                        .and_then(|tensor| {
                            tensor
                                .flatten_all()
                                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
                                .to_vec1::<f32>()
                                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
                        })
                })
            })
        })
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
