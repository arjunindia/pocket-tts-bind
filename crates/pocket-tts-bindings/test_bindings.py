import sys
import os
import time
import argparse
import wave
import struct

try:
    import pocket_tts_bindings
    print("✅ Successfully imported pocket_tts_bindings")
except ImportError:
    print("❌ Failed to import pocket_tts_bindings. Make sure you have built the bindings with 'maturin develop'.")
    sys.exit(1)

def save_wav(filename, samples, sample_rate=24000):
    """Save float samples to a 16-bit PCM WAV file."""
    # Scale to 16-bit integer range
    scaled = [max(-32768, min(32767, int(s * 32767))) for s in samples]
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack('<' + 'h' * len(scaled), *scaled))
    print(f"💾 Saved audio to {filename}")

def resolve_voice(voice_spec):
    """Resolve voice specification to a local file path."""
    if os.path.exists(voice_spec):
        return os.path.abspath(voice_spec)
        
    PREDEFINED_VOICES = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
    
    if voice_spec.lower() in PREDEFINED_VOICES:
        print(f"Resolving predefined voice '{voice_spec}'...")
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id="kyutai/pocket-tts-without-voice-cloning",
                filename=f"embeddings/{voice_spec.lower()}.safetensors"
            )
            print(f"   Downloaded to: {path}")
            return path
        except ImportError:
            print("❌ 'huggingface_hub' not installed. Cannot download predefined voices.")
            print("   Run: pip install huggingface-hub")
            return None
        except Exception as e:
            print(f"❌ Failed to download voice: {e}")
            return None
            
    # Try finding relative to script
    alt_path = os.path.join(os.path.dirname(__file__), voice_spec)
    if os.path.exists(alt_path):
        return os.path.abspath(alt_path)

    # Try finding in project root (common for ref.wav)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    root_path = os.path.join(project_root, voice_spec)
    if os.path.exists(root_path):
        return root_path

    print(f"❌ Voice file '{voice_spec}' not found.")
    return None

def test_save_voice_prompt(model, wav_path, output_safetensors_path):
    """Test converting a WAV file to a safetensors voice prompt."""
    if not os.path.exists(wav_path):
        print(f"⚠️  WAV file '{wav_path}' not found. Skipping voice prompt creation test.")
        return None
    
    print(f"\n=== Converting WAV to Safetensors Voice Prompt ===")
    print(f"Input WAV: {wav_path}")
    print(f"Output safetensors: {output_safetensors_path}")
    
    try:
        t0 = time.time()
        model.save_audio_as_voice_prompt(wav_path, output_safetensors_path)
        t1 = time.time()
        
        # Check the output file exists and has reasonable size
        if os.path.exists(output_safetensors_path):
            file_size = os.path.getsize(output_safetensors_path)
            print(f"✅ Created safetensors voice prompt in {t1 - t0:.4f}s")
            print(f"   File size: {file_size / 1024:.1f} KB")
            return output_safetensors_path
        else:
            print(f"❌ Output file was not created")
            return None
    except Exception as e:
        print(f"❌ Failed to create voice prompt: {e}")
        return None


def test_generation(text, voice_spec, output_file, variant="b6369a24", model=None):
    voice_path = resolve_voice(voice_spec)
    if not voice_path:
        return

    print(f"Using voice reference: {voice_path}")
    
    if model is None:
        print(f"Loading model '{variant}'...")
        try:
            t0 = time.time()
            model = pocket_tts_bindings.PyTTSModel.load(variant)
            t1 = time.time()
            print(f"✅ Model loaded in {t1 - t0:.4f}s")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return model
    
    print(f"\nGenerating audio for: '{text}'")
    
    try:
        t0 = time.time()
        # Note: generate returns a list of floats
        audio = model.generate(text, voice_path)
        t1 = time.time()
        
        duration_sec = len(audio) / 24000.0 # Assuming 24khz
        print(f"✅ Generated {len(audio)} samples ({duration_sec:.2f}s audio) in {t1 - t0:.4f}s")
        print(f"⚡ Real-time factor: {(t1 - t0) / duration_sec:.4f}x (lower is better)")
        
        if output_file:
            save_wav(output_file, audio)
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Pocket TTS Python Bindings")
    parser.add_argument("--text", type=str, default="This is a test of the Pocket TTS Rust bindings generated via PyO3.", help="Text to generate")
    parser.add_argument("--voice", type=str, default=None, help="Path to voice file OR predefined voice name. If not provided, runs a demo suite.")
    parser.add_argument("--output", type=str, default="test_output.wav", help="Output wav file")
    parser.add_argument("--variant", type=str, default="b6369a24", help="Model variant")
    
    args = parser.parse_args()
    
    if args.voice:
        test_generation(args.text, args.voice, args.output, args.variant)
    else:
        print("🔍 No voice specified. Running demo suite...")
        
        # Load model once to reuse
        print("\n=== Test 1: Predefined Voice (Alba) ===")
        model = test_generation(args.text, "alba", "output_alba.wav", args.variant)
        
        if model:
            print("\n=== Test 2: Reference WAV (ref.wav) ===")
            # Check for common ref.wav locations
            ref_candidates = ["ref.wav", "../../../ref.wav", "d:/pocket-tts-candle/ref.wav"]
            ref_wav_path = None
            for cand in ref_candidates:
                resolved = resolve_voice(cand)
                if resolved:
                    ref_wav_path = resolved
                    test_generation(args.text, cand, "output_ref.wav", args.variant, model=model)
                    break
            
            if not ref_wav_path:
                print("⚠️  Could not find 'ref.wav' for second test. skipping.")
            
            # Test 3: Convert WAV to safetensors voice prompt
            if ref_wav_path:
                print("\n=== Test 3: Convert WAV to Safetensors Voice Prompt ===")
                safetensors_path = test_save_voice_prompt(model, ref_wav_path, "custom_voice.safetensors")
                
                if safetensors_path:
                    # Test 4: Generate using the newly created safetensors
                    print("\n=== Test 4: Generate with Custom Voice Prompt ===")
                    test_generation(
                        "This is a test using a custom voice prompt created from a WAV file.",
                        safetensors_path,
                        "output_custom_voice.wav",
                        args.variant,
                        model=model
                    )
                    test_generation(
                        "I love the kind of woman that will actually just kill me. Y'know, when I left the the house today I was thinking, damn, I hope some hot chick paints my brains all over some fucking hallway. And here we are. I mean really, just absolutely destroy me. I'm talkin' watermelon in the thighs level carnage.",
                        safetensors_path,
                        "output_custom_voice_long.wav",
                        args.variant,
                        model=model
                    )
