"""
============================================================
Speaking AI Module - TensorFlow & NumPy
============================================================
Modul ini mengimplementasikan model AI untuk speaking/voice:
  - Text-to-Speech (TTS) model menggunakan TensorFlow
  - Audio feature extraction menggunakan NumPy & librosa
  - Speech-to-Text preprocessing pipeline
  - Voice activity detection (VAD)

Author      : ML Project Team
Description : TensorFlow + NumPy based Speaking AI system
              for text-to-speech synthesis and audio analysis.
============================================================
"""

import numpy as np
import os
import json
import time
from datetime import datetime
from typing import Tuple, List, Dict, Optional

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses


class AudioPreprocessor:
    """Preprocessing audio menggunakan NumPy dan DSP techniques."""

    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13):
        """
        Inisialisasi Audio Preprocessor.

        Args:
            sample_rate: Sample rate audio.
            n_mfcc: Jumlah MFCC features.
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def generate_sine_wave(
        self,
        frequency: float = 440.0,
        duration: float = 1.0,
        amplitude: float = 0.8
    ) -> np.ndarray:
        """
        Generate sine wave audio.

        Args:
            frequency: Frekuensi dalam Hz.
            duration: Durasi dalam detik.
            amplitude: Amplitude (0.0 - 1.0).

        Returns:
            NumPy array berisi audio signal.
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        return wave

    def generate_complex_tone(
        self,
        frequencies: List[float],
        duration: float = 1.0,
        amplitudes: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Generate complex tone dari multiple frequencies.

        Args:
            frequencies: List of frequencies (Hz).
            duration: Durasi dalam detik.
            amplitudes: Amplitudes untuk setiap frekuensi.

        Returns:
            NumPy array berisi complex audio signal.
        """
        if amplitudes is None:
            amplitudes = [1.0 / len(frequencies)] * len(frequencies)

        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        wave = np.zeros_like(t)

        for freq, amp in zip(frequencies, amplitudes):
            wave += amp * np.sin(2 * np.pi * freq * t)

        # Normalize
        wave = wave / (np.max(np.abs(wave)) + 1e-8)
        return wave

    def compute_mfcc_numpy(
        self, audio: np.ndarray, n_fft: int = 512, hop_length: int = 256
    ) -> np.ndarray:
        """
        Compute MFCC features menggunakan NumPy (simplified).

        Args:
            audio: Audio signal array.
            n_fft: FFT window size.
            hop_length: Hop length.

        Returns:
            MFCC features array.
        """
        # Apply pre-emphasis
        pre_emphasis = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

        # Frame the signal
        n_samples = len(pre_emphasis)
        n_frames = 1 + (n_samples - n_fft) // hop_length

        if n_frames <= 0:
            return np.zeros((1, self.n_mfcc))

        # Create frames using stride tricks
        frames = np.lib.stride_tricks.sliding_window_view(
            pre_emphasis, n_fft
        )[::hop_length]

        # Apply Hanning window
        window = np.hanning(n_fft)
        frames = frames * window

        # Compute power spectrum
        fft_result = np.fft.rfft(frames, n_fft)
        power_spectrum = np.abs(fft_result) ** 2 / n_fft

        # Mel filterbank (simplified)
        n_filters = 26
        mel_points = np.linspace(0, self.n_mfcc * 100, n_filters + 2)
        filterbank = np.zeros((n_filters, len(power_spectrum[0])))

        for i in range(n_filters):
            left = int(mel_points[i] * len(power_spectrum[0]) / (self.n_mfcc * 100))
            right = int(mel_points[i + 1] * len(power_spectrum[0]) / (self.n_mfcc * 100))
            center = int(mel_points[i + 2] * len(power_spectrum[0]) / (self.n_mfcc * 100))
            filterbank[i, left:center] = np.linspace(0, 1, center - left)
            filterbank[i, center:right] = np.linspace(1, 0, right - center)

        filter_energies = power_spectrum @ filterbank.T

        # Log and DCT
        filter_energies = np.where(filter_energies == 0, 1e-10, filter_energies)
        log_energies = np.log(filter_energies)

        # Apply DCT
        mfcc = np.zeros((len(log_energies), self.n_mfcc))
        for i in range(self.n_mfcc):
            for j in range(len(log_energies)):
                mfcc[j, i] = np.sum(
                    log_energies[j] * np.cos(np.pi * i * (np.arange(n_filters) + 0.5) / n_filters)
                )

        return mfcc

    def add_noise(self, audio: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add random noise ke audio signal."""
        noise = noise_level * np.random.randn(len(audio))
        return audio + noise

    def apply_gain(self, audio: np.ndarray, gain_db: float = 3.0) -> np.ndarray:
        """Apply gain (dB) ke audio signal."""
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear

    def detect_voice_activity(
        self, audio: np.ndarray, threshold: float = 0.02, frame_size: int = 1024
    ) -> np.ndarray:
        """
        Voice Activity Detection (VAD) sederhana berbasis energy.

        Args:
            audio: Audio signal.
            threshold: Energy threshold.
            frame_size: Frame size.

        Returns:
            Binary array (1 = voice, 0 = silence).
        """
        n_frames = len(audio) // frame_size
        vad = np.zeros(n_frames)

        for i in range(n_frames):
            frame = audio[i * frame_size: (i + 1) * frame_size]
            energy = np.mean(frame ** 2)
            vad[i] = 1 if energy > threshold else 0

        return vad


class TextToSpeechModel:
    """
    Text-to-Speech Model menggunakan TensorFlow.
    Mengimplementasikan model sequence-to-sequence untuk speech synthesis.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        max_text_length: int = 50,
        audio_feature_dim: int = 80,
        embedding_dim: int = 256,
        hidden_units: int = 512,
    ):
        """
        Inisialisasi TTS Model.

        Args:
            vocab_size: Ukuran vocabulary.
            max_text_length: Panjang maksimal teks input.
            audio_feature_dim: Dimensi audio features (mel-spectrogram).
            embedding_dim: Dimensi embedding.
            hidden_units: Hidden units untuk LSTM.
        """
        self.vocab_size = vocab_size
        self.max_text_length = max_text_length
        self.audio_feature_dim = audio_feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.model = None
        self.encoder = None
        self.decoder = None

    def build_model(self) -> keras.Model:
        """
        Build TTS model architecture (Seq2Seq with Attention).
        """
        # Text encoder
        text_input = layers.Input(shape=(self.max_text_length,), name="text_input")
        embedding = layers.Embedding(self.vocab_size, self.embedding_dim)(text_input)
        encoder_lstm = layers.LSTM(
            self.hidden_units, return_sequences=True, return_state=True, name="encoder_lstm"
        )
        encoder_output, state_h, state_c = encoder_lstm(embedding)

        # Attention mechanism
        attention = layers.Dense(1, activation="tanh")(encoder_output)
        attention_weights = layers.Softmax(axis=1, name="attention_weights")(attention)
        context = layers.Dot(axes=[1, 1])([attention_weights, encoder_output])

        # Audio decoder
        decoder_input = layers.Input(shape=(None, self.audio_feature_dim), name="audio_input")
        decoder_lstm = layers.LSTM(
            self.hidden_units, return_sequences=True, name="decoder_lstm"
        )

        # Concatenate context with decoder input
        context_repeated = layers.RepeatVector(
            tf.keras.backend.shape(decoder_input)[1] if tf.executing_eagerly() else 100
        )(context[:, -1:, :])

        # Simplified: merge encoder state with decoder
        combined = layers.Concatenate()([
            layers.Dense(self.hidden_units)(context_repeated),
            decoder_input
        ])

        decoder_output = decoder_lstm(combined)
        output = layers.TimeDistributed(
            layers.Dense(self.audio_feature_dim, activation="tanh")
        )(decoder_output)

        self.model = keras.Model(
            inputs=[text_input, decoder_input],
            outputs=output,
            name="TTS_Model"
        )

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )

        print(f"[OK] Built TTS Model with {self.model.count_params():,} parameters")
        self.model.summary()
        return self.model

    def build_lightweight_model(self) -> keras.Model:
        """
        Build lightweight TTS model untuk inference cepat.
        """
        # Simple feedforward TTS model
        text_input = layers.Input(shape=(self.max_text_length,), name="text_input")
        embedding = layers.Embedding(self.vocab_size, self.embedding_dim)(text_input)
        flatten = layers.GlobalAveragePooling1D()(embedding)

        dense1 = layers.Dense(self.hidden_units, activation="relu")(flatten)
        dropout = layers.Dropout(0.2)(dense1)
        dense2 = layers.Dense(self.hidden_units, activation="relu")(dropout)

        # Output mel-spectrogram frames
        output_frames = 100  # Fixed number of output frames
        output = layers.Reshape((output_frames, self.audio_feature_dim))(
            layers.Dense(output_frames * self.audio_feature_dim, activation="tanh")(dense2)
        )

        self.model = keras.Model(
            inputs=text_input,
            outputs=output,
            name="Lightweight_TTS_Model"
        )

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )

        print(f"[OK] Built Lightweight TTS Model with {self.model.count_params():,} parameters")
        self.model.summary()
        return self.model

    def generate_speech_from_text(
        self, text_indices: np.ndarray, model_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate speech audio dari text input.

        Args:
            text_indices: Tokenized text indices.
            model_path: Path ke model weights (opsional).

        Returns:
            Generated mel-spectrogram sebagai NumPy array.
        """
        if model_path and os.path.exists(model_path):
            self.model.load_weights(model_path)
            print(f"[OK] Loaded model weights from {model_path}")

        # Ensure input has correct shape
        if len(text_indices.shape) == 1:
            text_indices = np.expand_dims(text_indices, 0)

        # Pad or truncate
        if text_indices.shape[1] < self.max_text_length:
            padding = np.zeros((1, self.max_text_length - text_indices.shape[1]))
            text_indices = np.concatenate([text_indices, padding.astype(int)], axis=1)
        else:
            text_indices = text_indices[:, : self.max_text_length]

        prediction = self.model.predict(text_indices, verbose=0)
        print(f"[OK] Generated speech features: shape {prediction.shape}")
        return prediction[0]

    def train(
        self,
        X_text: np.ndarray,
        X_audio: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ) -> Dict:
        """
        Train TTS model.

        Args:
            X_text: Tokenized text data (batch, max_length).
            X_audio: Target mel-spectrogram (batch, n_frames, n_mel).
            epochs: Jumlah epoch training.
            batch_size: Batch size.
            validation_split: Validation split ratio.

        Returns:
            Training history dictionary.
        """
        if self.model is None:
            self.build_lightweight_model()

        # Ensure audio target has correct shape for lightweight model
        n_output_frames = 100
        if X_audio.shape[1] != n_output_frames:
            # Resample audio features to fixed length
            indices = np.linspace(0, X_audio.shape[1] - 1, n_output_frames).astype(int)
            X_audio = X_audio[:, indices, :]

        if X_audio.shape[2] != self.audio_feature_dim:
            # Pad or truncate mel features
            if X_audio.shape[2] < self.audio_feature_dim:
                padding = np.zeros(
                    (X_audio.shape[0], X_audio.shape[1],
                     self.audio_feature_dim - X_audio.shape[2])
                )
                X_audio = np.concatenate([X_audio, padding], axis=2)
            else:
                X_audio = X_audio[:, :, : self.audio_feature_dim]

        print(f"[INFO] Training TTS model...")
        print(f"  Text input shape: {X_text.shape}")
        print(f"  Audio target shape: {X_audio.shape}")
        print(f"  Epochs: {epochs}, Batch size: {batch_size}")

        start_time = time.time()
        history = self.model.fit(
            X_text, X_audio,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
        )
        training_time = time.time() - start_time

        result = {
            "loss": history.history["loss"],
            "val_loss": history.history.get("val_loss", []),
            "mae": history.history.get("mae", []),
            "val_mae": history.history.get("val_mae", []),
            "training_time_seconds": training_time,
            "epochs_completed": len(history.history["loss"]),
        }

        print(f"[OK] Training completed in {training_time:.2f} seconds")
        print(f"  Final loss: {result['loss'][-1]:.4f}")
        if result["val_loss"]:
            print(f"  Final val_loss: {result['val_loss'][-1]:.4f}")

        return result

    def save_model(self, filepath: str) -> None:
        """Save model weights."""
        if self.model is None:
            raise ValueError("Model has not been built yet!")
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        self.model.save_weights(filepath)
        print(f"[OK] Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load model weights."""
        if self.model is None:
            self.build_lightweight_model()
        self.model.load_weights(filepath)
        print(f"[OK] Model loaded from {filepath}")


class SpeakingAI:
    """
    Speaking AI System - Integrasi lengkap.
    Menggabungkan Audio Preprocessor dan TTS Model.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        vocab_size: int = 100,
        max_text_length: int = 50,
        audio_feature_dim: int = 80,
    ):
        """
        Inisialisasi Speaking AI System.

        Args:
            sample_rate: Sample rate audio.
            vocab_size: Ukuran vocabulary.
            max_text_length: Panjang maksimal teks.
            audio_feature_dim: Dimensi audio features.
        """
        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            n_mfcc=audio_feature_dim,
        )
        self.tts_model = TextToSpeechModel(
            vocab_size=vocab_size,
            max_text_length=max_text_length,
            audio_feature_dim=audio_feature_dim,
        )
        self.sample_rate = sample_rate
        self.vocab = {}
        self.inv_vocab = {}

    def build_vocabulary(self, texts: List[str]) -> None:
        """
        Build vocabulary dari list of texts.

        Args:
            texts: List of text strings.
        """
        all_chars = sorted(set("".join(texts)))
        self.vocab = {ch: i + 2 for i, ch in enumerate(all_chars)}
        self.vocab["<PAD>"] = 0
        self.vocab["<UNK>"] = 1
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        print(f"[OK] Built vocabulary with {len(self.vocab)} characters")

    def text_to_indices(self, text: str) -> np.ndarray:
        """Convert text ke integer indices."""
        indices = [self.vocab.get(ch, 1) for ch in text.lower()]
        return np.array(indices, dtype=np.int32)

    def indices_to_text(self, indices: np.ndarray) -> str:
        """Convert integer indices ke text."""
        chars = [self.inv_vocab.get(i, "") for i in indices if i > 1]
        return "".join(chars)

    def generate_training_data(
        self, texts: List[str], n_mel_frames: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data dari teks.

        Args:
            texts: List of training texts.
            n_mel_frames: Number of mel-spectrogram frames.

        Returns:
            Tuple (X_text, X_audio) untuk training.
        """
        if not self.vocab:
            self.build_vocabulary(texts)

        X_text = []
        X_audio = []

        for text in texts:
            # Tokenize text
            indices = self.text_to_indices(text)
            # Pad/truncate
            if len(indices) < self.tts_model.max_text_length:
                indices = np.pad(
                    indices,
                    (0, self.tts_model.max_text_length - len(indices)),
                    constant_values=0,
                )
            else:
                indices = indices[: self.tts_model.max_text_length]

            X_text.append(indices)

            # Generate synthetic mel-spectrogram from text
            np.random.seed(hash(text) % (2**32))
            mel_spec = np.random.randn(n_mel_frames, self.tts_model.audio_feature_dim) * 0.1

            # Create patterns based on text length and characters
            for i, ch in enumerate(text[:n_mel_frames]):
                char_val = self.vocab.get(ch.lower(), 1) / len(self.vocab)
                mel_spec[i] = char_val + np.random.randn(self.tts_model.audio_feature_dim) * 0.05

            X_audio.append(mel_spec)

        X_text = np.array(X_text, dtype=np.int32)
        X_audio = np.array(X_audio, dtype=np.float32)

        print(f"[OK] Generated training data: {X_text.shape}, {X_audio.shape}")
        return X_text, X_audio

    def speak(
        self,
        text: str,
        model_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate speech dari text input.

        Args:
            text: Input text.
            model_path: Path ke trained model (opsional).

        Returns:
            Generated audio features array.
        """
        if not self.vocab:
            self.build_vocabulary([text])

        indices = self.text_to_indices(text)

        # Build model if not built
        if self.tts_model.model is None:
            self.tts_model.build_lightweight_model()

        # Generate
        mel_features = self.tts_model.generate_speech_from_text(
            indices, model_path=model_path
        )

        print(f"[OK] Speaking: '{text[:50]}...'")
        print(f"  Generated {mel_features.shape[0]} frames of audio features")
        return mel_features


# =============================================================
# DEMO / MAIN
# =============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  SPEAKING AI - DEMO")
    print("=" * 60)

    # 1. Audio Preprocessor demo
    print("\n--- Audio Preprocessing ---")
    preprocessor = AudioPreprocessor(sample_rate=22050, n_mfcc=13)

    # Generate tones
    sine_wave = preprocessor.generate_sine_wave(frequency=440, duration=0.5)
    complex_tone = preprocessor.generate_complex_tone(
        frequencies=[261.63, 329.63, 392.00], duration=0.5
    )
    noisy_audio = preprocessor.add_noise(sine_wave, noise_level=0.05)
    vad_result = preprocessor.detect_voice_activity(noisy_audio)

    # MFCC
    mfcc_features = preprocessor.compute_mfcc_numpy(sine_wave)
    print(f"  MFCC features shape: {mfcc_features.shape}")

    # 2. Speaking AI demo
    print("\n--- Speaking AI System ---")
    ai = SpeakingAI(sample_rate=22050, vocab_size=100, max_text_length=30)

    # Sample texts
    sample_texts = [
        "hello world",
        "machine learning is awesome",
        "text to speech synthesis",
        "artificial intelligence",
        "deep learning models",
        "natural language processing",
        "neural networks",
        "speech recognition",
        "voice synthesis",
        "audio processing",
    ]

    # Generate training data
    X_text, X_audio = ai.generate_training_data(sample_texts)

    # Train lightweight model
    print("\n--- Training TTS Model ---")
    ai.tts_model.build_lightweight_model()
    history = ai.tts_model.train(
        X_text, X_audio,
        epochs=10,
        batch_size=4,
        validation_split=0.2,
    )

    # Generate speech
    print("\n--- Generate Speech ---")
    test_text = "hello machine learning"
    mel_output = ai.speak(test_text)
    print(f"  Output mel-spectrogram shape: {mel_output.shape}")

    # Save model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    ai.tts_model.save_model(os.path.join(model_dir, "tts_model.weights.h5"))

    print("\n" + "=" * 60)
    print("  SPEAKING AI DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
