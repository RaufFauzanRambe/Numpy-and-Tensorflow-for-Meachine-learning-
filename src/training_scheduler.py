"""
============================================================
Training Scheduler - Train with O'Clock
============================================================
Modul untuk menjadwalkan training model ML secara otomatis
berdasarkan waktu (o'clock) yang ditentukan.

Fitur:
  - Jadwal training berbasis jam tertentu (o'clock)
  - Cron-like scheduling
  - Callback system untuk pre/post training
  - Training history tracking & logging
  - Model checkpointing otomatis
  - NumPy array-based data pipeline

Author      : ML Project Team
Description : Automated training scheduler that runs ML model
              training at specified times ("o'clock" scheduling).
============================================================
"""

import numpy as np
import os
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple, Any

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks

try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False
    print("[WARN] 'schedule' package not installed. Install with: pip install schedule")

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False
    print("[WARN] 'APScheduler' not installed. Install with: pip install APScheduler")


# =============================================================
# LOGGING SETUP
# =============================================================
def setup_logging(log_dir: str = "outputs/logs"):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# =============================================================
# TRAINING DATA PIPELINE (NumPy-based)
# =============================================================
class TrainingDataPipeline:
    """Pipeline untuk menyiapkan data training menggunakan NumPy."""

    def __init__(self, random_seed: int = 42):
        self.seed = random_seed
        np.random.seed(self.seed)
        self.cached_data = {}

    def generate_dataset(
        self,
        n_samples: int = 10000,
        n_features: int = 20,
        n_classes: int = 2,
        task_type: str = "classification",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate dataset untuk training.

        Args:
            n_samples: Jumlah sampel.
            n_features: Jumlah fitur.
            n_classes: Jumlah kelas.
            task_type: 'classification' atau 'regression'.

        Returns:
            Tuple (X, y).
        """
        X = np.random.randn(n_samples, n_features).astype(np.float32)

        if task_type == "classification":
            # Create linearly separable-ish data
            weights = np.random.randn(n_features, n_classes).astype(np.float32)
            logits = X @ weights
            y = np.argmax(logits, axis=1)
        else:
            weights = np.random.randn(n_features, 1).astype(np.float32)
            y = (X @ weights).flatten() + 0.1 * np.random.randn(n_samples)

        logger.info(f"Generated dataset: X={X.shape}, y={y.shape} ({task_type})")
        return X, y

    def create_tf_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """
        Convert NumPy arrays ke TensorFlow Dataset.

        Args:
            X: Features.
            y: Labels.
            batch_size: Batch size.
            shuffle: Shuffle dataset.

        Returns:
            TensorFlow Dataset object.
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        logger.info(f"Created TF Dataset: batch_size={batch_size}, samples={len(X)}")
        return dataset

    def split_and_prepare(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_split: float = 0.2,
        batch_size: int = 32,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict]:
        """
        Split data dan convert ke TF Datasets.

        Returns:
            Tuple (train_dataset, val_dataset, info).
        """
        n_val = int(len(X) * val_split)
        indices = np.random.permutation(len(X))

        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_ds = self.create_tf_dataset(X_train, y_train, batch_size, shuffle=True)
        val_ds = self.create_tf_dataset(X_val, y_val, batch_size, shuffle=False)

        info = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)) if len(y.shape) == 1 else y.shape[1],
            "batch_size": batch_size,
        }

        logger.info(f"Data split: train={info['train_samples']}, val={info['val_samples']}")
        return train_ds, val_ds, info


# =============================================================
# MODEL BUILDER
# =============================================================
class ModelBuilder:
    """Builder untuk berbagai jenis model ML."""

    @staticmethod
    def build_classifier(
        input_dim: int,
        n_classes: int = 2,
        hidden_units: List[int] = None,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
    ) -> keras.Model:
        """
        Build classification model.

        Args:
            input_dim: Dimensi input.
            n_classes: Jumlah kelas output.
            hidden_units: List of hidden layer sizes.
            dropout_rate: Dropout rate.
            learning_rate: Learning rate.

        Returns:
            Compiled Keras Model.
        """
        if hidden_units is None:
            hidden_units = [128, 64, 32]

        inputs = layers.Input(shape=(input_dim,), name="input")
        x = inputs

        for units in hidden_units:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)

        if n_classes == 2:
            outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
            loss_fn = "binary_crossentropy"
        else:
            outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)
            loss_fn = "sparse_categorical_crossentropy"

        model = keras.Model(inputs=inputs, outputs=outputs, name="Classifier")

        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=["accuracy"],
        )

        logger.info(f"Built classifier: input={input_dim}, classes={n_classes}, "
                     f"params={model.count_params():,}")
        return model

    @staticmethod
    def build_regressor(
        input_dim: int,
        hidden_units: List[int] = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ) -> keras.Model:
        """
        Build regression model.

        Args:
            input_dim: Dimensi input.
            hidden_units: List of hidden layer sizes.
            dropout_rate: Dropout rate.
            learning_rate: Learning rate.

        Returns:
            Compiled Keras Model.
        """
        if hidden_units is None:
            hidden_units = [128, 64, 32]

        inputs = layers.Input(shape=(input_dim,), name="input")
        x = inputs

        for units in hidden_units:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)

        outputs = layers.Dense(1, activation="linear", name="output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="Regressor")

        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        logger.info(f"Built regressor: input={input_dim}, params={model.count_params():,}")
        return model


# =============================================================
# TRAINING SCHEDULER
# =============================================================
class TrainingScheduler:
    """
    Training Scheduler - Jadwalkan training model pada waktu tertentu.

    Mendukung:
      - Training pada jam tertentu (o'clock scheduling)
      - Interval-based training
      - Cron-like patterns
      - Pre/post training callbacks
      - Checkpointing otomatis
    """

    def __init__(
        self,
        checkpoint_dir: str = "models/checkpoints",
        output_dir: str = "outputs",
    ):
        """
        Inisialisasi Training Scheduler.

        Args:
            checkpoint_dir: Directory untuk model checkpoints.
            output_dir: Directory untuk output files.
        """
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.scheduled_jobs = {}
        self.training_history = []
        self.callbacks_registry = {}
        self.data_pipeline = TrainingDataPipeline()
        self.model_builder = ModelBuilder()

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

        # APScheduler instance
        self.apscheduler = None
        if HAS_APSCHEDULER:
            self.apscheduler = BackgroundScheduler()

        logger.info("Training Scheduler initialized")

    # =========================================================
    # SCHEDULING METHODS
    # =========================================================

    def schedule_at_o_clock(
        self,
        hour: int,
        minute: int = 0,
        training_config: Optional[Dict] = None,
        job_name: Optional[str] = None,
    ) -> str:
        """
        Jadwalkan training pada jam tertentu (o'clock).

        Args:
            hour: Jam (0-23).
            minute: Menit (0-59).
            training_config: Konfigurasi training.
            job_name: Nama job (opsional).

        Returns:
            Job ID.
        """
        if training_config is None:
            training_config = self._default_training_config()

        if job_name is None:
            job_name = f"training_{hour:02d}:{minute:02d}"

        # Create callback to run training at the scheduled time
        def run_training():
            self._execute_training(job_name, training_config)

        if HAS_SCHEDULE:
            # Using schedule library
            time_str = f"{hour:02d}:{minute:02d}"
            job = schedule.every().day.at(time_str).do(run_training)
            self.scheduled_jobs[job_name] = {"type": "schedule", "job": job, "time": time_str}
            logger.info(f"Scheduled '{job_name}' at {time_str} daily")
        elif HAS_APSCHEDULER:
            # Using APScheduler
            self.apscheduler.add_job(
                run_training,
                "cron",
                hour=hour,
                minute=minute,
                id=job_name,
                replace_existing=True,
            )
            self.scheduled_jobs[job_name] = {"type": "apscheduler", "hour": hour, "minute": minute}
            logger.info(f"Scheduled '{job_name}' at {hour:02d}:{minute:02d} daily (APScheduler)")
        else:
            logger.warning("No scheduler library available. Training scheduled in-memory only.")
            self.scheduled_jobs[job_name] = {"type": "manual", "hour": hour, "minute": minute}

        return job_name

    def schedule_interval(
        self,
        hours: float = 1.0,
        training_config: Optional[Dict] = None,
        job_name: Optional[str] = None,
    ) -> str:
        """
        Jadwalkan training dengan interval tertentu.

        Args:
            hours: Interval dalam jam.
            training_config: Konfigurasi training.
            job_name: Nama job.

        Returns:
            Job ID.
        """
        if training_config is None:
            training_config = self._default_training_config()

        if job_name is None:
            job_name = f"interval_{hours}h"

        def run_training():
            self._execute_training(job_name, training_config)

        if HAS_APSCHEDULER:
            self.apscheduler.add_job(
                run_training,
                "interval",
                hours=hours,
                id=job_name,
                replace_existing=True,
            )
            self.scheduled_jobs[job_name] = {"type": "apscheduler", "interval_hours": hours}
            logger.info(f"Scheduled '{job_name}' every {hours} hours")
        elif HAS_SCHEDULE:
            # Approximate with schedule
            job = schedule.every(int(hours * 60)).minutes.do(run_training)
            self.scheduled_jobs[job_name] = {"type": "schedule", "job": job, "interval": f"{hours}h"}
            logger.info(f"Scheduled '{job_name}' every {hours} hours (schedule lib)")
        else:
            self.scheduled_jobs[job_name] = {"type": "manual", "interval_hours": hours}

        return job_name

    def schedule_multiple_o_clock(
        self,
        times: List[str],
        training_config: Optional[Dict] = None,
    ) -> List[str]:
        """
        Jadwalkan training di multiple waktu o'clock.

        Args:
            times: List of time strings (e.g., ["06:00", "12:00", "18:00"]).
            training_config: Konfigurasi training.

        Returns:
            List of job IDs.
        """
        job_ids = []
        for t in times:
            parts = t.split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
            job_id = self.schedule_at_o_clock(hour, minute, training_config, f"training_{t.replace(':', '')}")
            job_ids.append(job_id)

        logger.info(f"Scheduled {len(job_ids)} training jobs at: {times}")
        return job_ids

    def schedule_every_hour(
        self,
        training_config: Optional[Dict] = None,
        minute_offset: int = 0,
    ) -> str:
        """
        Jadwalkan training setiap jam pada menit tertentu.

        Args:
            training_config: Konfigurasi training.
            minute_offset: Menit dalam jam (default: 0 = tepat o'clock).

        Returns:
            Job ID.
        """
        job_name = f"hourly_at_{minute_offset:02d}"

        if HAS_APSCHEDULER:
            self.apscheduler.add_job(
                lambda: self._execute_training(job_name, training_config or self._default_training_config()),
                "cron",
                minute=minute_offset,
                id=job_name,
                replace_existing=True,
            )
            self.scheduled_jobs[job_name] = {"type": "apscheduler", "every_hour": True, "minute": minute_offset}
            logger.info(f"Scheduled '{job_name}' every hour at :{minute_offset:02d}")
        else:
            self.schedule_interval(hours=1.0, training_config=training_config, job_name=job_name)

        return job_name

    # =========================================================
    # EXECUTION
    # =========================================================

    def run_now(
        self,
        training_config: Optional[Dict] = None,
        job_name: Optional[str] = None,
    ) -> Dict:
        """
        Jalankan training sekarang.

        Args:
            training_config: Konfigurasi training.
            job_name: Nama job.

        Returns:
            Training results.
        """
        if job_name is None:
            job_name = f"manual_{datetime.now().strftime('%H%M%S')}"

        if training_config is None:
            training_config = self._default_training_config()

        return self._execute_training(job_name, training_config)

    def _execute_training(self, job_name: str, config: Dict) -> Dict:
        """
        Execute training job.

        Args:
            job_name: Nama training job.
            config: Konfigurasi training.

        Returns:
            Training results dictionary.
        """
        start_time = time.time()
        logger.info(f"{'='*50}")
        logger.info(f"Starting training job: {job_name}")
        logger.info(f"Configuration: {json.dumps(config, indent=2, default=str)}")

        try:
            # Pre-training callback
            self._run_callbacks("pre_training", job_name, config)

            # 1. Prepare data
            logger.info("[Step 1/5] Preparing data...")
            X, y = self.data_pipeline.generate_dataset(
                n_samples=config.get("n_samples", 10000),
                n_features=config.get("n_features", 20),
                n_classes=config.get("n_classes", 2),
                task_type=config.get("task_type", "classification"),
            )

            train_ds, val_ds, data_info = self.data_pipeline.split_and_prepare(
                X, y,
                val_split=config.get("val_split", 0.2),
                batch_size=config.get("batch_size", 32),
            )

            # 2. Build model
            logger.info("[Step 2/5] Building model...")
            if config.get("task_type", "classification") == "classification":
                model = self.model_builder.build_classifier(
                    input_dim=data_info["n_features"],
                    n_classes=data_info["n_classes"],
                    hidden_units=config.get("hidden_units", [128, 64, 32]),
                    dropout_rate=config.get("dropout_rate", 0.3),
                    learning_rate=config.get("learning_rate", 0.001),
                )
            else:
                model = self.model_builder.build_regressor(
                    input_dim=data_info["n_features"],
                    hidden_units=config.get("hidden_units", [128, 64, 32]),
                    dropout_rate=config.get("dropout_rate", 0.2),
                    learning_rate=config.get("learning_rate", 0.001),
                )

            # 3. Setup callbacks
            logger.info("[Step 3/5] Setting up callbacks...")
            cb_list = self._create_tf_callbacks(job_name, config)

            # 4. Train
            logger.info("[Step 4/5] Training model...")
            epochs = config.get("epochs", 10)
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=cb_list,
                verbose=1,
            )

            # 5. Save results
            logger.info("[Step 5/5] Saving results...")
            model_path = os.path.join(self.checkpoint_dir, f"{job_name}_model.keras")
            model.save(model_path)

            training_time = time.time() - start_time

            # Extract final metrics
            final_metrics = {}
            for key in history.history:
                final_metrics[key] = float(history.history[key][-1])

            result = {
                "job_name": job_name,
                "timestamp": datetime.now().isoformat(),
                "training_time_seconds": round(training_time, 2),
                "epochs_completed": len(history.history["loss"]),
                "final_metrics": final_metrics,
                "data_info": data_info,
                "model_path": model_path,
                "config": config,
                "status": "success",
            }

            # Post-training callback
            self._run_callbacks("post_training", job_name, result)

            # Save metrics
            self._save_metrics(result)

            # Track history
            self.training_history.append(result)

            logger.info(f"Training completed in {training_time:.2f}s")
            logger.info(f"Final metrics: {final_metrics}")
            logger.info(f"Model saved to: {model_path}")

            return result

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            error_result = {
                "job_name": job_name,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e),
            }
            self._run_callbacks("on_error", job_name, error_result)
            return error_result

    # =========================================================
    # CALLBACKS
    # =========================================================

    def register_callback(
        self,
        event: str,
        callback_fn: Callable,
        name: Optional[str] = None,
    ) -> None:
        """
        Register callback function untuk event tertentu.

        Events: 'pre_training', 'post_training', 'on_error'

        Args:
            event: Nama event.
            callback_fn: Callback function.
            name: Nama callback (opsional).
        """
        if event not in self.callbacks_registry:
            self.callbacks_registry[event] = []

        cb_entry = {
            "name": name or callback_fn.__name__,
            "function": callback_fn,
        }
        self.callbacks_registry[event].append(cb_entry)
        logger.info(f"Registered callback '{cb_entry['name']}' for event '{event}'")

    def _run_callbacks(self, event: str, *args) -> None:
        """Run semua callbacks untuk event tertentu."""
        for cb_entry in self.callbacks_registry.get(event, []):
            try:
                logger.info(f"  Running callback: {cb_entry['name']}")
                cb_entry["function"](*args)
            except Exception as e:
                logger.error(f"  Callback '{cb_entry['name']}' failed: {e}")

    def _create_tf_callbacks(self, job_name: str, config: Dict) -> List:
        """Create TensorFlow training callbacks."""
        cb_list = []

        # Early stopping
        if config.get("early_stopping", True):
            cb_list.append(
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=config.get("patience", 5),
                    restore_best_weights=True,
                )
            )

        # Model checkpoint
        cb_list.append(
            callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    self.checkpoint_dir, f"{job_name}_best.keras"
                ),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            )
        )

        # Reduce LR on plateau
        cb_list.append(
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            )
        )

        # CSV Logger
        cb_list.append(
            callbacks.CSVLogger(
                os.path.join(self.output_dir, "logs", f"{job_name}_log.csv")
            )
        )

        return cb_list

    # =========================================================
    # SCHEDULER CONTROL
    # =========================================================

    def start(self) -> None:
        """Start the scheduler."""
        if HAS_APSCHEDULER and not self.apscheduler.running:
            self.apscheduler.start()
            logger.info("APScheduler started")

        if HAS_SCHEDULE:
            def run_schedule():
                while True:
                    schedule.run_pending()
                    time.sleep(1)

            thread = threading.Thread(target=run_schedule, daemon=True)
            thread.start()
            logger.info("Schedule library thread started")

        logger.info("Training scheduler is RUNNING")
        self._print_schedule()

    def stop(self) -> None:
        """Stop the scheduler."""
        if HAS_APSCHEDULER and self.apscheduler.running:
            self.apscheduler.shutdown()
            logger.info("APScheduler stopped")

        logger.info("Training scheduler STOPPED")

    def _print_schedule(self) -> None:
        """Print semua scheduled jobs."""
        logger.info(f"Active jobs ({len(self.scheduled_jobs)}):")
        for name, info in self.scheduled_jobs.items():
            logger.info(f"  - {name}: {info}")

    # =========================================================
    # UTILITIES
    # =========================================================

    def _default_training_config(self) -> Dict:
        """Return default training configuration."""
        return {
            "n_samples": 10000,
            "n_features": 20,
            "n_classes": 2,
            "task_type": "classification",
            "epochs": 10,
            "batch_size": 32,
            "val_split": 0.2,
            "learning_rate": 0.001,
            "hidden_units": [128, 64, 32],
            "dropout_rate": 0.3,
            "early_stopping": True,
            "patience": 5,
        }

    def _save_metrics(self, result: Dict) -> None:
        """Save training metrics ke file."""
        metrics_file = os.path.join(self.output_dir, "metrics", "training_metrics.json")

        # Load existing or create new
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []

        all_metrics.append(result)

        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)

        logger.info(f"Metrics saved to {metrics_file}")

    def get_training_history(self) -> List[Dict]:
        """Return semua training history."""
        return self.training_history

    def get_next_run_times(self) -> Dict[str, str]:
        """
        Get estimated next run time untuk setiap job.

        Returns:
            Dictionary {job_name: next_run_time_str}.
        """
        now = datetime.now()
        next_times = {}

        for name, info in self.scheduled_jobs.items():
            if info.get("type") == "apscheduler":
                if "hour" in info and "minute" in info:
                    next_run = now.replace(
                        hour=info["hour"],
                        minute=info["minute"],
                        second=0,
                        microsecond=0,
                    )
                    if next_run <= now:
                        next_run += timedelta(days=1)
                    next_times[name] = next_run.strftime("%Y-%m-%d %H:%M:%S")
                elif info.get("every_hour"):
                    minute = info.get("minute", 0)
                    next_hour = now.replace(minute=minute, second=0, microsecond=0)
                    if next_hour <= now:
                        next_hour += timedelta(hours=1)
                    next_times[name] = next_hour.strftime("%Y-%m-%d %H:%M:%S")
                elif "interval_hours" in info:
                    next_run = now + timedelta(hours=info["interval_hours"])
                    next_times[name] = next_run.strftime("%Y-%m-%d %H:%M:%S")

        return next_times

    def list_jobs(self) -> List[Dict]:
        """List semua scheduled jobs."""
        return [
            {"name": name, **info}
            for name, info in self.scheduled_jobs.items()
        ]


# =============================================================
# DEMO / MAIN
# =============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  TRAINING SCHEDULER - DEMO (Train with O'Clock)")
    print("=" * 60)

    scheduler = TrainingScheduler(
        checkpoint_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "checkpoints"),
        output_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs"),
    )

    # 1. Custom callbacks
    def on_training_start(job_name, config):
        print(f"\n  >>> CUSTOM CALLBACK: Training '{job_name}' is starting!")
        print(f"  >>> Config: {config.get('epochs', 10)} epochs")

    def on_training_complete(job_name, result):
        print(f"\n  >>> CUSTOM CALLBACK: Training '{job_name}' completed!")
        print(f"  >>> Status: {result.get('status', 'unknown')}")
        print(f"  >>> Time: {result.get('training_time_seconds', 0):.2f}s")

    scheduler.register_callback("pre_training", on_training_start, "start_notifier")
    scheduler.register_callback("post_training", on_training_complete, "complete_notifier")

    # 2. Schedule training at specific o'clock times
    print("\n--- Scheduling Training Jobs ---")
    scheduler.schedule_at_o_clock(hour=6, minute=0, job_name="morning_training")
    scheduler.schedule_at_o_clock(hour=12, minute=30, job_name="noon_training")
    scheduler.schedule_at_o_clock(hour=18, minute=0, job_name="evening_training")
    scheduler.schedule_at_o_clock(hour=22, minute=0, job_name="night_training")

    # 3. Schedule multiple o'clock times at once
    print("\n--- Schedule Multiple Times ---")
    scheduler.schedule_multiple_o_clock(
        times=["03:00", "09:00", "15:00", "21:00"],
        training_config={"epochs": 5, "n_samples": 5000, "task_type": "classification"},
    )

    # 4. Schedule every hour at :00 (o'clock)
    print("\n--- Schedule Every O'Clock ---")
    scheduler.schedule_every_hour(minute_offset=0, training_config={"epochs": 3, "n_samples": 2000})

    # 5. List all jobs
    print("\n--- All Scheduled Jobs ---")
    jobs = scheduler.list_jobs()
    for job in jobs:
        print(f"  - {job['name']}: {job}")

    # 6. Show next run times
    print("\n--- Next Run Times ---")
    next_times = scheduler.get_next_run_times()
    for name, next_time in next_times.items():
        print(f"  - {name}: {next_time}")

    # 7. Run training NOW (demo)
    print("\n--- Running Training NOW ---")
    result = scheduler.run_now(
        training_config={
            "n_samples": 5000,
            "n_features": 20,
            "n_classes": 3,
            "task_type": "classification",
            "epochs": 10,
            "batch_size": 64,
            "hidden_units": [128, 64, 32],
        },
        job_name="demo_training",
    )

    print(f"\n--- Training Result ---")
    print(f"  Status: {result.get('status')}")
    print(f"  Time: {result.get('training_time_seconds', 0):.2f}s")
    print(f"  Epochs: {result.get('epochs_completed')}")
    if "final_metrics" in result:
        for key, val in result["final_metrics"].items():
            print(f"  {key}: {val:.4f}")

    # Note: To start the actual scheduler daemon, call:
    # scheduler.start()  # This will run in background
    # scheduler.stop()   # To stop it

    print("\n" + "=" * 60)
    print("  TRAINING SCHEDULER DEMO COMPLETED!")
    print("  (Scheduler jobs configured but daemon not started)")
    print("=" * 60)
