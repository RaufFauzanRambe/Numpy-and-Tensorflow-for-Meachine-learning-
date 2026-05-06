"""
============================================================
NumPy Automation Module
============================================================
Modul ini menyediakan berbagai fungsi otomasi menggunakan NumPy
untuk pengolahan data, analisis statistik, dan transformasi.

Author      : ML Project Team
Description : NumPy-based automation for data processing,
              statistical analysis, and feature engineering.
============================================================
"""

import numpy as np
import os
import json
from datetime import datetime
from typing import Tuple, List, Dict, Optional


class NumPyAutomation:
    """Class untuk otomasi pengolahan data menggunakan NumPy."""

    def __init__(self, random_seed: int = 42):
        """
        Inisialisasi NumPy Automation.

        Args:
            random_seed: Seed untuk reproducibility.
        """
        self.seed = random_seed
        np.random.seed(self.seed)
        self.history = []

    # =========================================================
    # 1. DATA GENERATION & LOADING
    # =========================================================

    def generate_synthetic_data(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        noise_level: float = 0.1,
        task_type: str = "regression"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data sintetis untuk training/testing.

        Args:
            n_samples: Jumlah sampel.
            n_features: Jumlah fitur.
            noise_level: Tingkat noise pada data.
            task_type: 'regression', 'classification', atau 'multiclass'.

        Returns:
            Tuple (X, y) dengan fitur dan label.
        """
        X = np.random.randn(n_samples, n_features)

        if task_type == "regression":
            weights = np.random.randn(n_features, 1)
            y = X @ weights + noise_level * np.random.randn(n_samples, 1)
        elif task_type == "classification":
            linear = X @ np.random.randn(n_features, 1)
            y = (linear.flatten() > 0).astype(int)
        elif task_type == "multiclass":
            linear = X @ np.random.randn(n_features, 3)
            y = np.argmax(linear, axis=1)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        log_entry = {
            "action": "generate_synthetic_data",
            "timestamp": datetime.now().isoformat(),
            "n_samples": n_samples,
            "n_features": n_features,
            "task_type": task_type
        }
        self.history.append(log_entry)
        print(f"[OK] Generated {n_samples} samples x {n_features} features ({task_type})")

        return X, y

    def load_csv_to_numpy(self, filepath: str, delimiter: str = ",") -> np.ndarray:
        """
        Load file CSV ke dalam NumPy array.

        Args:
            filepath: Path ke file CSV.
            delimiter: Delimiter pemisah kolom.

        Returns:
            NumPy array berisi data dari CSV.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        data = np.genfromtxt(filepath, delimiter=delimiter, skip_header=1)
        print(f"[OK] Loaded data from {filepath}: shape {data.shape}")

        log_entry = {
            "action": "load_csv_to_numpy",
            "timestamp": datetime.now().isoformat(),
            "filepath": filepath,
            "shape": list(data.shape)
        }
        self.history.append(log_entry)
        return data

    # =========================================================
    # 2. DATA PREPROCESSING
    # =========================================================

    def normalize(self, X: np.ndarray, method: str = "minmax") -> Tuple[np.ndarray, Dict]:
        """
        Normalisasi data menggunakan berbagai metode.

        Args:
            X: Input array 2D.
            method: 'minmax', 'zscore', atau 'robust'.

        Returns:
            Tuple (X_normalized, params) dengan data yang dinormalisasi
            dan parameter yang digunakan.
        """
        params = {}

        if method == "minmax":
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1  # Avoid division by zero
            X_norm = (X - min_vals) / range_vals
            params = {"method": "minmax", "min": min_vals.tolist(), "max": max_vals.tolist()}

        elif method == "zscore":
            mean_vals = np.mean(X, axis=0)
            std_vals = np.std(X, axis=0)
            std_vals[std_vals == 0] = 1  # Avoid division by zero
            X_norm = (X - mean_vals) / std_vals
            params = {"method": "zscore", "mean": mean_vals.tolist(), "std": std_vals.tolist()}

        elif method == "robust":
            median_vals = np.median(X, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            iqr = q75 - q25
            iqr[iqr == 0] = 1
            X_norm = (X - median_vals) / iqr
            params = {"method": "robust", "median": median_vals.tolist(), "iqr": iqr.tolist()}
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        print(f"[OK] Normalized data using '{method}' method. Shape: {X_norm.shape}")
        return X_norm, params

    def one_hot_encode(self, y: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
        """
        One-hot encoding untuk label kelas.

        Args:
            y: Label array (1D).
            n_classes: Jumlah kelas (opsional, auto-detect jika None).

        Returns:
            One-hot encoded array.
        """
        if n_classes is None:
            n_classes = int(np.max(y)) + 1

        one_hot = np.zeros((len(y), n_classes))
        one_hot[np.arange(len(y)), y.astype(int)] = 1
        print(f"[OK] One-hot encoded {len(y)} labels into {n_classes} classes")
        return one_hot

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        shuffle: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Split data menjadi train, validation, dan test sets.

        Args:
            X: Feature array.
            y: Label array.
            test_ratio: Proporsi test set.
            val_ratio: Proporsi validation set (dari training).
            shuffle: Apakah data di-shuffle.

        Returns:
            Dictionary dengan keys: X_train, X_val, X_test, y_train, y_val, y_test
        """
        n = len(X)
        indices = np.arange(n)

        if shuffle:
            np.random.shuffle(indices)

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        test_size = int(n * test_ratio)
        train_size = n - test_size
        val_size = int(train_size * val_ratio)

        splits = {
            "X_train": X_shuffled[: train_size - val_size],
            "X_val": X_shuffled[train_size - val_size: train_size],
            "X_test": X_shuffled[train_size:],
            "y_train": y_shuffled[: train_size - val_size],
            "y_val": y_shuffled[train_size - val_size: train_size],
            "y_test": y_shuffled[train_size:],
        }

        print(f"[OK] Data split -> Train: {len(splits['X_train'])}, "
              f"Val: {len(splits['X_val'])}, Test: {len(splits['X_test'])}")
        return splits

    # =========================================================
    # 3. FEATURE ENGINEERING
    # =========================================================

    def create_polynomial_features(
        self, X: np.ndarray, degree: int = 2
    ) -> np.ndarray:
        """
        Buat polynomial features dari input data.

        Args:
            X: Input array 2D.
            degree: Derajat polinomial.

        Returns:
            Array dengan polynomial features.
        """
        n_samples, n_features = X.shape
        features = [X]

        for d in range(2, degree + 1):
            poly_features = np.power(X, d)
            features.append(poly_features)

        # Add interaction terms for degree 2+
        if degree >= 2:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                    features.append(interaction)

        result = np.hstack(features)
        print(f"[OK] Created polynomial features (degree={degree}): {X.shape} -> {result.shape}")
        return result

    def extract_statistical_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract statistical features dari setiap row data.

        Args:
            X: Input array 2D.

        Returns:
            Array dengan additional statistical features per row.
        """
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        min_val = np.min(X, axis=1, keepdims=True)
        max_val = np.max(X, axis=1, keepdims=True)
        median = np.median(X, axis=1, keepdims=True)
        skew = self._calculate_skew(X)

        features = np.hstack([mean, std, min_val, max_val, median, skew])
        print(f"[OK] Extracted 6 statistical features per sample: {features.shape}")
        return features

    @staticmethod
    def _calculate_skew(X: np.ndarray) -> np.ndarray:
        """Calculate skewness for each row."""
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        std[std == 0] = 1
        n = X.shape[1]
        skew = np.mean(((X - mean) / std) ** 3, axis=1, keepdims=True)
        return skew

    # =========================================================
    # 4. DATA ANALYSIS & STATISTICS
    # =========================================================

    def compute_statistics(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Hitung statistik deskriptif dari data.

        Args:
            X: Input array 2D.

        Returns:
            Dictionary dengan berbagai statistik.
        """
        stats = {
            "mean": np.mean(X, axis=0),
            "std": np.std(X, axis=0),
            "median": np.median(X, axis=0),
            "min": np.min(X, axis=0),
            "max": np.max(X, axis=0),
            "q25": np.percentile(X, 25, axis=0),
            "q75": np.percentile(X, 75, axis=0),
            "skewness": self._calculate_skew(X).flatten(),
            "variance": np.var(X, axis=0),
            "range": np.ptp(X, axis=0),
        }

        print(f"[OK] Computed statistics for data with shape {X.shape}")
        return stats

    def compute_correlation_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Hitung correlation matrix dari data.

        Args:
            X: Input array 2D.

        Returns:
            Correlation matrix (n_features x n_features).
        """
        # Standardize
        X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        corr_matrix = (X_std.T @ X_std) / len(X)
        print(f"[OK] Computed correlation matrix: {corr_matrix.shape}")
        return corr_matrix

    def detect_outliers_iqr(
        self, X: np.ndarray, threshold: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Deteksi outlier menggunakan IQR method.

        Args:
            X: Input array 2D.
            threshold: IQR multiplier untuk outlier detection.

        Returns:
            Tuple (outlier_mask, outlier_indices).
        """
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        iqr = q75 - q25

        lower_bound = q25 - threshold * iqr
        upper_bound = q75 + threshold * iqr

        outlier_mask = np.any((X < lower_bound) | (X > upper_bound), axis=1)
        outlier_indices = np.where(outlier_mask)[0]

        print(f"[OK] Detected {len(outlier_indices)} outliers out of {len(X)} samples "
              f"({100 * len(outlier_indices) / len(X):.1f}%)")
        return outlier_mask, outlier_indices

    # =========================================================
    # 5. MATRIX OPERATIONS FOR ML
    # =========================================================

    def compute_pca_manual(
        self, X: np.ndarray, n_components: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        PCA (Principal Component Analysis) manual menggunakan NumPy.

        Args:
            X: Input array 2D (n_samples, n_features).
            n_components: Jumlah principal components.

        Returns:
            Tuple (X_projected, eigenvalues, eigenvectors).
        """
        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # Compute covariance matrix
        cov_matrix = (X_centered.T @ X_centered) / (len(X) - 1)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by descending eigenvalue
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Project data
        components = eigenvectors[:, :n_components]
        X_projected = X_centered @ components

        explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
        print(f"[OK] PCA: {X.shape} -> {X_projected.shape} "
              f"(explained variance: {explained_variance_ratio})")

        return X_projected, eigenvalues, eigenvectors

    def compute_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = None
    ) -> np.ndarray:
        """
        Hitung confusion matrix.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            n_classes: Jumlah kelas.

        Returns:
            Confusion matrix (n_classes x n_classes).
        """
        if n_classes is None:
            n_classes = max(int(np.max(y_true)), int(np.max(y_pred))) + 1

        matrix = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true.astype(int), y_pred.astype(int)):
            matrix[t, p] += 1

        print(f"[OK] Confusion matrix ({n_classes}x{n_classes}):")
        print(matrix)
        return matrix

    # =========================================================
    # 6. BATCH PROCESSING AUTOMATION
    # =========================================================

    def batch_process(
        self,
        data_list: List[np.ndarray],
        operation: str = "normalize",
        **kwargs
    ) -> List[np.ndarray]:
        """
        Proses batch multiple arrays secara otomatis.

        Args:
            data_list: List of NumPy arrays.
            operation: Operasi yang akan diterapkan.
            **kwargs: Parameter tambahan untuk operasi.

        Returns:
            List of processed arrays.
        """
        results = []
        for i, data in enumerate(data_list):
            print(f"  Processing batch {i + 1}/{len(data_list)}...")

            if operation == "normalize":
                processed, _ = self.normalize(data, **kwargs)
            elif operation == "polynomial":
                processed = self.create_polynomial_features(data, **kwargs)
            elif operation == "stats":
                processed = self.extract_statistical_features(data)
            elif operation == "pca":
                processed, _, _ = self.compute_pca_manual(data, **kwargs)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            results.append(processed)

        print(f"[OK] Batch processed {len(data_list)} arrays")
        return results

    # =========================================================
    # 7. UTILITY METHODS
    # =========================================================

    def save_array(self, data: np.ndarray, filepath: str) -> None:
        """Save NumPy array ke file .npy."""
        np.save(filepath, data)
        print(f"[OK] Saved array {data.shape} to {filepath}")

    def load_array(self, filepath: str) -> np.ndarray:
        """Load NumPy array dari file .npy."""
        data = np.load(filepath)
        print(f"[OK] Loaded array {data.shape} from {filepath}")
        return data

    def get_history(self) -> List[Dict]:
        """Return history log dari semua operasi."""
        return self.history

    def clear_history(self) -> None:
        """Clear history log."""
        self.history = []
        print("[OK] History cleared")


# =============================================================
# DEMO / MAIN
# =============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  NUMPY AUTOMATION - DEMO")
    print("=" * 60)

    auto = NumPyAutomation(random_seed=42)

    # 1. Generate synthetic data
    print("\n--- Generating Data ---")
    X, y = auto.generate_synthetic_data(n_samples=1000, n_features=10, task_type="classification")

    # 2. Normalize
    print("\n--- Normalizing ---")
    X_norm, params = auto.normalize(X, method="zscore")

    # 3. Split data
    print("\n--- Splitting Data ---")
    splits = auto.split_data(X_norm, y, test_ratio=0.2, val_ratio=0.1)

    # 4. Extract features
    print("\n--- Feature Engineering ---")
    poly_features = auto.create_polynomial_features(splits["X_train"], degree=2)
    stat_features = auto.extract_statistical_features(splits["X_train"])

    # 5. Statistics
    print("\n--- Statistics ---")
    stats = auto.compute_statistics(splits["X_train"])
    corr = auto.compute_correlation_matrix(splits["X_train"])

    # 6. PCA
    print("\n--- PCA ---")
    X_pca, eigenvalues, eigenvectors = auto.compute_pca_manual(splits["X_train"], n_components=2)

    # 7. Outlier detection
    print("\n--- Outlier Detection ---")
    outlier_mask, outlier_idx = auto.detect_outliers_iqr(splits["X_train"])

    print("\n" + "=" * 60)
    print("  ALL AUTOMATION TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
