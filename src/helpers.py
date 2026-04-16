import numpy as np
from typing import Optional, Union, Literal, Dict, Any
import warnings
from dataclasses import dataclass


@dataclass
class ArrayInfo:
    """Data class to store array information."""
    shape: tuple
    dtype: str
    size: int
    ndim: int
    memory_bytes: int
    is_empty: bool
    has_nan: bool
    has_inf: bool


def print_shape(
    array: Union[np.ndarray, list, tuple],
    name: str = "Array",
    verbose: bool = False,
    return_info: bool = False,
    use_logger: bool = False
) -> Optional[Union[Dict[str, Any], ArrayInfo]]:
    """
    Print detailed information about NumPy array shape and properties.
    
    Parameters
    ----------
    array : np.ndarray or array-like
        Input array to inspect
    name : str, default="Array"
        Label name for output display
    verbose : bool, default=False
        If True, show additional info (dtype, size, memory usage)
    return_info : bool, default=False
        If True, return dictionary/info object instead of printing
    use_logger : bool, default=False
        Use logging module instead of print
    
    Returns
    -------
    dict or ArrayInfo or None
        Array information if return_info=True
    
    Raises
    ------
    TypeError
        If input is not array-like
    
    Examples
    --------
    >>> arr = np.random.rand(3, 4, 5)
    >>> print_shape(arr, "Data", verbose=True)
    Data shape: (3, 4, 5)
    Data dtype: float64
    Data size: 60 elements
    Data ndim: 3
    Data memory: 480 bytes (0.00 MB)
    Data has NaN: False
    Data has Inf: False
    """
    
    # ===== INPUT VALIDATION =====
    if not isinstance(array, (np.ndarray, list, tuple)):
        raise TypeError(f"Expected array-like, got {type(array).__name__}")
    
    # Convert to numpy array if needed
    arr = np.asarray(array) if not isinstance(array, np.ndarray) else array
    
    # ===== CALCULATE INFORMATION =====
    info = ArrayInfo(
        shape=arr.shape,
        dtype=str(arr.dtype),
        size=arr.size,
        ndim=arr.ndim,
        memory_bytes=arr.nbytes,
        is_empty=(arr.size == 0),
        has_nan=bool(np.isnan(arr).any()) if np.issubdtype(arr.dtype, np.floating) else False,
        has_inf=bool(np.isinf(arr).any()) if np.issubdtype(arr.dtype, np.floating) else False
    )
    
    # ===== FORMAT OUTPUT =====
    if return_info:
        return info
    
    # Choose output method
    output_func = print if not use_logger else lambda msg: __import__('logging').info(msg)
    
    # Basic output
    output_func(f"✅ {name} shape: {info.shape}")
    
    # Verbose output
    if verbose:
        output_func(f"   📊 dtype: {info.dtype}")
        output_func(f"   🔢 size: {info.size:,} elements")
        output_func(f"   📐 ndim: {info.ndim} dimensions")
        
        # Format memory with appropriate units
        mem_mb = info.memory_bytes / (1024 * 1024)
        if mem_mb >= 1:
            output_func(f"   💾 memory: {info.memory_bytes:,} bytes ({mem_mb:.2f} MB)")
        elif info.memory_bytes >= 1024:
            output_func(f"   💾 memory: {info.memory_bytes:,} bytes ({info.memory_bytes/1024:.2f} KB)")
        else:
            output_func(f"   💾 memory: {info.memory_bytes:,} bytes")
        
        output_func(f"   ⚠️  is empty: {info.is_empty}")
        if np.issubdtype(arr.dtype, np.floating):
            output_func(f"   🚫 has NaN: {info.has_nan}")
            output_func(f"   ∞  has Inf: {info.has_inf}")
    
    return None


def normalize(
    data: Union[np.ndarray, list, tuple],
    axis: Optional[int] = None,
    method: Literal['zscore', 'minmax', 'robust', 'l2'] = 'zscore',
    clip: bool = False,
    epsilon: float = 1e-8,
    return_stats: bool = False,
    inplace: bool = False
) -> Union[np.ndarray, tuple]:
    """
    Normalize data using various statistical methods.
    
    Parameters
    ----------
    data : np.ndarray or array-like
        Input data to normalize
    axis : int or None, default=None
        Axis for normalization. None for global normalization.
        For 2D arrays: 0=per column, 1=per row
    method : str, default='zscore'
        Normalization method:
        - 'zscore': Standard score (mean=0, std=1)
        - 'minmax': Min-Max scaling [0, 1]
        - 'robust': Robust scaling (median, IQR) - resistant to outliers ⭐
        - 'l2': L2 normalization (unit vector)
    clip : bool, default=False
        Clip extreme values after normalization (for zscore/minmax/robust only)
    epsilon : float, default=1e-8
        Small value to avoid division by zero
    return_stats : bool, default=False
        If True, return tuple (normalized_data, stats_dict)
    inplace : bool, default=False
        If True, modify original array in-place (only for np.ndarray input)
    
    Returns
    -------
    np.ndarray or tuple
        Normalized data, or (data, stats) if return_stats=True
    
    Raises
    ------
    ValueError
        If method is invalid or data is empty
    TypeError
        If input is not array-like
    
    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 100])  # with outlier
    >>> normalized = normalize(data, method='robust')
    >>> print(normalized)
    [-1.0 -0.5  0.0  0.5  24.0]
    
    >>> # Per-column normalization for 2D data
    >>> X = np.random.rand(100, 5)
    >>> X_norm = normalize(X, axis=0, method='minmax')
    """
    
    # ===== INPUT VALIDATION =====
    if not isinstance(data, (np.ndarray, list, tuple)):
        raise TypeError(f"Expected array-like, got {type(data).__name__}")
    
    # Convert to numpy array
    arr = np.array(data, copy=not inplace) if not isinstance(data, np.ndarray) else \
          (data if inplace else data.copy())
    
    if arr.size == 0:
        raise ValueError("Cannot normalize empty array")
    
    stats = {}
    
    # ===== NORMALIZATION BY METHOD =====
    if method == 'zscore':
        # Z-Score Normalization: (x - mean) / std
        mean = np.mean(arr, axis=axis, keepdims=True)
        std = np.std(arr, axis=axis, keepdims=True)
        
        # Handle zero std (constant values)
        std_safe = np.where(std < epsilon, epsilon, std)
        
        stats = {
            'method': 'zscore',
            'mean': mean.squeeze() if axis is not None else mean.item(),
            'std': std.squeeze() if axis is not None else std.item(),
            'epsilon_used': bool((std < epsilon).any())
        }
        
        result = (arr - mean) / std_safe
        
        if stats['epsilon_used']:
            warnings.warn(
                f"Zero standard deviation detected. Using epsilon={epsilon} to avoid division by zero.",
                RuntimeWarning
            )
    
    elif method == 'minmax':
        # Min-Max Scaling: (x - min) / (max - min) → [0, 1]
        min_val = np.min(arr, axis=axis, keepdims=True)
        max_val = np.max(arr, axis=axis, keepdims=True)
        range_val = max_val - min_val
        
        # Handle constant range
        range_safe = np.where(range_val < epsilon, epsilon, range_val)
        
        stats = {
            'method': 'minmax',
            'min': min_val.squeeze() if axis is not None else min_val.item(),
            'max': max_val.squeeze() if axis is not None else max_val.item(),
            'range': range_val.squeeze() if axis is not None else range_val.item()
        }
        
        result = (arr - min_val) / range_safe
        
        if (range_val < epsilon).any():
            warnings.warn("Constant values detected (min=max). Result may contain artifacts.", RuntimeWarning)
    
    elif method == 'robust':
        # Robust Scaling: (x - median) / IQR
        median = np.median(arr, axis=axis, keepdims=True)
        q75 = np.quantile(arr, 0.75, axis=axis, keepdims=True)
        q25 = np.quantile(arr, 0.25, axis=axis, keepdims=True)
        iqr = q75 - q25
        
        # Handle zero IQR
        iqr_safe = np.where(iqr < epsilon, epsilon, iqr)
        
        stats = {
            'method': 'robust',
            'median': median.squeeze() if axis is not None else median.item(),
            'q25': q25.squeeze() if axis is not None else q25.item(),
            'q75': q75.squeeze() if axis is not None else q75.item(),
            'iqr': iqr.squeeze() if axis is not None else iqr.item()
        }
        
        result = (arr - median) / iqr_safe
    
    elif method == 'l2':
        # L2 Normalization: x / ||x||_2
        norm = np.linalg.norm(arr, axis=axis, keepdims=True)
        norm_safe = np.where(norm < epsilon, epsilon, norm)
        
        stats = {
            'method': 'l2',
            'norm': norm.squeeze() if axis is not None else norm.item()
        }
        
        result = arr / norm_safe
        
        if (norm < epsilon).any():
            warnings.warn("Zero-norm vectors detected.", RuntimeWarning)
    
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: 'zscore', 'minmax', 'robust', 'l2'")
    
    # ===== CLIPPING (optional) =====
    if clip and method != 'l2':
        if method == 'minmax':
            result = np.clip(result, 0, 1)
        elif method in ['zscore', 'robust']:
            # Clip extreme outliers (> 3 standard deviations or similar)
            result = np.clip(result, -10, 10)
    
    # ===== RETURN =====
    if return_stats:
        return result, stats
    return result


# ===== ADDITIONAL UTILITY FUNCTIONS =====

def denormalize(
    normalized_data: np.ndarray,
    stats: Dict[str, Any],
    method: str = 'zscore'
) -> np.ndarray:
    """
    Reverse normalization to restore original scale.
    
    Parameters
    ----------
    normalized_data : np.ndarray
        Normalized data
    stats : dict
        Statistics from normalize() function (with return_stats=True)
    method : str
        Normalization method that was used originally
    
    Returns
    -------
    np.ndarray
        Data in original scale
    """
    if method == 'zscore':
        return normalized_data * stats['std'] + stats['mean']
    elif method == 'minmax':
        return normalized_data * stats['range'] + stats['min']
    elif method == 'robust':
        return normalized_data * stats['iqr'] + stats['median']
    elif method == 'l2':
        raise ValueError("L2 normalization is not reversible without original data")
    else:
        raise ValueError(f"Unknown method: {method}")


def compare_normalization_methods(
    data: np.ndarray,
    methods: list = ['zscore', 'minmax', 'robust'],
    show_plots: bool = False
) -> Dict[str, np.ndarray]:
    """
    Compare results from multiple normalization methods.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    methods : list
        List of methods to compare
    show_plots : bool
        Display comparison plots (requires matplotlib)
    
    Returns
    -------
    dict
        Dictionary with key=method name, value=normalized data
    """
    results = {}
    
    for method in methods:
        try:
            results[method] = normalize(data, method=method)
        except Exception as e:
            print(f"⚠️ Method '{method}' failed: {e}")
    
    if show_plots:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, len(results)+1, figsize=(15, 4))
            
            # Original data
            axes[0].hist(data.ravel(), bins=30, alpha=0.7, color='gray')
            axes[0].set_title('Original')
            
            # Normalized data
            for idx, (method, norm_data) in enumerate(results.items(), 1):
                axes[idx].hist(norm_data.ravel(), bins=30, alpha=0.7)
                axes[idx].set_title(method.upper())
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("⚠️ matplotlib not available. Install with: pip install matplotlib")
    
    return results


# ===== DEMO & TESTING =====
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 DEMO: Upgraded NumPy Array Utilities")
    print("=" * 60)
    
    # Example data with outliers
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(50, 10, 100),  # Normal data
        [999, -999]  # Outliers
    ])
    
    # 1. Print Shape with verbose mode
    print("\n📋 ARRAY INSPECTION:")
    print("-" * 40)
    print_shape(data, "SampleData", verbose=True)
    
    # 2. Compare normalization methods
    print("\n🔄 NORMALIZATION COMPARISON:")
    print("-" * 40)
    
    methods_to_test = ['zscore', 'minmax', 'robust']
    
    for method in methods_to_test:
        norm_data, stats = normalize(data, method=method, return_stats=True)
        print(f"\n✨ Method: {method.upper()}")
        print(f"   Stats: {stats}")
        print(f"   Result (first 5): {norm_data[:5]}")
        print(f"   Mean: {norm_data.mean():.4f}, Std: {norm_data.std():.4f}")
        
        # Denormalize test
        original_recovered = denormalize(norm_data, stats, method)
        error = np.abs(original_recovered - data).max()
        print(f"   Reconstruction error: {error:.6f}")
    
    # 3. Example with 2D data (typical ML dataset)
    print("\n📊 2D DATA EXAMPLE:")
    print("-" * 40)
    X = np.random.randn(100, 4) * [1, 10, 100, 1000]  # Features with different scales
    print_shape(X, "FeatureMatrix", verbose=True)
    
    X_normalized = normalize(X, axis=0, method='zscore')
    print(f"\nBefore normalization - Column means: {X.mean(axis=0)}")
    print(f"After normalization  - Column means: {X_normalized.mean(axis=0)}")
    print(f"After normalization  - Column stds:  {X_normalized.std(axis=0)}")
    
    print("\n✅ Demo completed!")
