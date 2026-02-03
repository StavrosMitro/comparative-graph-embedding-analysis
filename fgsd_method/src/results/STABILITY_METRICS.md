# FGSD Stability Analysis Metrics

## Overview

Stability analysis measures how FGSD embeddings change when the input graphs are perturbed.
This is important for understanding robustness to noise and measurement errors in real-world graphs.

## Perturbation Methods

### Default Mode
- Removes `n/2` random edges
- Adds `n/2` random non-existing edges
- Where `n = perturbation_ratio × total_edges`

### Remove-Only Mode
- Only removes edges (simulates missing data)
- `n = perturbation_ratio × total_edges`

### Add-Only Mode  
- Only adds edges (simulates noise/false positives)
- `n = perturbation_ratio × total_edges`

## Stability Metrics

### 1. Cosine Similarity
```
similarity = (X_original · X_perturbed) / (||X_original|| × ||X_perturbed||)
```
- **Range**: [-1, 1], typically [0.7, 1.0] for stable embeddings
- **Interpretation**: Higher = more stable embeddings
- **Ideal**: 1.0 (embeddings unchanged)

### 2. Relative Change (L2)
```
relative_change = ||X_original - X_perturbed|| / ||X_original||
```
- **Range**: [0, ∞), typically [0, 0.5]
- **Interpretation**: Lower = less sensitive to perturbation
- **Ideal**: 0.0 (no change)

### 3. Classification Accuracy Drop
```
accuracy_drop_pct = (acc_original - acc_perturbed) / acc_original × 100
```
- **Range**: Can be negative (improvement) or positive (degradation)
- **Interpretation**: Lower = more robust classification
- **Ideal**: 0% (no degradation)

## Perturbation Ratios Tested

| Ratio | Description |
|-------|-------------|
| 1% | Minor noise |
| 5% | Moderate noise |
| 10% | Significant noise |
| 20% | Heavy noise |

## Classifier Configuration

For stability analysis, we use **Random Forest** only:
```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    n_jobs=-1
)
```

## Expected Behavior

1. **Cosine similarity** should decrease as perturbation increases
2. **Relative change** should increase with perturbation
3. **Accuracy drop** should increase with perturbation
4. **Harmonic function** typically shows higher stability than polynomial
5. **Hybrid methods** may show intermediate stability
