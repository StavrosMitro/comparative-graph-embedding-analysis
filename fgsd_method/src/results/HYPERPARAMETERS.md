# FGSD Classification Hyperparameters

## Classifier Hyperparameters

All experiments use the following classifier configurations:

### SVM (RBF Kernel)
```python
Pipeline([
    StandardScaler(),
    SVC(kernel='rbf', C=100, gamma='scale', probability=True)
])
```

### Random Forest
```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    n_jobs=-1
)
```

### MLP (Multi-Layer Perceptron)
```python
Pipeline([
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=1000,
        early_stopping=True
    )
])
```

## FGSD Embedding Parameters

### Function Types
- **harmonic**: Uses harmonic mean wavelet function
- **polynomial**: Uses polynomial wavelet function  
- **biharmonic**: Uses biharmonic wavelet function

### Range Values (from preanalysis)
- Computed as p99 (99th percentile) of eigenvalue distribution
- Dataset-specific values stored in preanalysis cache

### Bin Sizes (embedding dimension)
- Tested values vary by dataset
- ENZYMES: [50, 100, 150]
- IMDB-MULTI: [50, 100, 200]
- REDDIT-MULTI-12K: [200, 500]

## Train/Test Split
- Test size: 15% (stratified by class labels)
- Random state: 42
