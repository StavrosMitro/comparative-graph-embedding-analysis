import numpy as np
import time
import tracemalloc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize, StandardScaler
import networkx as nx
from fgsd import FGSD
import warnings
import os
import urllib.request
import zipfile
warnings.filterwarnings('ignore')


def download_and_load_reddit():
    """Download and load REDDIT-MULTI-12K dataset from TU Dortmund."""
    data_dir = '/tmp/REDDIT-MULTI-12K'
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = 'https://www.chrsmrrs.com/graphkerneldatasets/REDDIT-MULTI-12K.zip'
    zip_path = os.path.join(data_dir, 'REDDIT-MULTI-12K.zip')
    
    # Download if not exists
    if not os.path.exists(os.path.join(data_dir, 'REDDIT-MULTI-12K')):
        print("Downloading REDDIT-MULTI-12K dataset (this may take a while)...")
        urllib.request.urlretrieve(base_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Download complete.")
    
    # Parse dataset files
    dataset_path = os.path.join(data_dir, 'REDDIT-MULTI-12K')
    
    print("Loading dataset files...")
    # Read graph indicator
    graph_indicator = np.loadtxt(os.path.join(dataset_path, 'REDDIT-MULTI-12K_graph_indicator.txt'), dtype=int)
    
    # Read edges
    edges = np.loadtxt(os.path.join(dataset_path, 'REDDIT-MULTI-12K_A.txt'), dtype=int, delimiter=',')
    
    # Read graph labels
    graph_labels = np.loadtxt(os.path.join(dataset_path, 'REDDIT-MULTI-12K_graph_labels.txt'), dtype=int)
    
    print("Building NetworkX graphs...")
    # Build NetworkX graphs
    num_graphs = len(graph_labels)
    graphs = [nx.Graph() for _ in range(num_graphs)]
    
    # Add nodes
    for node_id, graph_id in enumerate(graph_indicator, start=1):
        graphs[graph_id - 1].add_node(node_id)
    
    # Add edges
    print(f"Adding {len(edges)} edges...")
    for i, edge in enumerate(edges):
        if i % 100000 == 0:
            print(f"  Processed {i}/{len(edges)} edges...")
        node1, node2 = edge
        graph_id = graph_indicator[node1 - 1]
        graphs[graph_id - 1].add_edge(node1, node2)
    
    # Relabel nodes to be contiguous starting from 0
    print("Relabeling nodes...")
    graphs = [nx.convert_node_labels_to_integers(g) for g in graphs]
    
    # Convert labels to 0-indexed
    labels = graph_labels - 1
    
    print(f"Dataset loaded: {len(graphs)} graphs")
    return graphs, labels


def analyze_dataset(graphs, labels):
    """Analyze the dataset to understand its properties."""
    print("\n" + "="*70)
    print("DATASET ANALYSIS")
    print("="*70)
    print(f"Number of graphs: {len(graphs)}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution:")
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} graphs ({100*count/len(labels):.2f}%)")
    
    # Sample graphs for statistics (full dataset is large)
    sample_size = min(1000, len(graphs))
    sample_indices = np.random.choice(len(graphs), sample_size, replace=False)
    sample_graphs = [graphs[i] for i in sample_indices]
    
    # Graph statistics
    num_nodes = [g.number_of_nodes() for g in sample_graphs]
    num_edges = [g.number_of_edges() for g in sample_graphs]
    densities = [nx.density(g) if g.number_of_nodes() > 1 else 0 for g in sample_graphs]
    
    print(f"\nGraph statistics (based on {sample_size} samples):")
    print(f"  Nodes - Min: {min(num_nodes)}, Max: {max(num_nodes)}, Mean: {np.mean(num_nodes):.2f}, Std: {np.std(num_nodes):.2f}")
    print(f"  Edges - Min: {min(num_edges)}, Max: {max(num_edges)}, Mean: {np.mean(num_edges):.2f}, Std: {np.std(num_edges):.2f}")
    print(f"  Density - Min: {min(densities):.4f}, Max: {max(densities):.4f}, Mean: {np.mean(densities):.4f}")
    
    # Check connectivity
    connected = [nx.is_connected(g) for g in sample_graphs]
    print(f"  Connected graphs: {sum(connected)}/{len(sample_graphs)} ({100*sum(connected)/len(sample_graphs):.2f}%)")
    print("="*70 + "\n")


def evaluate_classifier(X_train, X_test, y_train, y_test, classifier_name, clf, use_scaling=True):
    """Evaluate a single classifier with optional feature scaling."""
    # Feature scaling
    if use_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Training
    start_time = time.time()
    clf.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    # Prediction
    start_time = time.time()
    y_pred = clf.predict(X_test_scaled)
    inference_time = time.time() - start_time
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # AUC (one-vs-rest for multiclass)
    try:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
        if hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(X_test_scaled)
        elif hasattr(clf, 'decision_function'):
            y_score = clf.decision_function(X_test_scaled)
            if len(y_score.shape) == 1:
                y_score = y_score.reshape(-1, 1)
        else:
            y_score = None
        
        if y_score is not None and y_test_bin.shape[1] > 1:
            auc = roc_auc_score(y_test_bin, y_score, average='weighted', multi_class='ovr')
        else:
            auc = None
    except Exception as e:
        auc = None
    
    return {
        'classifier': classifier_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'train_time': train_time,
        'inference_time': inference_time
    }


def run_cross_validation(graphs, labels, embedding_dim, n_folds=3, sample_size=3000):
    """Run cross-validation experiment on a sample (dataset is large)."""
    print(f"\nRunning {n_folds}-fold cross-validation on {sample_size} samples...")
    
    # Sample data for CV (full dataset is too large)
    indices = np.random.choice(len(graphs), sample_size, replace=False)
    graphs_sample = [graphs[i] for i in indices]
    labels_sample = labels[indices]
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_results = {
        'SVM (RBF)': [],
        'Random Forest': [],
        'Logistic Regression': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(graphs_sample, labels_sample), 1):
        print(f"  Fold {fold}/{n_folds}...", end=' ')
        
        # Split data
        graphs_train = [graphs_sample[i] for i in train_idx]
        graphs_test = [graphs_sample[i] for i in test_idx]
        y_train = labels_sample[train_idx]
        y_test = labels_sample[test_idx]
        
        # Generate embeddings
        model = FGSD(hist_bins=embedding_dim, hist_range=20, seed=42)
        model.fit(graphs_train)
        X_train = model.get_embedding()
        X_test = model.infer(graphs_test)
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Evaluate classifiers
        for clf_name in cv_results.keys():
            if clf_name == 'SVM (RBF)':
                clf = SVC(kernel='rbf', C=1.0, random_state=42)
            elif clf_name == 'Random Forest':
                clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:  # Logistic Regression
                clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cv_results[clf_name].append(acc)
        
        print("Done")
    
    print("\nCross-validation results:")
    for clf_name, accs in cv_results.items():
        print(f"  {clf_name}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    
    return cv_results


def run_experiment(embedding_dims, test_size=0.2, random_state=42, max_train_size=8000):
    """Run complete experiment with varying embedding dimensions."""
    print("Loading REDDIT-MULTI-12K dataset...")
    graphs, labels = download_and_load_reddit()
    
    # Analyze dataset
    analyze_dataset(graphs, labels)
    
    # Convert labels to numpy array if needed
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    # Use subset for faster experiments (full dataset is large)
    if len(graphs) > max_train_size / (1 - test_size):
        print(f"\nUsing subset of {int(max_train_size / (1 - test_size))} graphs for efficiency...")
        indices = np.random.choice(len(graphs), int(max_train_size / (1 - test_size)), replace=False)
        graphs = [graphs[i] for i in indices]
        labels = labels[indices]
    
    # Split data
    graphs_train, graphs_test, y_train, y_test = train_test_split(
        graphs, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    print(f"Train set: {len(graphs_train)} graphs, Test set: {len(graphs_test)} graphs")
    
    results = []
    
    for dim in embedding_dims:
        print(f"\n{'='*70}")
        print(f"Evaluating with embedding dimension (hist_bins): {dim}")
        print(f"{'='*70}")
        
        # Generate embeddings with memory tracking
        tracemalloc.start()
        start_time = time.time()
        
        print("Generating embeddings for training set...")
        model = FGSD(hist_bins=dim, hist_range=20, seed=random_state)
        model.fit(graphs_train)
        X_train = model.get_embedding()
        
        print("Generating embeddings for test set...")
        X_test = model.infer(graphs_test)
        
        generation_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_mb = peak / 1024 / 1024
        
        print(f"Embedding generation time: {generation_time:.2f}s")
        print(f"Peak memory usage: {memory_mb:.2f} MB")
        print(f"Train embeddings shape: {X_train.shape}")
        print(f"Test embeddings shape: {X_test.shape}")
        
        # Check for variance in embeddings
        variance = np.var(X_train, axis=0)
        print(f"Feature variance - Min: {np.min(variance):.4f}, Max: {np.max(variance):.4f}, Mean: {np.mean(variance):.4f}")
        
        # Define classifiers (optimized for large dataset)
        classifiers = {
            'SVM (Linear)': SVC(kernel='linear', C=1.0, random_state=random_state, probability=True),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=-1),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, 
                                                   random_state=random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                           random_state=random_state)
        }
        
        # Evaluate each classifier
        for clf_name, clf in classifiers.items():
            print(f"\nEvaluating {clf_name}...")
            result = evaluate_classifier(X_train, X_test, y_train, y_test, clf_name, clf, use_scaling=True)
            result['embedding_dim'] = dim
            result['generation_time'] = generation_time
            result['memory_mb'] = memory_mb
            results.append(result)
            
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  F1-Score: {result['f1_score']:.4f}")
            if result['auc'] is not None:
                print(f"  AUC: {result['auc']:.4f}")
            print(f"  Training time: {result['train_time']:.4f}s")
            print(f"  Inference time: {result['inference_time']:.4f}s")
    
    # Run cross-validation for best dimension
    best_dim = embedding_dims[len(embedding_dims)//2]
    run_cross_validation(graphs, labels, best_dim)
    
    return results


def print_summary(results):
    """Print summary of all results."""
    print("\n" + "="*100)
    print("SUMMARY OF RESULTS")
    print("="*100)
    print(f"{'Dim':<6} {'Classifier':<22} {'Accuracy':<10} {'F1-Score':<10} {'AUC':<10} "
          f"{'Gen Time':<11} {'Train Time':<12} {'Infer Time':<12} {'Memory (MB)':<12}")
    print("-"*100)
    
    for r in results:
        auc_str = f"{r['auc']:.4f}" if r['auc'] is not None else "N/A"
        print(f"{r['embedding_dim']:<6} {r['classifier']:<22} {r['accuracy']:<10.4f} "
              f"{r['f1_score']:<10.4f} {auc_str:<10} {r['generation_time']:<11.2f} "
              f"{r['train_time']:<12.4f} {r['inference_time']:<12.4f} {r['memory_mb']:<12.2f}")
    
    # Find best results
    print("\n" + "="*100)
    print("BEST RESULTS PER METRIC")
    print("="*100)
    
    best_acc = max(results, key=lambda x: x['accuracy'])
    best_f1 = max(results, key=lambda x: x['f1_score'])
    fastest_gen = min(results, key=lambda x: x['generation_time'])
    
    print(f"Best Accuracy: {best_acc['accuracy']:.4f} - {best_acc['classifier']} (dim={best_acc['embedding_dim']})")
    print(f"Best F1-Score: {best_f1['f1_score']:.4f} - {best_f1['classifier']} (dim={best_f1['embedding_dim']})")
    print(f"Fastest Generation: {fastest_gen['generation_time']:.2f}s - dim={fastest_gen['embedding_dim']}")


if __name__ == "__main__":
    # Vary embedding dimensions
    embedding_dimensions = [100, 200, 400]
    
    print("="*70)
    print("FGSD EVALUATION ON REDDIT-MULTI-12K DATASET")
    print("="*70)
    
    results = run_experiment(embedding_dimensions, max_train_size=8000)
    print_summary(results)
    
    # Save results
    import pandas as pd
    os.makedirs('/home/stavros/emb3/results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('/home/stavros/emb3/results/fgsd_reddit_results.csv', index=False)
    print(f"\nResults saved to /home/stavros/emb3/results/fgsd_reddit_results.csv")
