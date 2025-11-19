import numpy as np
import time
import tracemalloc
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import networkx as nx
from fgsd import FGSD
import warnings
import os
import urllib.request
import zipfile
warnings.filterwarnings('ignore')


def download_and_load_enzymes():
    """Download and load ENZYMES dataset from TU Dortmund."""
    data_dir = '/tmp/ENZYMES'
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = 'https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip'
    zip_path = os.path.join(data_dir, 'ENZYMES.zip')
    
    # Download if not exists
    if not os.path.exists(os.path.join(data_dir, 'ENZYMES')):
        print("Downloading ENZYMES dataset...")
        urllib.request.urlretrieve(base_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Download complete.")
    
    # Parse dataset files
    dataset_path = os.path.join(data_dir, 'ENZYMES')
    
    # Read graph indicator (which graph each node belongs to)
    graph_indicator = np.loadtxt(os.path.join(dataset_path, 'ENZYMES_graph_indicator.txt'), dtype=int)
    
    # Read edges (A.txt contains edge list)
    edges = np.loadtxt(os.path.join(dataset_path, 'ENZYMES_A.txt'), dtype=int, delimiter=',')
    
    # Read graph labels
    graph_labels = np.loadtxt(os.path.join(dataset_path, 'ENZYMES_graph_labels.txt'), dtype=int)
    
    # Build NetworkX graphs
    num_graphs = len(graph_labels)
    graphs = [nx.Graph() for _ in range(num_graphs)]
    
    # Add nodes
    for node_id, graph_id in enumerate(graph_indicator, start=1):
        graphs[graph_id - 1].add_node(node_id)
    
    # Add edges
    for edge in edges:
        node1, node2 = edge
        graph_id = graph_indicator[node1 - 1]
        graphs[graph_id - 1].add_edge(node1, node2)
    
    # Relabel nodes to be contiguous starting from 0
    graphs = [nx.convert_node_labels_to_integers(g) for g in graphs]
    
    # Convert labels to 0-indexed
    labels = graph_labels - 1
    
    return graphs, labels


def load_enzymes_dataset():
    """Load ENZYMES dataset."""
    return download_and_load_enzymes()


def evaluate_classifier(X_train, X_test, y_train, y_test, classifier_name, clf):
    """Evaluate a single classifier and return metrics."""
    # Training
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Prediction
    start_time = time.time()
    y_pred = clf.predict(X_test)
    inference_time = time.time() - start_time
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # AUC (one-vs-rest for multiclass)
    try:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
        if hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(X_test)
        elif hasattr(clf, 'decision_function'):
            y_score = clf.decision_function(X_test)
            # Normalize for multiclass
            if len(y_score.shape) == 1:
                y_score = y_score.reshape(-1, 1)
        else:
            y_score = None
        
        if y_score is not None and y_test_bin.shape[1] > 1:
            auc = roc_auc_score(y_test_bin, y_score, average='weighted', multi_class='ovr')
        else:
            auc = None
    except:
        auc = None
    
    return {
        'classifier': classifier_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'train_time': train_time,
        'inference_time': inference_time
    }


def run_experiment(embedding_dims, test_size=0.3, random_state=42):
    """Run complete experiment with varying embedding dimensions."""
    print("Loading ENZYMES dataset...")
    graphs, labels = load_enzymes_dataset()
    
    # Split data
    graphs_train, graphs_test, y_train, y_test = train_test_split(
        graphs, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    results = []
    
    for dim in embedding_dims:
        print(f"\n{'='*60}")
        print(f"Evaluating with embedding dimension (hist_bins): {dim}")
        print(f"{'='*60}")
        
        # Generate embeddings with memory tracking
        tracemalloc.start()
        start_time = time.time()
        
        model = FGSD(hist_bins=dim, hist_range=20, seed=random_state)
        model.fit(graphs_train)
        X_train = model.get_embedding()
        X_test = model.infer(graphs_test)
        
        generation_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_mb = peak / 1024 / 1024
        
        print(f"Embedding generation time: {generation_time:.2f}s")
        print(f"Peak memory usage: {memory_mb:.2f} MB")
        print(f"Train embeddings shape: {X_train.shape}")
        print(f"Test embeddings shape: {X_test.shape}")
        
        # Define classifiers
        classifiers = {
            'SVM (RBF)': SVC(kernel='rbf', random_state=random_state, probability=True),
            'SVM (Linear)': SVC(kernel='linear', random_state=random_state, probability=True),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, 
                                random_state=random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state)
        }
        
        # Evaluate each classifier
        for clf_name, clf in classifiers.items():
            print(f"\nEvaluating {clf_name}...")
            result = evaluate_classifier(X_train, X_test, y_train, y_test, clf_name, clf)
            result['embedding_dim'] = dim
            result['generation_time'] = generation_time
            result['memory_mb'] = memory_mb
            results.append(result)
            
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  F1-Score: {result['f1_score']:.4f}")
            if result['auc'] is not None:
                print(f"  AUC: {result['auc']:.4f}")
            print(f"  Training time: {result['train_time']:.4f}s")
    
    return results


def print_summary(results):
    """Print summary of all results."""
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"{'Dim':<6} {'Classifier':<20} {'Accuracy':<10} {'F1-Score':<10} {'AUC':<10} "
          f"{'Gen Time':<10} {'Train Time':<12} {'Memory (MB)':<12}")
    print("-"*80)
    
    for r in results:
        auc_str = f"{r['auc']:.4f}" if r['auc'] is not None else "N/A"
        print(f"{r['embedding_dim']:<6} {r['classifier']:<20} {r['accuracy']:<10.4f} "
              f"{r['f1_score']:<10.4f} {auc_str:<10} {r['generation_time']:<10.2f} "
              f"{r['train_time']:<12.4f} {r['memory_mb']:<12.2f}")


if __name__ == "__main__":
    # Vary embedding dimensions
    embedding_dimensions = [50, 100, 200, 300, 500]
    
    print("Starting FGSD evaluation on ENZYMES dataset")
    results = run_experiment(embedding_dimensions)
    print_summary(results)
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('../results/fgsd_enzymes_results.csv', index=False)
    print(f"\nResults saved to results/fgsd_enzymes_results.csv")
