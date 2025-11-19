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
from karateclub.estimator import Estimator # Χρειαζόμαστε τη βάση
import warnings
import os
import urllib.request
import zipfile
import pandas as pd
import sys
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fgsd import FlexibleFGSD 

def download_and_load_enzymes():
    """Download and load ENZYMES dataset from TU Dortmund."""
    data_dir = '/tmp/ENZYMES'
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = 'https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip'
    zip_path = os.path.join(data_dir, 'ENZYMES.zip')
    
    if not os.path.exists(os.path.join(data_dir, 'ENZYMES')):
        print("Downloading ENZYMES dataset...")
        urllib.request.urlretrieve(base_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Download complete.")
    
    dataset_path = os.path.join(data_dir, 'ENZYMES')
    graph_indicator = np.loadtxt(os.path.join(dataset_path, 'ENZYMES_graph_indicator.txt'), dtype=int)
    edges = np.loadtxt(os.path.join(dataset_path, 'ENZYMES_A.txt'), dtype=int, delimiter=',')
    graph_labels = np.loadtxt(os.path.join(dataset_path, 'ENZYMES_graph_labels.txt'), dtype=int)
    
    num_graphs = len(graph_labels)
    graphs = [nx.Graph() for _ in range(num_graphs)]
    
    for node_id, graph_id in enumerate(graph_indicator, start=1):
        graphs[graph_id - 1].add_node(node_id)
    
    for edge in edges:
        node1, node2 = edge
        graph_id = graph_indicator[node1 - 1]
        graphs[graph_id - 1].add_edge(node1, node2)
    
    graphs = [nx.convert_node_labels_to_integers(g) for g in graphs]
    labels = graph_labels - 1
    return graphs, labels

def load_enzymes_dataset():
    return download_and_load_enzymes()

# --- 3. EVALUATION (UNCHANGED) ---
def evaluate_classifier(X_train, X_test, y_train, y_test, classifier_name, clf):
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
    
    try:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
        if hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(X_test)
        elif hasattr(clf, 'decision_function'):
            y_score = clf.decision_function(X_test)
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

# --- 4. EXPERIMENT RUNNER (UPDATED FOR TRIPLETS) ---
def run_experiment(configs, test_size=0.3, random_state=42):
    """
    Run complete experiment iterating over configuration triplets.
    configs: List of dictionaries {'func': str, 'bins': int, 'range': float}
    """
    print("Loading ENZYMES dataset...")
    graphs, labels = load_enzymes_dataset()
    
    # Split data
    graphs_train, graphs_test, y_train, y_test = train_test_split(
        graphs, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    results = []
    
    for i, config in enumerate(configs):
        func = config['func']
        bins = config['bins']
        rng = config['range']
        
        print(f"\n{'='*80}")
        print(f"Experiment {i+1}/{len(configs)}: Function='{func}', Bins={bins}, Range={rng}")
        print(f"{'='*80}")
        
        # Generate embeddings
        tracemalloc.start()
        start_time = time.time()
        
        # Χρησιμοποιούμε τη FlexibleFGSD
        model = FlexibleFGSD(hist_bins=bins, hist_range=rng, func_type=func, seed=random_state)
        model.fit(graphs_train)
        X_train = model.get_embedding()
        X_test = model.infer(graphs_test)
        
        generation_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_mb = peak / 1024 / 1024
        
        print(f"Generation Time: {generation_time:.2f}s | Memory: {memory_mb:.2f} MB")
        print(f"Embeddings Shape: {X_train.shape}")
        
        classifiers = {
            'SVM (RBF) + Scaler': make_pipeline(
                StandardScaler(), # we added scaler because we put raw data
                SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=random_state)
            ),
            'Random Forest': RandomForestClassifier(n_estimators=1000, random_state=random_state)
        }
        
        for clf_name, clf in classifiers.items():
            res = evaluate_classifier(X_train, X_test, y_train, y_test, clf_name, clf)
            # Αποθηκεύουμε την τριάδα στα αποτελέσματα
            res['func'] = func
            res['bins'] = bins
            res['range'] = rng
            res['generation_time'] = generation_time
            results.append(res)
            print(f"  -> {clf_name}: Acc={res['accuracy']:.4f}, F1={res['f1_score']:.4f}")
    
    return results

def print_summary(results):
    print("\n" + "="*100)
    print("SUMMARY OF RESULTS")
    print("="*100)
    print(f"{'Func':<12} {'Bins':<6} {'Range':<8} {'Classifier':<15} {'Accuracy':<10} {'F1':<10} {'GenTime':<8}")
    print("-" * 100)
    
    # Sort by Accuracy descending
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    for r in sorted_results:
        print(f"{r['func']:<12} {r['bins']:<6} {r['range']:<8.1f} {r['classifier']:<15} "
              f"{r['accuracy']:<10.4f} {r['f1_score']:<10.4f} {r['generation_time']:<8.2f}")

# --- 5. MAIN CONFIGURATION ---
if __name__ == "__main__":
    
    # ΕΔΩ ΟΡΙΖΕΙΣ ΤΙΣ ΤΡΙΑΔΕΣ ΣΟΥ
    # Προσοχή: Για την 'polynomial' και 'biharmonic' το Range πρέπει 
    # κανονικά να βρεθεί με Pre-analysis. Εδώ βάζω ενδεικτικές τιμές.
    
    configurations = [
        # --- HARMONIC (Global) ---
        # Χρησιμοποιούμε το Range=33 που βρήκες εσύ
        {'func': 'harmonic', 'bins': 100, 'range': 33},
        {'func': 'harmonic', 'bins': 200, 'range': 33},
        {'func': 'harmonic', 'bins': 300, 'range': 33},
        
        # # --- POLYNOMIAL (Local - f(λ)=λ^2) ---
        # {'func': 'polynomial', 'bins': 50, 'range': 5},
        # {'func': 'polynomial', 'bins': 100, 'range': 5},
        
        # # --- BIHARMONIC (Global - f(λ)=1/λ^2) ---

        # {'func': 'biharmonic', 'bins': 100, 'range': 800},
    ]
    
    print("Starting Multi-Configuration FGSD Experiment...")
    results = run_experiment(configurations)
    print_summary(results)
    
    # Save
    df = pd.DataFrame(results)
    output_path = '../results/fgsd_triplets_results.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")