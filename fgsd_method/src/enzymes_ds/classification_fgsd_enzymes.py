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
from optimized_method import HybridFGSD

def download_and_load_enzymes():
    """Download and load ENZYMES dataset including Node Labels."""
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
    
    #1d grapjh_indicator[i]=node i belongs to X graph
    graph_indicator = np.loadtxt(os.path.join(dataset_path, 'ENZYMES_graph_indicator.txt'), dtype=int)
    
    #edges...
    edges = np.loadtxt(os.path.join(dataset_path, 'ENZYMES_A.txt'), dtype=int, delimiter=',')
    #each graph in which class belongs
    graph_labels = np.loadtxt(os.path.join(dataset_path, 'ENZYMES_graph_labels.txt'), dtype=int)
    #node label... chemichal type
    node_labels_raw = np.loadtxt(os.path.join(dataset_path, 'ENZYMES_node_labels.txt'), dtype=int)
    
    num_graphs = len(graph_labels) #how many classes do we have?
    graphs = [nx.Graph() for _ in range(num_graphs)]#list of networkx graphs empty
    node_labels_list = [] # Λίστα που θα κρατάει τα labels για κάθε γράφο ξεχωριστά
    
    # Add nodes
    for node_id, graph_id in enumerate(graph_indicator, start=1):
        graphs[graph_id - 1].add_node(node_id) #netowrkx func
    
    # Split node labels by graph
    current_idx = 0
    for i in range(1, num_graphs + 1):
        # Βρίσκουμε πόσοι κόμβοι ανήκουν στον γράφο i
        count = np.sum(graph_indicator == i)
        # Παίρνουμε το κομμάτι των labels που αντιστοιχεί σε αυτόν τον γράφο
        labels_of_graph = node_labels_raw[current_idx : current_idx + count]
        node_labels_list.append(labels_of_graph)
        current_idx += count

    # Add edges
    for edge in edges:
        node1, node2 = edge
        graph_id = graph_indicator[node1 - 1]
        graphs[graph_id - 1].add_edge(node1, node2)
    
    graphs = [nx.convert_node_labels_to_integers(g) for g in graphs]
    labels = graph_labels - 1
    
    return graphs, labels, node_labels_list



def create_node_label_features(node_labels_list):
    all_labels = np.concatenate(node_labels_list)
    unique_labels = np.unique(all_labels)
    n_unique = len(unique_labels)
    min_lbl, max_lbl = min(unique_labels), max(unique_labels)
    
    print(f"Node Labels Found: {unique_labels} (Total unique: {n_unique})")
    
    features = []
    for labels in node_labels_list:
        hist, _ = np.histogram(labels, bins=n_unique, range=(min_lbl, max_lbl + 1))
        features.append(hist)
        
    return np.array(features)

# --- 3. EVALUATION (UNCHANGED) ---
def evaluate_classifier(X_train, X_test, y_train, y_test, classifier_name, clf):
    # Training
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Calculate training accuracy to check for overfitting
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

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
        'train_accuracy': train_accuracy,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'train_time': train_time,
        'inference_time': inference_time
    }

# --- 4. EXPERIMENT RUNNER (UPDATED FOR TRIPLETS) ---
def run_experiment(configs, test_size=0.15, random_state=42):
    """
    Run complete experiment with Node Labels Enrichment.
    """
    print("Loading ENZYMES dataset...")
    # 1. Unpack και τα 3 στοιχεία
    graphs, labels, node_labels_list = download_and_load_enzymes()
    
    # 2. Δημιουργία των 'Chemical Features' (Node Labels)
    # Αυτά υπολογίζονται μία φορά και είναι σταθερά
    X_node_labels = create_node_label_features(node_labels_list)
    print(f"Node Label Features Shape: {X_node_labels.shape}")

    # 3. Split Data
    # Περνάμε ΚΑΙ τα X_node_labels στο split για να κοπούν συγχρονισμένα
    graphs_train, graphs_test, y_train, y_test, X_labels_train, X_labels_test = train_test_split(
        graphs, labels, X_node_labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    results = []
    
    for i, config in enumerate(configs):
        func = config['func']
        
        print(f"\n{'='*80}")
        if func == 'hybrid':
            harm_bins = config.get('harm_bins', 200)
            harm_range = config.get('harm_range', 33)
            pol_bins = config.get('pol_bins', 70)
            pol_range = config.get('pol_range', 4.1)
            print(f"Experiment {i+1}/{len(configs)}: Function='{func}' + Node Labels")
        else:
            bins = config['bins']
            rng = config['range']
            print(f"Experiment {i+1}/{len(configs)}: Function='{func}', Bins={bins}, Range={rng} + Node Labels")
        print(f"{'='*80}")
        
        tracemalloc.start()
        start_time = time.time()
        
        # Initialize spectral model
        if func == 'hybrid':
            model = HybridFGSD(
                harm_bins=harm_bins, harm_range=harm_range,
                pol_bins=pol_bins, pol_range=pol_range,
                func_type='hybrid', seed=random_state
            )
        else:
            model = FlexibleFGSD(hist_bins=bins, hist_range=rng, func_type=func, seed=random_state)

        # Generate Spectral Embeddings
        model.fit(graphs_train)
        X_train_spectral = model.get_embedding()
        X_test_spectral = model.infer(graphs_test)
        
        # --- ENRICHMENT STEP (Η Ένωση) ---
        # Ενώνουμε τα Spectral Features με τα Node Label Features
        # np.hstack κολλάει τους πίνακες οριζόντια
        X_train = np.hstack([X_train_spectral, X_labels_train])
        X_test = np.hstack([X_test_spectral, X_labels_test])
        
        generation_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_mb = peak / 1024 / 1024
        
        print(f"Total Embedding Shape (Spectral + Labels): {X_train.shape}")
        
        classifiers = {
            'SVM (RBF) + Scaler': make_pipeline(
                StandardScaler(),
                SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=random_state)
            ),
            'Random Forest': RandomForestClassifier(n_estimators=1000, random_state=random_state),
            'MLP': make_pipeline(
                StandardScaler(), 
                MLPClassifier(
                    hidden_layer_sizes=(1024, 512, 256, 128), 
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    learning_rate_init=0.001,
                    learning_rate='adaptive',
                    max_iter=2000, 
                    early_stopping=True,      
                    n_iter_no_change=20,      
                    random_state=random_state
                )
            )
        }
        
        for clf_name, clf in classifiers.items():
            res = evaluate_classifier(X_train, X_test, y_train, y_test, clf_name, clf)
            res.update(config)
            res['generation_time'] = generation_time
            results.append(res)
            print(f"  -> {clf_name}: Train Acc={res['train_accuracy']:.4f}, Test Acc={res['accuracy']:.4f}, F1={res['f1_score']:.4f}")
    
    return results

def print_summary(results):
    print("\n" + "="*120)
    print("SUMMARY OF RESULTS")
    print("="*120)
    # Adjusted header to be more generic
    print(f"{'Func':<12} {'Parameters':<30} {'Classifier':<20} {'Train Acc':<11} {'Test Acc':<10} {'F1':<10} {'GenTime':<8}")
    print("-" * 120)
    
    # Sort by Accuracy descending
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    for r in sorted_results:
        if r['func'] == 'hybrid':
            params = f"h_bins={r.get('harm_bins', 'N/A')},h_range={r.get('harm_range', 'N/A')},p_bins={r.get('pol_bins', 'N/A')},p_range={r.get('pol_range', 'N/A')}"
        else:
            params = f"bins={r.get('bins', 'N/A')}, range={r.get('range', 'N/A')}"

        print(f"{r['func']:<12} {params:<30} {r['classifier']:<20} "
              f"{r['train_accuracy']:<11.4f} {r['accuracy']:<10.4f} {r['f1_score']:<10.4f} {r['generation_time']:<8.2f}")

# --- 5. MAIN CONFIGURATION ---
if __name__ == "__main__":
    
    configurations = [
        # --- HYBRID EXAMPLE ---
        # For 'hybrid', specify bins and range for both parts.
        {'func': 'hybrid', 'harm_bins': 100, 'harm_range': 33, 'pol_bins': 100, 'pol_range': 4.1},

        # # --- HARMONIC (Global) ---
        # # Uses 'bins' and 'range' keys.
        # {'func': 'harmonic', 'bins': 300, 'range': 33},
        
        # # --- POLYNOMIAL (Local - f(λ)=λ^2) ---
        # # Uses 'bins' and 'range' keys.
        {'func': 'polynomial', 'bins': 100, 'range': 4.1},
    ]
    
    print("Starting Multi-Configuration FGSD Experiment...")
    results = run_experiment(configurations)
    print_summary(results)
    
    # Save
    df = pd.DataFrame(results)
    output_path = '../results/fgsd_triplets_labels.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")