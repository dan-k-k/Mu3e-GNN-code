# evaluategraphs.py

# MULTI; def evaluate and plot for individual classes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import torch
import torch.nn.functional as F
from scipy.stats import beta
import seaborn as sns
from torch_geometric.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# you can evaluate either for individual or combined classes real classes ie treating labels 0 and 1 both as real and 2 as fake
# and you can evaluate with a threshold or by using argmax which simply has predictions of the class with highest GNN output.

# def evaluate using a threshold
# the 'optimal' threshold is calculated from the validation set where the F1 score is maximised ie. the product of purity and efficiency is maximised. 
# and must not be found for the test set (must be treated as unseen). i have commented out plots as these are shown overlapping later in the GNN notebook.
def evaluate_for_class(model, data_loader, class_index, dataset_name="Dataset", num_thresholds=1000, fixed_threshold=None):
    model.eval()
    y_true = []
    y_probs = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            outputs = model(batch)  # shape: [batch_size, num_classes]
            # Compute softmax probabilities.
            probs = F.softmax(outputs, dim=1).cpu().numpy()  # shape: [batch_size, num_classes]
            # Convert true labels to a binary indicator: 1 if equal to class_index, 0 otherwise.
            labels = (batch.label.cpu().numpy().astype(int) == class_index).astype(int)
            y_true.extend(labels)
            # Extract predicted probability for the class of interest.
            y_probs.extend(probs[:, class_index])
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    # Use fixed_threshold if provided; else compute one via precision-recall.
    if fixed_threshold is not None:
        optimal_threshold = fixed_threshold
        print(f"Using fixed threshold: {optimal_threshold:.4f}")
    else:
        precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_true, y_probs)
        if len(thresholds_pr) > 0:
            optimal_idx = np.argmax(precision_vals[:-1] * recall_vals[:-1])
            optimal_threshold = thresholds_pr[optimal_idx]
            print(f"Optimal Threshold for class {class_index} based on {dataset_name}: {optimal_threshold:.4f}")
        else:
            optimal_threshold = 0.5
            print("No thresholds found; using default threshold: 0.6")
    
    # Compute efficiency (true positive rate) and background rejection rate (BRR) across thresholds.
    sampled_thresholds = np.linspace(0, 1, num_thresholds)
    brr = np.zeros(num_thresholds)
    efficiency = np.zeros(num_thresholds)
    for i, thresh in enumerate(sampled_thresholds):
        temp_pred = (y_probs >= thresh).astype(int)
        TP = np.sum((temp_pred == 1) & (y_true == 1))
        FP = np.sum((temp_pred == 1) & (y_true == 0))
        TN = np.sum((temp_pred == 0) & (y_true == 0))
        FN = np.sum((temp_pred == 0) & (y_true == 1))
        efficiency[i] = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        brr[i] = (TN + FP) / FP if FP > 0 else np.nan
    valid = ~np.isnan(brr) & ~np.isnan(efficiency)
    brr = brr[valid]
    efficiency = efficiency[valid]
    sampled_thresholds = sampled_thresholds[valid]
        
    # Plot ROC Curve.
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc_val = auc(fpr, tpr)
    epsilon = 1e-6
    fpr_nonzero = np.maximum(fpr, epsilon)
    fpr_diag = np.linspace(epsilon, 1, 100)
        
    # Final predictions using the fixed threshold.
    y_pred_final = (y_probs >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_final)
    cr = classification_report(y_true, y_pred_final, target_names=["Negative", "Positive"], zero_division=0, digits=4)
    
    return {"confusion_matrix": cm, "classification_report": cr, "optimal_threshold": optimal_threshold, "sampled_thresholds": sampled_thresholds, "efficiency": efficiency, "brr": brr, "y_true": y_true, "y_probs": y_probs, "y_pred": y_pred_final, "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc_val
    }

def evaluate_combined(model, data_loader, dataset_name="Dataset", fixed_threshold=None, num_thresholds=1000):
    model.eval()
    y_true = []
    y_probs = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            outputs = model(batch)  # shape: [batch_size, 3]
            probs = F.softmax(outputs, dim=1).cpu().numpy()  # softmax over classes
            # Aggregate real probability: sum of probabilities for class 0 and class 1
            aggregated_probs = probs[:, 0] + probs[:, 1]
            # Ground truth: consider tracks with label 0 or 1 as "real" (1) and label 2 as "fake" (0)
            true_labels = (batch.label.cpu().numpy() != 2).astype(int)
            
            y_true.extend(true_labels)
            y_probs.extend(aggregated_probs)
    
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    # Determine the optimal threshold using precision-recall curve if not fixed.
    if fixed_threshold is not None:
        optimal_threshold = fixed_threshold
        print(f"Using fixed threshold: {optimal_threshold:.4f}")
    else:
        precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_true, y_probs)
        if len(thresholds_pr) > 0:
            # A simple way: maximize the product of precision and recall.
            optimal_idx = np.argmax(precision_vals[:-1] * recall_vals[:-1])
            optimal_threshold = thresholds_pr[optimal_idx]
        else:
            optimal_threshold = 0.5  # default fallback threshold
        print(f"Optimal Threshold for Real vs Fake on {dataset_name}: {optimal_threshold:.4f}")
    
    # Plotting Efficiency vs. Background Rejection Rate (BRR)
    sampled_thresholds = np.linspace(0, 1, num_thresholds)
    efficiency = np.zeros(num_thresholds)
    brr = np.zeros(num_thresholds)
    
    for i, thresh in enumerate(sampled_thresholds):
        temp_pred = (y_probs >= thresh).astype(int)
        TP = np.sum((temp_pred == 1) & (y_true == 1))
        FP = np.sum((temp_pred == 1) & (y_true == 0))
        TN = np.sum((temp_pred == 0) & (y_true == 0))
        FN = np.sum((temp_pred == 0) & (y_true == 1))
        efficiency[i] = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        # Background rejection rate: here, you could define it as TN/FP or similar; adjust as needed.
        brr[i] = (FP + TN) / FP if FP > 0 else np.nan

    # Filter out invalid values
    valid = ~np.isnan(brr) & ~np.isnan(efficiency)
    sampled_thresholds = sampled_thresholds[valid]
    efficiency = efficiency[valid]
    brr = brr[valid]

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc_val = auc(fpr, tpr)
    epsilon = 1e-6
    fpr_nonzero = np.maximum(fpr, epsilon)
    fpr_diag = np.linspace(epsilon, 1, 100)
    
    # Final predictions based on the optimal threshold
    y_pred_final = (y_probs >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_final)    
    cr=classification_report(y_true, y_pred_final, target_names=["Fake", "Real"], zero_division=0, digits=4)
    
    return {"confusion_matrix": cm, "classification_report": cr, "optimal_threshold": optimal_threshold, "sampled_thresholds": sampled_thresholds, "efficiency": efficiency, "brr": brr, "y_true": y_true, "y_probs": y_probs, "y_pred": y_pred_final, "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc_val
    }

# or simply find with def argmax, no threshold needed. define combined and individual approach.
def evaluate_with_argmax_combined(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            outputs = model(batch)  # shape: [batch_size, 3]
            # Use argmax over the three classes
            preds = torch.argmax(outputs, dim=1)
            # Convert predictions into a binary decision:
            # real if prediction in [0, 1] and fake if prediction == 2.
            binary_preds = (preds != 2).int()  # real: 1, fake: 0
            y_pred.extend(binary_preds.cpu().numpy())
            
            # Ground truth: same logic, true real if label in [0,1]
            binary_true = (batch.label != 2).int()
            y_true.extend(binary_true.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Now you can compute and print a confusion matrix or classification report:
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=["Fake", "Real"], zero_division=0, digits=4)
    
    return y_true, y_pred, cm, cr

def evaluate_class_with_argmax(model, data_loader, device, class_index):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            outputs = model(batch)  # shape: [batch_size, num_classes]
            preds = torch.argmax(outputs, dim=1)  # Get the predicted class for each sample
            
            # For individual class evaluation, consider the sample "positive" if its label equals class_index.
            binary_preds = (preds == class_index).int()  # 1 if predicted == class_index, else 0
            binary_true  = (batch.label == class_index).int()  # 1 if true label == class_index, else 0
            
            y_pred.extend(binary_preds.cpu().numpy())
            y_true.extend(binary_true.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute confusion matrix and classification report.
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=["Not Class", "Class"], zero_division=0, digits=4)
    
    return y_true, y_pred, cm, cr

#before overlap removal
def efficiency_with_CP(y_true, y_pred, label="dataset", alpha=0.3173):
    """
    Print efficiency and its 1‑σ Clopper–Pearson interval.
    y_true, y_pred are the binary arrays returned by evaluate_with_argmax_combined.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    total = TP + FN + FP + TN

    n  = TP + FN
    eff = TP / n if n else 0.0
    pur = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    acc = (TP + TN) / total if total > 0 else 0.0

    # --- Clopper–Pearson (±1 σ) ---
    if n == 0:
        lo = hi = 0.0
    else:
        lo = 0.0 if TP == 0 else beta.ppf(alpha/2, TP, n - TP + 1)
        hi = 1.0 if TP == n else beta.ppf(1 - alpha/2, TP + 1, n - TP)

    print(f"{label:20s}  efficiency = {eff:.6f}  (+{hi-eff:.6f} / -{eff-lo:.6f})")
    return {'efficiency': eff, 'lower_bound': lo, 'upper_bound': hi, 'purity': pur, 'accuracy': acc}

#after overlap removal
def compute_OR_metrics(original_graphs, survivors, dataset_name):
    # 1) Count original real graphs in the deduplicated set.
    N_real_orig = sum(1 for g in original_graphs if g.label.item() in [0, 1])
    print(f"{dataset_name} - Number of true real graphs in deduplicated test set: {N_real_orig}")

    # 2) Among the final survivors, count real and fake graphs.
    N_real_surv = sum(1 for g in survivors if g.label.item() in [0, 1])
    N_real_surv_correct = sum(1 for g in survivors if g.label.item() in [0, 1] and g.pred_label in [0, 1])
    print(f"{dataset_name} - Number of true real graphs after overlap removal: {N_real_surv}")

    N_fake_surv = sum(1 for g in survivors if g.label.item() == 2)
    N_fake_surv_incorrect = sum(1 for g in survivors if g.label.item() == 2 and g.pred_label in [0, 1])

    # 3) Compute efficiency and purity.
    final_efficiency = N_real_surv_correct / N_real_orig if N_real_orig > 0 else 0
    final_purity = N_real_surv_correct / (N_real_surv_correct + N_fake_surv_incorrect) if (N_real_surv_correct + N_fake_surv_incorrect) > 0 else 0

    print(f"{dataset_name} - N_real_surv: {N_real_surv}, N_fake_surv: {N_fake_surv}")
    print(f"{dataset_name} - Final Efficiency (Recall): {final_efficiency:.6f}")
    print(f"{dataset_name} - Final Purity (Precision): {final_purity:.6f}")
    print(f"Total graphs in {dataset_name} after overlap removal: {len(survivors)}")

    # 3b) Compute Clopper–Pearson one sigma confidence interval for the efficiency.
    alpha = 0.3173  # corresponds roughly to 1-sigma for a normal approximation
    x = N_real_surv_correct
    n = N_real_orig
    if n > 0:
        if x == 0:
            lower_bound = 0.0
        else:
            lower_bound = beta.ppf(alpha/2, x, n - x + 1)
        if x == n:
            upper_bound = 1.0
        else:
            upper_bound = beta.ppf(1 - alpha/2, x + 1, n - x)
        print(f"{dataset_name} - 1 sigma Confidence Interval for Efficiency: ({lower_bound:.6f}, {upper_bound:.6f})")
    else:
        lower_bound = 0.0
        upper_bound = 0.0
        print(f"{dataset_name} - No valid data to compute confidence intervals.")

    # 4) Compute final accuracy (optional for other metrics).
    survivor_dict = {id(g): g for g in survivors}
    correct_count = 0
    total_count = len(original_graphs)
    for g in original_graphs:
        if id(g) in survivor_dict:
            surviving_prediction = survivor_dict[id(g)].pred_label
            if (g.label.item() in [0, 1] and surviving_prediction in [0, 1]) or (g.label.item() == 2 and surviving_prediction == 2):
                correct_count += 1
    final_accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"{dataset_name} - Final Accuracy = {final_accuracy:.6f}\n")
    
    # Return the key metrics as a dictionary.
    return {
        'final_efficiency': final_efficiency,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'final_purity': final_purity,
        'final_accuracy': final_accuracy
    }

def e_separation(graphs, dataset_name, model, device):
    # Filter for real graphs (labels 0 or 1)
    real_graphs = [g for g in graphs if g.label.item() in [0, 1]]
    loader = DataLoader(real_graphs, batch_size=128, exclude_keys=['mc_pid'])
    
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = batch.label.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Compute confusion matrix and report.
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    # print(f"Confusion Matrix ({dataset_name}):")
    # print(cm)
    
    # print(f"\nClassification Report ({dataset_name}):")
    # print(classification_report(all_labels, all_preds, labels=[0,1],
    #       target_names=["e⁺ (label 0)", "e⁻ (label 1)"], zero_division=0))
    
    # Plot the confusion matrix.
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                annot_kws={"size": 16},
                xticklabels=["Predicted e⁺", "Predicted e⁻"],
                yticklabels=["Actual e⁺", "Actual e⁻"])
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f"Confusion Matrix ({dataset_name})", fontsize=16)
    plt.show()

def save_OR_mc_tids(graphs, csv_path, confidence_threshold=0.5):
    """
    From a list of post‐Overlap‐Removal graphs, pick out all *real* tracks
    (labels 0 or 1) with a valid mc_tid and pred_confidence ≥ confidence_threshold,
    then extract unique (mc_tid, frameId) pairs and write them to `csv_path`.
    """
    # 1) Filter for surviving graphs that are “real” (label 0 or 1), have a valid mc_tid,
    #    and have pred_confidence ≥ confidence_threshold.
    real_graphs = [
        g for g in graphs
        if g.label.item() in [0, 1]
        and g.mc_tid not in [0, None]
        and hasattr(g, "pred_confidence")
        and g.pred_confidence >= confidence_threshold
    ]

    # 2) Build a set of unique (mc_tid, frameId) pairs
    unique_pairs = set()
    for g in real_graphs:
        tid = g.mc_tid
        # normalize mc_tid → tuple form
        if isinstance(tid, list):
            tid = tuple(tid)
        elif isinstance(tid, int):
            tid = (tid,)
        frame = g.frameId
        unique_pairs.add((tid, frame))

    # 3) Convert to DataFrame and save
    df = pd.DataFrame(list(unique_pairs), columns=["mc_tid", "frameId"])
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} unique (mc_tid, frameId) to {csv_path}")