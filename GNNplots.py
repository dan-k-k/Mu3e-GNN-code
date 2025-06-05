# GNNplots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_efficiency_vs_feature_step(
    df_signal, df_IC, df_michel, df_beam,
    feature,                          # e.g. "mc_p", "mc_pt", "mc_phi", or "mc_lam"
    x_label,                          # e.g. r"$p_{\mathrm{true}}$ [MeV]", r"$\phi_{\mathrm{true}}$ [rad]", etc.
    label_signal="Signal",
    label_IC="I.C.",
    label_michel="Michel",
    label_beam="Beam",
    bins=20
):
    """
    Plots “efficiency vs. <feature>” for four datasets (signal / IC / michel / beam).

    - feature: string, name of the DataFrame column to bin (must exist in each df).
    - x_label: string for the x‐axis label (LaTeX or plain).
    - bins: number of bins (default 20).
    """
    # 1) Keep only true‐real tracks (label ∈ {0,1}) in each DataFrame
    real_sig    = df_signal[   df_signal['label'].isin([0, 1]) ]
    real_IC     = df_IC[       df_IC['label'].isin([0, 1]) ]
    real_michel = df_michel[   df_michel['label'].isin([0, 1]) ]
    real_beam   = df_beam[     df_beam['label'].isin([0, 1]) ]

    # 2) Build a common set of edges that covers all four datasets
    combined_series = pd.concat([
        real_sig[feature],
        real_IC[feature],
        real_michel[feature],
        real_beam[feature]
    ]).dropna()

    if combined_series.empty:
        print("No real tracks found in any of the four datasets.")
        return

    edges = np.linspace(combined_series.min(), combined_series.max(), bins + 1)

    # 3) Helper: given one “real” df, compute the (x‐coords, y‐coords) for a step plot
    def _step_data(df_real):
        if df_real.empty:
            return None, None

        # digitize into 0..bins−1
        idx = np.digitize(df_real[feature], edges) - 1
        idx[idx == bins] = bins - 1

        # sums of “correct” flags in each bin
        sums   = np.bincount(idx, weights=df_real['correct'], minlength=bins)
        counts = np.bincount(idx,                          minlength=bins)

        eff = np.divide(sums, counts, out=np.full(bins, np.nan), where=counts > 0)

        # build the “step” style x–y arrays
        x = np.repeat(edges, 2)[1:-1]
        y = np.repeat(eff,   2)
        return x, y

    sx_sig, sy_sig         = _step_data(real_sig)
    sx_IC,  sy_IC          = _step_data(real_IC)
    sx_mi,  sy_mi          = _step_data(real_michel)
    sx_be,  sy_be          = _step_data(real_beam)

    # 4) Plot
    plt.figure(figsize=(4.5, 3))

    if sx_sig is not None:
        plt.step(sx_sig, sy_sig, where='pre', color='blue',   ls='-',  label=label_signal)
    if sx_IC is not None:
        plt.step(sx_IC,  sy_IC,  where='pre', color='orange', ls='--', label=label_IC)
    if sx_mi is not None:
        plt.step(sx_mi,  sy_mi,  where='pre', color='red',    ls=':',  label=label_michel)
    if sx_be is not None:
        plt.step(sx_be,  sy_be,  where='pre', color='green',  ls='-.', label=label_beam)

    plt.xlabel(x_label, fontsize=14)
    plt.ylabel("Efficiency", fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(edges[0], edges[-1])
    plt.ylim(0.895, 1.005)         # adjust vertical limits if needed
    plt.legend(fontsize=11)
    plt.show()

def plot_purity_vs_graphcount_multi(
    dfs, labels, colors, linestyles,
    max_graphs=800, bin_width=40
):
    """
    Overlayed step‐plots of average purity vs. #graphs/frame,
    grouping counts into bins of width `bin_width`.
    """
    plt.figure(figsize=(4.5,3))
    
    # 1) Define the half‐shifted edges for the step plot:
    edges = np.arange(0.5, max_graphs + bin_width + 0.5, bin_width)
    # 2) Compute the bin‐centres for plotting & ticks
    centers = edges[:-1] + bin_width/2

    for df, label, c, ls in zip(dfs, labels, colors, linestyles):
        by_frame = (
            df.groupby('frameId')
              .agg(TP=('TP','sum'),
                   FP=('FP','sum'),
                   n_graphs=('frameId','size'))
              .reset_index()
        )
        by_frame['purity'] = by_frame['TP'] / (by_frame['TP'] + by_frame['FP'])
        by_frame = by_frame.dropna(subset=['purity'])

        # cap counts > max_graphs
        by_frame.loc[by_frame['n_graphs'] > max_graphs, 'n_graphs'] = max_graphs

        # assign to bin index
        bin_idx = np.floor((by_frame['n_graphs'] - 1) / bin_width).astype(int)
        by_frame['bin'] = bin_idx

        # average purity per bin
        summary = (
            by_frame.groupby('bin')['purity']
                    .mean()
                    .reindex(range(len(centers)), fill_value=np.nan)
        )

        # build step coords
        xs = np.repeat(edges, 2)[1:-1]
        ys = np.repeat(summary.values, 2)

        plt.step(xs, ys, where='pre', color=c, linestyle=ls, label=label)

    plt.xlabel("Number of Graphs per Frame", fontsize=14)
    plt.ylabel("Average Purity",              fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.xlim(0.5, max_graphs + 0.5)

    # only label every other bin centre
    tick_centres = centers[::2]
    tick_labels  = [f"{int(c)}" for c in tick_centres]
    plt.xticks(tick_centres, tick_labels, rotation=45)

    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11, loc="lower right", bbox_to_anchor=(1.33,0))
    plt.show()

def plot_purity_vs_hitcount_multi(
    class_dfs, hits_dfs, labels, colors, linestyles,
    start_hit=6, max_hits=20, michel_max=6
):
    """
    Step‐plots of average purity vs. #unique hits per frame,
    for each dataset in class_dfs/hits_dfs.
    Michel is forced to purity=1.0 for frames with ≤ michel_max hits.
    """
    plt.figure(figsize=(4.5, 3))

    # Integer‐bin edges from start_hit–0.5 up to max_hits+0.5
    edges = np.arange(start_hit - 0.5, max_hits + 1.5, 1.0)
    bins  = list(range(start_hit, max_hits + 1))

    for cls_df, hits_df, label, c, ls in zip(
        class_dfs, hits_dfs, labels, colors, linestyles
    ):
        if label == "Michel":
            # Pure 1.0 up to michel_max, then NaN
            purity_vals = [1.0 if h <= michel_max else np.nan for h in bins]
            summary = pd.Series(purity_vals, index=bins)
        else:
            # 1) Compute per‐frame purity
            by_frame = (
                cls_df
                  .groupby('frameId')
                  .agg(TP=('TP','sum'), FP=('FP','sum'))
                  .assign(purity=lambda df: df.TP / (df.TP + df.FP))
                  .dropna(subset=['purity'])
                  .reset_index()
            )

            # 2) Count unique hits per frame (dedupe on x,y,z)
            hit_counts = (
                hits_df
                  .drop_duplicates(subset=['frameId','x','y','z'])
                  .groupby('frameId').size()
                  .rename('n_hits')
                  .reset_index()
            )

            # 3) Merge and restrict to ≤ max_hits
            merged = by_frame.merge(hit_counts, on='frameId', how='inner')
            merged = merged[merged['n_hits'] <= max_hits]

            # 4) Compute average purity in each integer‐hit bin
            summary = (
                merged
                  .groupby('n_hits')['purity']
                  .mean()
                  .reindex(bins, fill_value=np.nan)
            )

        # 5) Build step‐plot coords
        xs = np.repeat(edges, 2)[1:-1]
        ys = np.repeat(summary.values, 2)
        plt.step(xs, ys, where='pre', color=c, linestyle=ls, label=label)

    plt.xlabel("Number of Hits per Frame", fontsize=14)
    plt.ylabel("Average Purity",            fontsize=14)
    plt.xticks(bins)
    plt.xlim(start_hit - 0.5, max_hits + 0.5)
    plt.ylim(0.895, 1.005)
    plt.tick_params(axis='both', labelsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc="lower left")
    plt.tight_layout()
    plt.show()

def plot_confidence_histograms_multi(dataset_label, elec_results, pos_results, fake_results):
    """
    Plots a histogram (with log-scaled density) for electrons, positrons, and fakes
    for a given dataset.
    
    For fakes, I plot 1 - predicted_probability (as in your original code).
    """
    # Convert outputs to arrays
    e_y_true  = np.array(elec_results['y_true'])
    e_y_probs = np.array(elec_results['y_probs'])
    p_y_true  = np.array(pos_results['y_true'])
    p_y_probs = np.array(pos_results['y_probs'])
    
    # For each class, select only those predictions for which the sample is truly of that class
    e_conf = e_y_probs[e_y_true == 1]
    p_conf = p_y_probs[p_y_true == 1]
    
    # Define bins
    bin_edges   = np.linspace(0, 1, 41)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    width       = bin_edges[1] - bin_edges[0]
    
    plt.figure(figsize=(5, 3.2))
    
    # Plot electrons as a filled histogram
    e_hist, _ = np.histogram(e_conf, bins=bin_edges, density=True)
    plt.scatter(bin_centers, e_hist, marker='o', s=10, color='blue', label=f'{dataset_label}Electrons') # – e⁻
    # Poisson error bars for electrons
    counts_e, _  = np.histogram(e_conf, bins=bin_edges)
    density_e    = counts_e / (counts_e.sum() * width)
    errors_e     = np.sqrt(counts_e) / (counts_e.sum() * width)
    plt.errorbar(bin_centers, density_e, yerr=errors_e, fmt='none', ecolor='blue', capsize=3)
    
    # Plot positrons as scatter points (using a histogram to compute density)
    p_hist, _ = np.histogram(p_conf, bins=bin_edges, density=True)
    plt.scatter(bin_centers, p_hist, marker='o', s=10, color='red', label=f'{dataset_label}Positrons') #  – e⁺
    # Poisson error bars for positrons
    counts_p, _ = np.histogram(p_conf, bins=bin_edges)
    density_p   = counts_p / (counts_p.sum() * width)
    errors_p    = np.sqrt(counts_p) / (counts_p.sum() * width)
    plt.errorbar(bin_centers, density_p, yerr=errors_p, fmt='none', ecolor='red', capsize=3)
    
    # Only process fakes if fake_results is provided
    if fake_results is not None:
        f_y_true  = np.array(fake_results['y_true'])
        f_y_probs = np.array(fake_results['y_probs'])
        f_conf    = 1 - f_y_probs[f_y_true == 1]
        plt.hist(f_conf, bins=bin_edges, label=f'{dataset_label}Fakes', histtype='step', 
                 linestyle='-', linewidth=1.4, color='black', density=True, log=True)
        # Poisson error bars for fakes
        counts_f, _  = np.histogram(f_conf, bins=bin_edges)
        density_f    = counts_f / (counts_f.sum() * width)
        errors_f     = np.sqrt(counts_f) / (counts_f.sum() * width)
        plt.errorbar(bin_centers, density_f, yerr=errors_f, fmt='none', ecolor='black', capsize=3)
    
    plt.xlabel('GNN output', fontsize=14)
    plt.ylabel('Frequency Density', fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.ylim(bottom=2e-3, top=8e1)
    # plt.xlim(0.955, 1.005)
    plt.legend(loc='upper center', prop={'size': 11}, framealpha=0.5)
    plt.grid(True)
    plt.show()

def plot_class_histograms_with_errors(
    dataset_labels,
    results_list,
    axis_label="GNN output",
    y_label="Frequency Density",
    color_list=None,
    style_list=None,
    bins=601,
    bin_range=(0, 1),
    xlim=(0.965, 1.001),
    ylim=(2e-3, 8.6e1)
):
    """
    Plots step‐histograms for the 'true' class confidence (y_true==1) from multiple datasets,
    adding Poisson error bars to each curve.
    
    Parameters:
      dataset_labels (list of str): Labels for each dataset.
      results_list (list of dict): Each dict must contain:
          - 'y_true': array‐like of 0/1 (1 if sample belongs to the class)
          - 'y_probs': array‐like of predicted probability for that class
      axis_label (str): Label for the x‐axis.
      y_label (str): Label for the y‐axis.
      color_list (list): Colors for each dataset.
      style_list (list): Line styles for each dataset.
      bins (int): Number of bins.
      bin_range (tuple): (min, max) range for bins.
      xlim (tuple): x‐axis limits.
      ylim (tuple): y‐axis limits.
    """
    # Prepare common bin edges and centers
    bin_edges   = np.linspace(bin_range[0], bin_range[1], bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width       = bin_edges[1] - bin_edges[0]

    plt.figure(figsize=(5, 3.2))
    for i, res in enumerate(results_list):
        y_true  = np.asarray(res['y_true'])
        y_probs = np.asarray(res['y_probs'])
        # select only true‐class samples
        conf = y_probs[y_true == 1]

        # compute histogram counts, density and Poisson errors
        counts, _ = np.histogram(conf, bins=bin_edges)
        total_counts = counts.sum()
        if total_counts > 0:
            density   = counts / (counts.sum() * width)
            errors    = np.sqrt(counts) / (counts.sum() * width)
        else:
            # no true‐class events → zero density & zero error
            density = np.zeros_like(counts, dtype=float)
            errors  = np.zeros_like(counts, dtype=float)

        # extend density so the final bin is drawn
        density_extended = np.concatenate([density, [density[-1]]])

        color = color_list[i] if color_list is not None else None
        ls    = style_list[i] if style_list is not None else '-'
        plt.step(
            bin_edges,
            density_extended,
            where='post',
            color=color,
            linestyle=ls,
            linewidth=1.4,
            label=dataset_labels[i]
        )
        plt.errorbar(
            bin_centers,
            density,
            yerr=errors,
            fmt='none',
            ecolor=color,
            capsize=2
        )

    plt.xlabel(axis_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend(loc='upper left', fontsize=11, framealpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def compute_purity(y_true, y_probs, thresholds):
    """Compute purity (precision) at each threshold."""
    purity = []
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        if TP + FP > 0:
            purity.append(TP / (TP + FP))
        else:
            purity.append(np.nan)
    return np.array(purity)

def compute_efficiency(y_true, y_probs, thresholds):
    """Compute efficiency (TPR) at each threshold."""
    eff = []
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        eff.append(TP / (TP + FN) if (TP + FN) > 0 else np.nan)
    return np.array(eff)
