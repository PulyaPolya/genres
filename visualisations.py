import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from matplotlib import cm
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import math
from matplotlib.patches import Patch


def get_tsne_plots(spectrograms_scaled):
    X_2d = TSNE(n_components=2, random_state=42).fit_transform(spectrograms_scaled)
    #X_2d = spectrograms_reduced.copy()

    # Ensure labels are 0..K-1 and consistent
    unique_labels = np.unique(true_labels_enc)
    K = len(unique_labels)

    # Color map with exactly K discrete colors (stable across panels)
    cmap_name = "tab20"
    cmap_k = cm.get_cmap(cmap_name, K)

    # Build handles for a single shared legend (use true-label counts)
    counts = [(true_labels_enc == lab).sum() for lab in unique_labels]
    handles = [
        Line2D([0],[0], marker='o', linestyle='', markersize=8,
            markerfacecolor=cmap_k(i), markeredgecolor='none',
            label=f"{label_names[i]} ")
        for i in range(K)
    ]
    matplotlib.rcParams.update({'font.size': 11})
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharex=True, sharey=True)

    # Left: clustered result (use same discrete cmap)
    axes[0].scatter(X_2d[:, 0], X_2d[:, 1],
                    c=predicted_labels, cmap=cmap_k, s=18, alpha=0.8)
    axes[0].set_title("Predicted labels")

    # Right: true labels
    axes[1].scatter(X_2d[:, 0], X_2d[:, 1],
                    c=true_labels_enc, cmap=cmap_k, s=18, alpha=0.9)
    axes[1].set_title("True labels")

    # Clean axes
    for ax, tag in [(axes[0], "(a)"), (axes[1], "(b)")]:
        #ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.text(0.02, 0.98, tag, transform=ax.transAxes,
                ha="left", va="top", fontsize=10, fontweight="bold")

    # One shared legend above both plots, multi-column
    leg = fig.legend(handles=handles, loc="upper center", ncol=7,
                    bbox_to_anchor=(0.5, 0), frameon=True, title="Genres")
    #leg._legend_box.align = "left"
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")

    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave room for legend
    #plt.savefig("figures/plot_k-means_52.pdf", format="pdf", bbox_inches="tight")
    plt.show()



def get_pie_chart_viz(cluster_counts, layer, gtzan=True, save=False, save_suffix=""):
    gtzan_genres  = {'Disco', 'Metal', 'Hip-Hop', 'Pop','Reggae', 'Rock', 'Country', 'Blues','Jazz', "Classical" }
    all_genres_mapped = sorted({
        'Blues', 'Alternative', 'Hip-Hop', 'World', "Unknown",
        'R&B', 'Country', 'Jazz', 'Reggae', 'Soul', 'Disco', 'HJDB',
        'Metal', 'Electronic', 'Rock', 'Classical', 'Pop', 'Folk', 'Funk','Groove'
    })

    cmap = plt.get_cmap("tab20")
    colors = {genre: cmap(i % 20) for i, genre in enumerate(all_genres_mapped)}
    colors["Unknown"] = "#bbbfbd"
    alpha, hatch_color = 0.7, "#777777"

    def make_autopct(values):
        total = sum(values)
        top_idx = set(sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:4])
        top_idx = {idx for idx in top_idx if values[idx] > 0.14 * total}
        def autopct(pct):
            autopct.current += 1
            i = autopct.current
            return f"{pct:.1f}%" if i in top_idx else ""
        autopct.current = -1
        return autopct

    n = len(cluster_counts)
    n_cols = 2
    n_rows = math.ceil(n / n_cols)

    fig = plt.figure(figsize=(7, 6.5))
    gs = fig.add_gridspec(n_rows, n_cols)

    items = list(cluster_counts.items())

    full_rows = n // 2  # number of rows with 2 clusters
    idx = 0
    for r in range(full_rows):
        for c in range(2):
            cluster, counts = items[idx]
            ax = fig.add_subplot(gs[r, c])

            values = list(counts.values())
            keys = list(counts.keys())
            autopct = make_autopct(values)

            wedges, _, _ = ax.pie(
                values,
                labels=None,
                colors=[colors[g] for g in keys],
                autopct=autopct,
                startangle=90
            )

            # style Unknown genres
            for wedge, genre in zip(wedges, keys):
                if genre == "Unknown":
                    wedge.set_hatch("//")
                    wedge.set_edgecolor(hatch_color)
                    wedge.set_linewidth(0.6)
                    wedge.set_alpha(alpha)

            ax.set_title(f"Cluster {idx + 1} ({sum(values)})")
            idx += 1

    if n % 2 == 1:
        cluster, counts = items[idx]
        ax = fig.add_subplot(gs[n_rows - 1, :])  # span both columns

        values = list(counts.values())
        keys = list(counts.keys())
        autopct = make_autopct(values)

        wedges, _, _ = ax.pie(
            values,
            labels=None,
            colors=[colors[g] for g in keys],
            autopct=autopct,
            startangle=90
        )

        for wedge, genre in zip(wedges, keys):
            if genre == "Unknown":
                wedge.set_hatch("//")
                wedge.set_edgecolor(hatch_color)
                wedge.set_linewidth(0.6)
                wedge.set_alpha(alpha)

        ax.set_title(f"Cluster {idx + 1} ({sum(values)})")

    # --- legend ---
    legend_handles = []
    for g in all_genres_mapped:
        if ((gtzan and g in gtzan_genres) or (not gtzan)):
            if g == "Unknown":
                legend_handles.append(Patch(facecolor=colors[g], hatch="//",
                                            alpha=alpha, edgecolor=hatch_color, label=g))
            else:
                legend_handles.append(Patch(facecolor=colors[g], label=g))

    fig.legend(handles=legend_handles, loc="right", ncol=1, fontsize=10, bbox_to_anchor=(1.1, 0.5))

    if save:
        save_suffix += "gtzan" if gtzan else "full"
        plt.savefig(f"figures/pie_chart_layer_{layer}_{save_suffix}.pdf", format="pdf", bbox_inches="tight")

    plt.show()


def beat_this_scratch_run():
    df = pd.read_csv("csv/wandb_scratch_continue_figures_classical.csv")
    columns = [col for col in df.columns.values if ("val_F" in col) & ("MAX" not in col) & ("MIN" not in col)] + ["epoch"]
    df = df[columns]
    df.head(10)
    df_interp = df.interpolate()
    baseline_acc = df.loc[0, 'cluster_2_resume0 - val_F-measure_beat']

    plt.figure(figsize=(8, 4))
    # plt.plot(df_interp['epoch'], df_interp['cluster_2_resume0 - val_F-measure_beat'],
    #           label='continue')
    plt.plot(df_interp['epoch'], df_interp['cluster_2_from_scratch - val_F-measure_beat'],
            label='From scratch', color = "green")
    plt.hlines(y = baseline_acc, xmin=0, xmax = max(df["epoch"]), label = "Pretrained baseline",  linestyle='--')
    #plt.title("Validation F-measure Beat vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")
    plt.legend(loc = "lower right")
    plt.ylim(0,1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/bad_example_beat_grid.pdf", format="pdf", bbox_inches="tight")

    plt.show()