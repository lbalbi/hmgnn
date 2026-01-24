import numpy as np
import matplotlib.pyplot as plt

# Raw results
results = { # FPR
    "GCN": 0.9707,
    "GCN + (-)": 0.9554,
    "C-GCN": 0.7872,
    "C-GCN +NRS + (-)": 0.9547,
    "RGCN": 0.9698,
    "RGCN + (-)": 0.9858,
    "C-RGCN": 0.9682,
    "C-RGCN + (-)": 0.9496,
    "C-RGCN +NRS + (-)": 0.9803,
}


x_labels = ["GNN", "GNN + (-)", "GNN +NRS + (-)"]
x = np.arange(len(x_labels))

def get_or_nan(key):
    return results.get(key, np.nan)

# Build the 4 lines (each MUST be length 3)
series = {
    "GCN": [
        get_or_nan("GCN"),
        get_or_nan("GCN + (-)"),
        np.nan,
    ],
    "C-GCN": [
        get_or_nan("C-GCN"),
        np.nan,
        get_or_nan("C-GCN +NRS + (-)"),
    ],
    "RGCN": [
        get_or_nan("RGCN"),
        get_or_nan("RGCN + (-)"),
        np.nan,
    ],
    "C-RGCN": [
        get_or_nan("C-RGCN"),
        get_or_nan("C-RGCN + (-)"),
        get_or_nan("C-RGCN +NRS + (-)"),
    ],
}

markers = {"GCN": "o", "C-GCN": "s", "RGCN": "^", "C-RGCN": "D"}
linestyles = {"GCN": "-", "C-GCN": "--", "RGCN": "-.", "C-RGCN": ":"}
# Per-point label offsets in *data units* (same scale as FPR)
# Order matches x_labels: ["GNN", "GNN + (-)", "GNN +NRS + (-)"]
label_dy_data = {
    "GCN":    [0.003, 0.005, np.nan],
    "C-GCN":  [0.012, np.nan, 0.008],
    "RGCN":   [0.014, 0.005, np.nan],
    "C-RGCN": [-0.008, -0.008, 0.003],
}


fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)

for name, yvals in series.items():
    y = np.array(yvals, dtype=float)

    (line,) = ax.plot(
        x, y,
        marker=markers.get(name, "o"),
        linestyle=linestyles.get(name, "-"),
        linewidth=2,
        markersize=7,
        alpha=0.95,
        label=name
    )

    line_color = line.get_color()
    dy_list = np.array(label_dy_data.get(name, [0.0]*len(x)), dtype=float)

    for idx, (xi, yi) in enumerate(zip(x, y)):
        if np.isfinite(yi):
            dy = dy_list[idx] if idx < len(dy_list) else 0.0
            if not np.isfinite(dy): dy = 0.0
            ax.text(xi, yi + dy, f"{yi:.3f}", ha="center",
                va="bottom" if dy >= 0 else "top",
                fontsize=12, color=line_color, zorder=5)

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=14)
ax.set_xlim(-0.25, len(x_labels) - 1 + 0.25)

ax.set_ylabel("False Positive Rate (FPR)", fontsize=16)
ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(
    loc="lower right",          # or "upper left", "best", etc.
    fontsize=12,
    frameon=True,              # box around it
    fancybox=True,
    framealpha=0.9,            # slightly transparent
    borderpad=0.6,
    handlelength=2.2
)
# ax.legend(frameon=False, fontsize=13, ncol=4,
#          loc="upper center", bbox_to_anchor=(0.5, 1.18))
plt.savefig("fpr_lines_by_family.png", dpi=250, bbox_inches="tight")
plt.show()



# import matplotlib.pyplot as plt
# from matplotlib.ticker import PercentFormatter
# experiments = ["GCN", "GCN + (-)", "C-GCN", "C-GCN +NRS + (-)", "RGCN", "RGCN + (-)", "C-RGCN",
#                 "C-RGCN + (-)", "C-RGCN +NRS + (-)"]
# fpr = [ 0.9707, 0.9554, 0.7872, 0.9547, 0.9698, 0.9858, 0.9682, 0.9496, 0.9803]

# # Controls how far apart points are on the x-axis
# spacing = 0.1
# x = [i * spacing for i in range(len(experiments))]

# # Figure/axes
# fig, ax = plt.subplots(figsize=(13, 4.8), constrained_layout=True)

# # Line + markers (no explicit colors; uses Matplotlib defaults)
# ax.plot(x, fpr, marker="o", linewidth=2, markersize=7)

# # X axis formatting
# ax.set_xticks(x)
# ax.set_xticklabels(experiments, rotation=25, ha="right", fontsize=14)
# ax.margins(x=0.02)

# # Y axis formatting (fit to your values)
# ymin, ymax = min(fpr), max(fpr)
# pad = 0.02
# ax.set_ylim(max(0, ymin - pad), min(1, ymax + pad))
# # ax.yaxis.set_major_formatter(PercentFormatter(1.0))  # show as %

# # Labels/title
# # ax.set_xlabel("Experiment")
# ax.set_ylabel("False Positive Rate (FPR)", fontsize=16)
# # ax.set_title("FPR across experiments", pad=12, fontsize=20)

# # Grid (subtle, y-only)
# ax.set_axisbelow(True)
# ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.6)

# # Clean spines
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# # Value labels above each point
# for xi, yi in zip(x, fpr):
#     ax.annotate(f"{yi:.3f}", (xi, yi),
#                 textcoords="offset points", xytext=(0, 8),
#                 ha="center", fontsize=14)

# # Call out the best (lowest) FPR
# min_idx = fpr.index(min(fpr))

# fig.savefig("sorted_fpr_across_methods.png", dpi=250, bbox_inches="tight")
# plt.show()



# import matplotlib.pyplot as plt

# experiments = ["RGCN + (-)", "C-RGCN +NRS + (-)", "GCN", "RGCN", "C-RGCN",
#                "GCN + (-)", "C-RGCN + (-)", "C-GCN"]
# fpr = [0.9858, 0.9803, 0.9707, 0.9698, 0.9682, 0.9554, 0.9496, 0.7872]

# # Increase this to spread points farther apart
# spacing = 0.5
# x = [i * spacing for i in range(len(experiments))]

# plt.figure(figsize=(12, 4))  # wider figure also helps
# plt.plot(x, fpr, marker="o")

# plt.xticks(x, experiments, rotation=30, ha="right")
# plt.xlabel("Experiment / Method")
# plt.ylabel("False Positive Rate (FPR)")
# plt.title("FPR across methods")
# plt.ylim(0.7, 1)
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("sorted_fpr_across_methods.png", dpi=200, bbox_inches="tight")
# plt.show()


# import matplotlib.pyplot as plt

# # experiments = ["GCN", "GCN + (-)", "C-GCN", "RGCN", "RGCN + (-)", "C-RGCN", "C-RGCN + (-)", "C-RGCN +NRS + (-)"]  # x-axis labels
# # fpr = [0.9707, 0.9554, 0.7872, 0.9698, 0.9858, 0.9682, 0.9496, 0.9803]  # y-axis values

# experiments = ["RGCN + (-)", "C-RGCN +NRS + (-)", "GCN", "RGCN", "C-RGCN", "GCN + (-)", "C-RGCN + (-)", "C-GCN"]
# fpr = [0.9858,0.9803,0.9707,0.9698,0.9682,0.9554,0.9496,0.7872]

# plt.figure()
# plt.plot(experiments, fpr, marker="o")
# plt.xlabel("Experiment / Method")
# plt.ylabel("False Positive Rate (FPR)")
# plt.title("FPR across methods")
# plt.ylim(0.7, 1)
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# plt.savefig("sorted_fpr_across_methods.png", dpi=200, bbox_inches="tight")