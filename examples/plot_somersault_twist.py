from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

fs = 120
base_path = Path("/Users/mickaelbegon/Documents/Playground/output/1_partie_0429/reconstructions/")

# Récupérer tous les dossiers triés alphabétiquement
all_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
print(all_dirs)

# Sélectionner les dossiers par index (1-based → -1)
labels = ["EKF_2D[1]", "EKF_2D[0]", "EKF_3D[1]", "EKF_3D[0]", "Triang.[1]", "Triang.[0]", "Triang. [once]"]

selected_indices = [1, 2, 5, 6, 7, 8, 9]
folders = [all_dirs[i - 1] for i in selected_indices]

colors = plt.cm.tab10.colors
linestyles = ["-", "--", "-.", ":", "-", "--","-"]
markers = ["o", "s", "^", "d", "x", "+","."]

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)


for i, folder_path in enumerate(folders):
    print(folder_path)
    print(i)
    files = list(folder_path.glob("*.npz"))
    data = None

    for f in files:
        try:
            d = np.load(f)
            if "q" in d:
                q_tmp = d["q"]
                if q_tmp is not None and q_tmp.size > 0 and q_tmp.ndim == 2:
                    data = d
                    break
        except Exception:
            continue

    if data is None:
        continue

    q = data["q"]

    n_frames = q.shape[0]

    markevery_base = max(n_frames // 20, 1)
    offset = i * 10 % markevery_base

    frames = np.arange(n_frames) / fs  # temps en frames (120 Hz)

    # Conversion en rotations
    q4 = np.unwrap(q[:, 3]) / (2 * np.pi)
    q6 = np.unwrap(q[:, 5]) / (2 * np.pi)

    indices_nan = np.argwhere(np.isnan(q6))
    print(indices_nan)

    for qi in [q4, q6]:
        nan_mask = np.isnan(qi)

        if nan_mask.all():
            continue

        if nan_mask.any():
            indices = np.arange(len(qi))
            valid_mask = ~nan_mask
            qi[nan_mask] = np.interp(indices[nan_mask], indices[valid_mask], qi[valid_mask])



    style = dict(
        color=colors[i % len(colors)],
        linestyle=linestyles[i],
        marker=markers[i],
        markevery=(offset, markevery_base),
        linewidth=2,
        alpha=0.9,
    )

    # label = f"{selected_indices[i]}"  # numéro demandé

    t_start = 5.5
    t_end = 8.0

    i_start = int(t_start * fs)  # 660
    i_end = int(t_end * fs)  # 960

    print(q6[-1])

    axes[0].plot(frames[i_start:i_end], q4[i_start:i_end], label=labels[i], **style)
    axes[1].plot(frames[i_start:i_end], q6[i_start:i_end], label=labels[i], **style)

# Labels
axes[0].set_ylabel("Somersault [rotation]")
axes[1].set_ylabel("Twist [rotation]")
axes[1].set_xlabel("Time (s)")

# axes[0].legend(title="Folder index", ncol=3)
axes[0].grid(True)

axes[1].grid(True)
axes[0].legend(loc="upper right")

plt.rcParams.update(
    {
        "font.size": 30,
        "axes.titlesize": 30,
        "axes.labelsize": 30,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "legend.fontsize": 30,
        "legend.title_fontsize": 30,
    }
)
plt.tight_layout()
plt.show()
