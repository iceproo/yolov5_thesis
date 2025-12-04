import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_metrics.py <csv_file>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Load CSV
    df = pd.read_csv(csv_path, skip_blank_lines=True)

    # Selected columns (0-indexed)
    cols = [0, 1, 2, 6, 8, 9]

    df = df.iloc[:, cols]
    df.columns = ["epoch", "box_loss", "obj_loss", "map50", "val_box_loss", "val_obj_loss"]

    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # ---- Left axis (losses) ----
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_ylim(0, 0.1)  
    ax1.plot(df["epoch"], df["box_loss"], label="train box loss")
    ax1.plot(df["epoch"], df["obj_loss"], label="train obj loss")
    ax1.plot(df["epoch"], df["val_box_loss"], label="val box loss")
    ax1.plot(df["epoch"], df["val_obj_loss"], label="val obj loss")
    ax1.grid(True)


    # ---- Right axis (mAP50) ----
    ax2 = ax1.twinx()
    ax2.set_ylabel("mAP50")
    ax2.plot(df["epoch"], df["map50"], color="red", linestyle="--", label="mAP 0.5")

    # Get last x and y
    x_last = len(df["map50"]) - 1
    y_last_list = df["map50"][x_last-50:x_last]
    y_last = np.mean(y_last_list)

    ax2.plot(df["epoch"], df["map50"], color="red", linestyle="--", label="mAP 0.5")

    # Annotate last value
    plt.annotate(
        "mean_50_last_mAP: " + str(y_last.round(decimals=2)),
        xy=(x_last, y_last),
        xytext=(x_last - 30, y_last+0.05),   # small offset so the text doesn’t overlap
        fontsize=12,
        arrowprops=dict(arrowstyle="->", lw=1)
    )
    #ax2.plot(df["epoch"], y_last_list, color="red", linestyle="--", label="mAP 0.5")

    # ---- Combine legends ----
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    save_dir_root, _ = os.path.split(csv_path)
    exp_name = save_dir_root.split('/')[-1]
    filepath_save = os.path.join(save_dir_root, "training_logs.png")
    plt.title(exp_name)
    plt.tight_layout()
    plt.savefig(filepath_save)

if __name__ == "__main__":
    main()