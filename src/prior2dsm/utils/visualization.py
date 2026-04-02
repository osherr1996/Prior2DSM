from pathlib import Path
import json
import matplotlib.pyplot as plt

def save_summary_figure(rgb_cpu, final_dsm, loss_hist, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Input RGB")
    plt.imshow(rgb_cpu.permute(1, 2, 0).cpu())

    plt.subplot(1, 3, 2)
    plt.title("Predicted DSM")
    plt.imshow(final_dsm.cpu().numpy(), cmap="terrain")
    plt.colorbar(fraction=0.046)

    plt.subplot(1, 3, 3)
    plt.title("TTO Huber Loss")
    plt.plot(loss_hist)
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

def save_metrics(metrics: dict, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
