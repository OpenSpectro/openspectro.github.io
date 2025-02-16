import matplotlib.pyplot as plt
import numpy as np

import os
import re
import glob
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """
    Sets random seed for reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_biomarker_name(full_filename):
    """
    Extracts the portion of the biomarker name before the '20xx' part.
    For example:
        Bilirubin-2025-02-08-22-45-33.csv  ->  Bilirubin
        Glucose_CRS-2025-02-14-13-42-22.csv -> Glucose_CRS
    """
    base_name = os.path.splitext(os.path.basename(full_filename))[0]
    match = re.match(r"(.*?)-20\d{2}", base_name)
    if match:
        return match.group(1)
    else:
        return base_name


def min_max_normalize_per_sample(data_tensor):
    mins = data_tensor.amin(dim=1, keepdim=True)
    maxs = data_tensor.amax(dim=1, keepdim=True)
    ranges = (maxs - mins).clamp(min=1e-8)
    return (data_tensor - mins) / ranges

def load_biomarker_data_2d(data_dir):
    excluded_biomarkers = [
        ""
    ]

    sample_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    sample_files = [
        f for f in sample_files
        if os.path.splitext(os.path.basename(f))[0] not in excluded_biomarkers
    ]

    all_biomarkers = []
    filenames = []
    for sample_file in sample_files:
        short_name = parse_biomarker_name(sample_file)
        filenames.append(short_name)

        sample_df = pd.read_csv(sample_file, header=None)
        spectrometer_wavelengths = sample_df.iloc[0].values
        intensities = sample_df.iloc[1:].values

        laser_wavelengths = np.arange(380, 1101)
        diagonal_indices = [
            np.argmin(np.abs(spectrometer_wavelengths - lw))
            for lw in laser_wavelengths
        ]

        diagonal_values = intensities[np.arange(721), diagonal_indices]
        all_biomarkers.append(diagonal_values)

    data_array = np.stack(all_biomarkers, axis=0)
    data_tensor = torch.from_numpy(data_array).float()
    data_tensor[data_tensor < 0.3] = 0
    data_tensor = min_max_normalize_per_sample(data_tensor)
    return data_tensor, laser_wavelengths, filenames


class SpectralAttention2D(nn.Module):
    def __init__(self, num_biomarkers: int, M: int):
        super().__init__()
        self.attention_logits = nn.Parameter(torch.randn(num_biomarkers, M) * 0.01)

    def forward(self):
        return torch.sigmoid(self.attention_logits)

    def sum_attention(self):
        return self.forward().sum()


def spectral_2d_loss(
    attention: torch.Tensor,
    data: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    max_sum: float = 5.0,
    penalty_weight: float = 1.0,
    return_per_biomarker: bool = False
):
    same_sum = (attention * data).sum()
    total_sum = (attention.sum(dim=0) * data.sum(dim=0)).sum()
    beta_term = total_sum - same_sum

    loss = - (alpha * same_sum - beta * beta_term)

    sum_attention = attention.sum()
    penalty = 0.0
    if sum_attention > max_sum:
        penalty = penalty_weight * (sum_attention - max_sum) ** 2
        loss += penalty

    if not return_per_biomarker:
        return loss

    N, M = data.shape
    per_biomarker_losses = []
    same_sum_list = []
    partial_total_sum_list = []
    beta_term_list = []

    penalty_per_biomarker = penalty / N

    for i in range(N):
        same_sum_i = (attention[i] * data[i]).sum()

        partial_total_sum_i = (attention.sum(dim=0) * data[i]).sum()
        beta_term_i = partial_total_sum_i - same_sum_i

        partial_loss_i = - (alpha * same_sum_i - beta * beta_term_i)
        partial_loss_i += penalty_per_biomarker  # distribute penalty

        per_biomarker_losses.append(partial_loss_i)
        same_sum_list.append(same_sum_i)
        partial_total_sum_list.append(partial_total_sum_i)
        beta_term_list.append(beta_term_i)

    per_biomarker_losses = torch.stack(per_biomarker_losses)
    same_sum_list = torch.stack(same_sum_list)
    partial_total_sum_list = torch.stack(partial_total_sum_list)
    beta_term_list = torch.stack(beta_term_list)

    return loss, per_biomarker_losses, same_sum_list, partial_total_sum_list, beta_term_list


def train_spectral_attention_2d(
    data_tensor,
    alpha=1.0,
    beta=0.5,
    max_sum=5.0,
    penalty_weight=1.0,
    lr=1e-2,
    epochs=100,
    device="cpu",
    checkpoint_path="spectral_attention_2d.pth"
):
    data_tensor = data_tensor.to(device)
    N, M = data_tensor.shape
    model = SpectralAttention2D(N, M).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    for epoch in range(epochs):
        optimizer.zero_grad()
        attention = model()

        loss = spectral_2d_loss(
            attention,
            data_tensor,
            alpha=alpha,
            beta=beta,
            max_sum=max_sum,
            penalty_weight=penalty_weight,
            return_per_biomarker=False
        )
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), checkpoint_path)

        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

    model.load_state_dict(torch.load(checkpoint_path))

    with torch.no_grad():
        attention_weights = model().cpu()
        total_loss, per_biomarker_losses, same_sum_list, partial_total_sum_list, beta_term_list = spectral_2d_loss(
            attention_weights.to(device),
            data_tensor.to(device),
            alpha=alpha,
            beta=beta,
            max_sum=max_sum,
            penalty_weight=penalty_weight,
            return_per_biomarker=True
        )

    print(f"\n=== Final Total Loss Across All Biomarkers: {total_loss.item():.4f} ===\n")
    print("=== Per-Biomarker Partial Loss ===")

    for i, filename in enumerate(filenames):
        print(f"[{i}] {filename}:")
        print(f"    same_sum_i:          {same_sum_list[i].item():.4f}")
        print(f"    partial_total_sum_i: {partial_total_sum_list[i].item():.4f}")
        print(f"    beta_term_i:         {beta_term_list[i].item():.4f}")
        print(f"    partial_loss_i:      {per_biomarker_losses[i].item():.4f}")

    sum_of_partials = per_biomarker_losses.sum().item()
    print(f"\nSum of partial losses = {sum_of_partials:.4f} (should match {total_loss.item():.4f})")

    return model

def pick_top_k_with_min_gap(a, k=5, min_gap=10):
    sorted_indices = np.argsort(-a)
    selected = []
    for idx in sorted_indices:
        if all(abs(idx - s) >= min_gap for s in selected):
            selected.append(idx)
            if len(selected) == k:
                break
    return selected


def plot_all_biomarkers_on_same_axis(data_tensor, attention_weights, wavelengths, filenames,
                                     output_path="all_biomarkers_same_axis.png",
                                     min_gap=20):
    """
    Plots all biomarkers (N of them) on the same figure and the same axes.
    Each biomarker has:
      - Intensity curve in a distinct color.
      - The top 5 attention peaks highlighted on a shared secondary y-axis.

    A single figure is saved to output_path.
    """

    color_list = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors) + list(plt.cm.Paired.colors)
    marker_list = ["o", "s", "^", "D", "v", "x", "P", "*", "h", "8"]

    # Convert to numpy
    data_np = data_tensor.cpu().numpy() if hasattr(data_tensor, 'cpu') else data_tensor
    attn_np = attention_weights.cpu().numpy() if hasattr(attention_weights, 'cpu') else attention_weights

    mask = (430 <= wavelengths) & (wavelengths <= 600)
    masked_wavelengths = wavelengths[mask]

    fig, ax1 = plt.subplots(figsize=(25, 10)) 
    ax2 = ax1.twinx() 


    for i in range(data_np.shape[0]):
        # Apply mask
        masked_intensity = data_np[i][mask]
        masked_attention = attn_np[i][mask]
        attn_min = masked_attention.min()
        attn_max = masked_attention.max()
        attn_range = attn_max - attn_min
        if attn_range < 1e-8:
            attn_range = 1e-8
        masked_attention = (masked_attention - attn_min) / attn_range

        top5_indices = pick_top_k_with_min_gap(masked_attention, k=5, min_gap=min_gap)

        color_i = color_list[i % len(color_list)]
        marker_i = marker_list[i % len(marker_list)]

        ax1.plot(masked_wavelengths, masked_intensity,
                 color=color_i, label=f"{filenames[i]}")

        peak_x = masked_wavelengths[top5_indices]
        peak_y = masked_attention[top5_indices]

        ax2.scatter(
            peak_x,
            peak_y,
            color=color_i,
            marker=marker_i,
            s=80,
        )

        for px, py in zip(peak_x, peak_y):
            ax2.annotate(
                f"{int(px)} nm",
                xy=(px, py),
                xytext=(0, 10),
                textcoords="offset points",
                ha='center',
                color=color_i,
                fontsize=25 
            )

    ax1.set_xlabel('Wavelength (nm)', fontsize=30, labelpad=15) 
    ax1.set_ylabel('Normalized Absorbance', color='k', fontsize=30, labelpad=15)  
    ax2.set_ylabel('Attention Weight', color='k', fontsize=30, labelpad=15)  

    ax1.set_ylim(0, 1.2)  
    ax2.set_ylim(0, 1.2)  
    ax1.tick_params(axis='both', which='major', labelsize=25)  
    ax2.tick_params(axis='both', which='major', labelsize=25) 

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        h1, l1,
        loc='lower right',
        fontsize=15 
    )

    plt.title("Comparison of Biomarkers", fontsize=30, pad=20) 

    plt.tight_layout()

    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    set_seed(19)

    data_dir = "sample"

    data_tensor, laser_wavelengths, filenames = load_biomarker_data_2d(data_dir)
    print("Data shape:", data_tensor.shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_spectral_attention_2d(
        data_tensor,
        alpha=1.0,
        beta=0.5,
        max_sum=500.0,
        penalty_weight=1.0,
        lr=1e-2,
        device=device,
        epochs=1000
    )

    with torch.no_grad():
        attention_weights = model().cpu()

    output_plot_path = os.path.join("submission_biomarkers.png")
    plot_all_biomarkers_on_same_axis(
        data_tensor, attention_weights,
        laser_wavelengths, filenames,
        output_path=output_plot_path,
        min_gap=10
    )
