# 🧬 Spectral Attention on Biomarker Spectroscopy Data

This project implements a **Spectral Attention model** for analyzing 2D spectral data from biomarker measurements. It extracts meaningful patterns from the spectral signatures and visually highlights regions (wavelengths) that are most important for each biomarker.

---

## 📁 Project Structure

```
.
├── sample/                         # Directory containing biomarker CSV files
├── submission_biomarkers.png      # Output image showing attention on spectra
└── main.py                        # Main script (code shown below)
```

---

## 🧪 What It Does

- **Loads** 2D absorbance spectra data (CSV files) for different biomarkers.
- **Extracts** diagonal laser-wavelength-related values.
- **Normalizes** the values per biomarker.
- **Trains** a spectral attention mechanism to learn important wavelengths.
- **Visualizes** the top attention weights on each biomarker's curve.

---

## 🔧 Setup

```bash
pip install numpy pandas torch matplotlib
```

---

## 📂 Data Format

Place your data files in a folder (e.g., `sample/`). Each `.csv` file should look like this:

- **Row 0**: Spectrometer wavelengths (x-axis)
- **Rows 1-722**: Absorbance values from 722 laser shots

File name examples:

```
Glucose_CRS-2025-02-14-13-42-22.csv
Bilirubin-2025-02-08-22-45-33.csv
```

---

## 🚀 Running the Script

Run the script with:

```bash
python main.py
```

This will:
- Train the model
- Save the best model
- Plot and save `submission_biomarkers.png`

---

## 🧠 Core Components

### 1. `set_seed()`

Sets the seed across Python, NumPy, and PyTorch for reproducibility.

---

### 2. `parse_biomarker_name()`

Extracts biomarker names from file names, trimming off timestamp info.

---

### 3. `load_biomarker_data_2d(data_dir)`

- Reads all `.csv` files in `data_dir`
- Extracts "diagonal" absorbance values (laser wavelength = spectrometer wavelength)
- Applies min-max normalization per biomarker
- Returns:
  - `data_tensor`: shape `(N, 721)`
  - `laser_wavelengths`: `np.arange(380, 1101)`
  - `filenames`: list of parsed biomarker names

---

### 4. `SpectralAttention2D(nn.Module)`

A PyTorch model that learns an attention vector over wavelengths **for each biomarker**.

- Learns shape: `(N_biomarkers, M_wavelengths)`
- Output is passed through sigmoid
- Attention values are learned via gradient descent

---

### 5. `spectral_2d_loss(...)`

Custom loss function:

- Encourages **high attention where the absorbance is high** (same sum)
- Penalizes **attention in unrelated places** (beta term)
- Adds a **penalty if total attention is too high**

Returns either total loss or per-biomarker loss components.

---

### 6. `train_spectral_attention_2d(...)`

- Trains the attention model with Adam optimizer
- Tracks best-performing model (lowest loss)
- Loads and returns best model
- Prints per-biomarker stats at the end

---

### 7. `plot_all_biomarkers_on_same_axis(...)`

- Plots **all biomarkers** on the same chart
- Shows absorbance curve (normalized)
- Highlights **top 5 attention peaks** per biomarker
- Annotates the wavelength (nm) of each peak
- Saves the figure to a PNG

---

### 8. `pick_top_k_with_min_gap(...)`

Helper to select top K attention peaks, ensuring they’re **at least `min_gap` indices apart** (to avoid clustering).

---

## 📈 Example Output

- All biomarkers plotted with colored absorbance curves
- Top 5 wavelengths with highest attention marked
- Output saved as `submission_biomarkers.png`

---

## 💡 Use Cases

- Spectral feature selection
- Spectroscopic biomarker analysis
- Interpretable attention modeling
- Identifying key wavelength bands

---

## 🖼 Sample Visualization

Here’s an example (auto-generated from code):

```
[0] Glucose_CRS:
    same_sum_i:          14.3710
    partial_total_sum_i: 28.7025
    beta_term_i:         14.3315
    partial_loss_i:      -7.2052
...
```

---

## 🧼 Cleaning Up

To restart training or visualize a different dataset:
- Replace the CSVs in the `sample/` folder
- Rerun `python main.py`

---

## 📜 License

MIT License. Feel free to use and modify!