# Google Colab Setup Guide for Crime Prediction Analysis

This guide provides step-by-step instructions for running the crime prediction analysis notebook in Google Colab.

---

## Step 1: Access Google Colab

1. Open your web browser
2. Go to [https://colab.research.google.com](https://colab.research.google.com)
3. Sign in with your Google account (if not already signed in)

---

## Step 2: Open the Notebook from GitHub

### Method A: Direct GitHub Import (Recommended)
1. In Colab, click **File** → **Open notebook**
2. Click the **GitHub** tab
3. Paste your repository URL: `https://github.com/Narasimhaponnada/predictive-policing`
4. Click the search icon or press Enter
5. Click on `crime_prediction_analysis.ipynb` to open it

### Method B: Clone Repository (Alternative)
1. Create a new notebook in Colab
2. **IMPORTANT:** Make sure your GitHub repository is PUBLIC (see troubleshooting below if you get authentication errors)
3. Run this command in the first cell:
```python
!git clone https://github.com/Narasimhaponnada/predictive-policing.git
%cd predictive-policing
```

**⚠️ If you get "fatal: could not read Username" error:**
Your repository is private. To fix this:
1. Go to [https://github.com/Narasimhaponnada/predictive-policing/settings](https://github.com/Narasimhaponnada/predictive-policing/settings)
2. Scroll down to **Danger Zone**
3. Click **Change visibility** → **Change to public**
4. Confirm by typing the repository name
5. Try cloning again in Colab

---

## Step 3: Download the Dataset

The CSV file is too large for GitHub (243 MB), so you need to download it separately.

### Option A: Direct Download from Data.gov (Recommended)

**Add this cell at the beginning of your notebook:**

```python
# Download the crime dataset
import urllib.request
import os

# Check if file already exists
if not os.path.exists('Crime_Data_from_2020_to_Present.csv'):
    print("Downloading crime dataset (this may take 2-3 minutes)...")
    url = 'https://data.lacity.org/api/views/2nrs-mtv8/rows.csv?accessType=DOWNLOAD'
    urllib.request.urlretrieve(url, 'Crime_Data_from_2020_to_Present.csv')
    print("✓ Download complete!")
else:
    print("✓ Dataset already exists")

# Verify file size
file_size = os.path.getsize('Crime_Data_from_2020_to_Present.csv') / (1024**2)
print(f"Dataset size: {file_size:.2f} MB")
```

### Option B: Upload from Your Computer

If you have the file locally:

1. Click the **folder icon** 📁 in the left sidebar
2. Click the **upload icon** ⬆️
3. Select `Crime_Data_from_2020_to_Present.csv` from your computer
4. Wait for upload to complete (may take several minutes due to file size)

**⚠️ Note:** Files uploaded to Colab are temporary and deleted when the session ends.

---

## Step 4: Install Required Packages

Google Colab comes with most packages pre-installed, but you may need to install or upgrade some:

```python
# Install/upgrade required packages
!pip install --upgrade xgboost shap plotly
```

**Packages that are usually pre-installed in Colab:**
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

---

## Step 5: Set Runtime Type (Important!)

For faster execution with large dataset:

1. Click **Runtime** in the top menu
2. Select **Change runtime type**
3. Choose:
   - **Runtime type:** Python 3
   - **Hardware accelerator:** GPU or T4 GPU (free tier)
   - **High-RAM** if available
4. Click **Save**

**Why GPU?** XGBoost can leverage GPU acceleration for faster training on large datasets.

---

## Step 6: Run the Notebook

### Option A: Run All Cells
1. Click **Runtime** → **Run all** (Ctrl/Cmd + F9)
2. Wait for all cells to execute sequentially
3. Total execution time: ~10-20 minutes depending on GPU availability

### Option B: Run Cells Step-by-Step
1. Click on a cell
2. Press **Shift + Enter** to run and move to next cell
3. Or click the **▶ Play button** on the left of each cell

---

## Step 7: Save Your Work

### Save to Google Drive (Recommended)
1. Click **File** → **Save a copy in Drive**
2. Your notebook will be saved to `Colab Notebooks` folder in Google Drive
3. Any modifications are automatically saved

### Download Results
1. Click the **folder icon** 📁 in left sidebar
2. Right-click on any file (e.g., trained models, plots)
3. Select **Download**

---

## Common Issues & Solutions

### ⚠️ Issue 0: "Could not read Username" When Cloning
**Problem:** Repository is private and requires authentication

**Solution:** Make your repository public
1. Go to your repository: [https://github.com/Narasimhaponnada/predictive-policing](https://github.com/Narasimhaponnada/predictive-policing)
2. Click **Settings** (top right)
3. Scroll to bottom **Danger Zone** section
4. Click **Change visibility** → **Change to public**
5. Type repository name to confirm: `predictive-policing`
6. Click **I understand, change repository visibility**
7. Return to Colab and try cloning again

**Alternative (Keep Private):** Use personal access token
```python
# Replace YOUR_TOKEN with your GitHub personal access token
!git clone https://YOUR_TOKEN@github.com/Narasimhaponnada/predictive-policing.git
%cd predictive-policing
```
To create a token: GitHub → Settings → Developer settings → Personal access tokens → Generate new token

### ⚠️ Issue 1: "Runtime Disconnected"
**Solution:** Colab has timeout limits (90 minutes idle, 12 hours max)
- Click **Reconnect** button
- Re-run cells starting from the dataset download cell

### ⚠️ Issue 2: "Out of Memory" Error
**Solution:** 
- Switch to High-RAM runtime (Runtime → Change runtime type)
- Reduce dataset size by sampling:
```python
# Add after loading data
df = df.sample(frac=0.5, random_state=42)  # Use 50% of data
```

### ⚠️ Issue 3: "Dataset Not Found" Error
**Solution:**
- Re-run the dataset download cell (Step 3)
- Check the file exists in the file browser (left sidebar)

### ⚠️ Issue 4: Package Import Errors
**Solution:**
- Run the package installation cell (Step 4)
- Restart runtime: **Runtime → Restart runtime**
- Re-run cells from the beginning

### ⚠️ Issue 5: Slow Execution
**Solution:**
- Ensure GPU is enabled (Step 5)
- Close other Colab notebooks
- Try running during off-peak hours

---

## Colab-Specific Tips

### 📌 Tip 1: Check GPU Status
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

### 📌 Tip 2: Monitor Memory Usage
```python
!nvidia-smi  # Shows GPU memory usage
```

### 📌 Tip 3: Keep Session Alive
- Colab disconnects after ~90 minutes of inactivity
- Keep the browser tab active or use browser extensions

### 📌 Tip 4: Download Trained Models
```python
# After training completes
from google.colab import files
files.download('xgboost_crime_model.pkl')
```

### 📌 Tip 5: Mount Google Drive for Persistent Storage
```python
from google.colab import drive
drive.mount('/content/drive')

# Save model to Drive
import joblib
joblib.dump(model, '/content/drive/MyDrive/crime_model.pkl')
```

---

## Expected Execution Time

| Cell/Section | Approximate Time |
|-------------|------------------|
| Dataset Download | 2-3 minutes |
| Package Installation | 1-2 minutes |
| Data Loading & Cleaning | 2-4 minutes |
| Feature Engineering | 1-2 minutes |
| Model Training (XGBoost) | 5-10 minutes |
| Model Evaluation | 2-3 minutes |
| Visualizations | 2-3 minutes |
| **Total** | **15-25 minutes** |

*Times vary based on Colab's available resources*

---

## Keyboard Shortcuts in Colab

| Action | Shortcut |
|--------|----------|
| Run cell | Shift + Enter |
| Run all cells | Ctrl/Cmd + F9 |
| Add cell above | Ctrl/Cmd + M, A |
| Add cell below | Ctrl/Cmd + M, B |
| Delete cell | Ctrl/Cmd + M, D |
| Comment/Uncomment | Ctrl/Cmd + / |
| Save | Ctrl/Cmd + S |

---

## Quick Start Checklist

- [ ] Opened Google Colab
- [ ] Loaded notebook from GitHub
- [ ] Downloaded crime dataset
- [ ] Installed required packages
- [ ] Set runtime to GPU
- [ ] Executed all cells
- [ ] Saved copy to Google Drive

---

## Need Help?

- **Colab Documentation:** [https://colab.research.google.com/notebooks/intro.ipynb](https://colab.research.google.com/notebooks/intro.ipynb)
- **Dataset Source:** [https://catalog.data.gov/dataset/crime-data-from-2020-to-present](https://catalog.data.gov/dataset/crime-data-from-2020-to-present)
- **LA Open Data Portal:** [https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8](https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8)

---

**Happy Analyzing! 🚀**
