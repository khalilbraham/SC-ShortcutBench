# HuggingFace Dataset Upload Instructions

Complete guide to upload SC-ShortcutBench datasets to HuggingFace.

## Prerequisites

1. **HuggingFace Account**
   - Create account at https://huggingface.co if you don't have one
   - Verify email

2. **Authentication**
   ```bash
   # Login with your HuggingFace token
   huggingface-cli login
   # Or export token:
   export HF_TOKEN="hf_your_token_here"
   ```
   Get your token from: https://huggingface.co/settings/tokens

3. **Git Configuration** (if not already set up)
   ```bash
   git config --global user.email "your.email@example.com"
   git config --global user.name "Your Name"
   ```

## Step 1: Create HuggingFace Repositories

### Public Dataset

1. Go to https://huggingface.co/new
2. Fill in:
   - **Repository name**: `sc-shortcutbench-public`
   - **Type**: Dataset
   - **License**: CC-BY-4.0
   - **Private**: No (PUBLIC)
3. Click "Create repository"

### Private Dataset

1. Go to https://huggingface.co/new
2. Fill in:
   - **Repository name**: `sc-shortcutbench-private`
   - **Type**: Dataset
   - **License**: CC-BY-4.0
   - **Private**: Yes (PRIVATE)
3. Click "Create repository"
4. Once created, go to **Settings → Approval**
5. Set to "Manual approval for new users"

## Step 2: Upload Public Dataset

```bash
cd /datadisks/datadisk1/khalil/sc_shortcut_project

# Make scripts executable
chmod +x upload_public_dataset.sh

# Run upload
./upload_public_dataset.sh
```

This will:
- Clone the HuggingFace repository
- Copy all expression data, metadata, and predictions
- Create dataset card
- Push to HuggingFace (takes ~10-20 minutes)

**Expected size**: ~1.3 GB

## Step 3: Upload Private Dataset

```bash
# Run upload
./upload_private_dataset.sh
```

This will:
- Clone the HuggingFace repository
- Copy test expression data only
- Create dataset card with privacy notice
- Push to HuggingFace (takes ~5-10 minutes)

**Expected size**: ~607 MB

## Step 4: Configure Private Dataset Access

1. Go to https://huggingface.co/datasets/khalil/sc-shortcutbench-private
2. Navigate to **Settings → Approval**
3. Ensure "Manual approval for new users" is selected
4. (Optional) Add trusted collaborators under **Settings → Manage access**

## Step 5: Verify Upload

### Check Public Dataset
```bash
python3 << 'EOF'
from datasets import load_dataset

# Load public dataset
ds = load_dataset("Khalilbraham/sc-shortcutbench-public")
print(f"✓ Public dataset loaded")
print(f"  Splits: {list(ds.keys())}")
EOF
```

### Check Private Dataset
```bash
python3 << 'EOF'
from datasets import load_dataset

# Load private dataset (requires token)
ds = load_dataset("khalil/sc-shortcutbench-private", token=True)
print(f"✓ Private dataset loaded")
print(f"  Splits: {list(ds.keys())}")
EOF
```

## Data Organization

### Public Dataset (`Khalilbraham/sc-shortcutbench-public`)
```
├── README.md
├── dataset_card.md
├── train_expression/
│   ├── balanced_expression_full.h5ad (340 MB)
│   └── decorrelated_expression_full.h5ad (268 MB)
├── metadata/
│   ├── balanced_challenge.csv
│   ├── decorrelated_challenge.csv
│   ├── conflict_rows_spec.json
│   └── dataset_summary.json
└── predictions/
    ├── downstream_reliance_table.csv
    ├── embedding_probe_report.json
    └── ... (40+ files with all results)
```

### Private Dataset (`khalil/sc-shortcutbench-private`)
```
├── PRIVATE_USAGE.md
├── dataset_card.md
└── test_expression/
    ├── balanced_expression_test.h5ad (340 MB)
    └── decorrelated_expression_test.h5ad (268 MB)
```

## Usage Examples

### For Dataset Users

```python
# Load public dataset
from datasets import load_dataset
import pandas as pd
import scanpy as sc

# Get metadata
public = load_dataset("Khalilbraham/sc-shortcutbench-public")
conflicts = pd.read_csv("balanced_challenge.csv")

# Load expression (if h5ad files are available)
adata_train = sc.read_h5ad("balanced_expression_full.h5ad")

# For evaluation (private)
private = load_dataset("khalil/sc-shortcutbench-private", token=True)
adata_test = sc.read_h5ad("balanced_expression_test.h5ad")

# Train and evaluate
model = train_model(adata_train)
results = evaluate_model(model, adata_test, conflict_indices=conflicts)
```

## Troubleshooting

### "Permission denied" when pushing
- Check that you're logged in: `huggingface-cli whoami`
- Verify the repository exists and you have write access
- Regenerate token if needed

### "File too large"
- HuggingFace has a 5 GB per file limit
- Our h5ad files (~340 MB) are well under this
- Splitting large files: already done in dataset structure

### Private dataset not restricted
- Go to repo **Settings → Approval**
- Change from "Public" to "Approval" mode
- Save changes

### Users can't request access
- Ensure private repo is set to PRIVATE (not PUBLIC)
- Check **Settings → Approval** → "Manual approval enabled"
- Share the direct URL: https://huggingface.co/datasets/khalil/sc-shortcutbench-private

## Next Steps

1. ✅ Datasets uploaded
2. ✅ Public dataset accessible
3. ✅ Private dataset with access control
4. 📝 Announce datasets (blog post, paper)
5. 📊 Monitor usage and access requests
6. 📋 Keep documentation up-to-date

## Contact & Support

For issues:
- HuggingFace docs: https://huggingface.co/docs/hub/datasets
- Paper info: [Your paper link]
- Dataset questions: [Your contact info]

---

**Status**: Ready for upload  
**Last updated**: 2026-05-05
