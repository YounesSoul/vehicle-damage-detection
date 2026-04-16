
# %% [markdown]
# ## 1. Setup and Imports

# %%
import os
import shutil
import uuid
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Automatically handle execution environment (Notebook/Script)
# Set paths relative to the project root
try:
    # If running as script
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # If running in a notebook located in the 'notebooks' folder
    PROJECT_ROOT = Path.cwd().parent

# Define Paths
RAW_DATA_DIR = PROJECT_ROOT / "car-damage-detection-dataset"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Class Mapping definitions
# Maps raw folders '00-damage' -> 'damage', '01-whole' -> 'whole'
CLASS_MAP = {
    "00-damage": "damage",
    "01-whole": "whole"
}

# Ensure random state for reproducibility
RANDOM_STATE = 42

print(f"Project root identified as: {PROJECT_ROOT}")

# %% [markdown]
# ## 2. Consolidate Images into a DataFrame
# We crawl the raw dataset directory recursively to find all `.jpg` or `.jpeg` files.

# %%
def collect_image_paths(base_dir: Path) -> pd.DataFrame:
    """
    Recursively scans the provided directory for image files within designated class folders.
    Returns a Pandas DataFrame containing 'filepath' and 'label'.
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {base_dir}")
        
    data = []
    
    # Recursively look for images
    for img_path in base_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            parent_dir_name = img_path.parent.name
            
            # Map the original folder to our cleaner label
            if parent_dir_name in CLASS_MAP:
                data.append({
                    "filepath": str(img_path),
                    "label": CLASS_MAP[parent_dir_name]
                })

    df = pd.DataFrame(data)
    return df

# Collect images
full_df = collect_image_paths(RAW_DATA_DIR)
print(f"✅ Found {len(full_df)} total images.")
print(full_df.head())

# %% [markdown]
# ## 3. Stratified Split (70% Train, 15% Validation, 15% Test)
# We use scikit-learn's `train_test_split` with the `stratify` parameter to ensure minority classes are represented equally across all splits.

# %%
# Step 1: Split into Train (70%) and Temporary (30%)
train_df, temp_df = train_test_split(
    full_df, 
    test_size=0.30, 
    stratify=full_df["label"], 
    random_state=RANDOM_STATE
)

# Step 2: Split the Temporary (30%) into Validation (15%) and Test (15%)
val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.50, # 50% of the 30% is 15% of the total
    stratify=temp_df["label"], 
    random_state=RANDOM_STATE
)

print(f"Split completed successfully!")

# %% [markdown]
# ## 4. Verification and Class Distribution Assessment
# Check the raw item counts and the proportion of classes within each split.

# %%
def print_split_distributions():
    splits = [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df)
    ]
    
    print("-" * 50)
    for name, df in splits:
        total = len(df)
        pct_of_total = (total / len(full_df)) * 100
        print(f"\n{name} Split: {total:,} images ({pct_of_total:.1f}% of total)")
        
        # Calculate distribution
        distribution = df["label"].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
        print(distribution)

print_split_distributions()

# %% [markdown]
# ## 5. Create Target Directory Structure & Safe Copy Strategy
# 
# Now we generate `data/processed/{train,val,test}/{damage,whole}`. 
# To guarantee no filename overwriting due to identical names coming from different source folders, we append a short UUID hash to every copied image.

# %%
def prepare_directory_structure(target_dir: Path):
    """Creates the necessary folder hierarchy."""
    splits = ["train", "val", "test"]
    categories = list(CLASS_MAP.values()) # ['damage', 'whole']
    
    if target_dir.exists():
        print(f"🧹 Clearing existing directory at {target_dir} for a clean run...")
        shutil.rmtree(target_dir)
        
    for split in splits:
        for cat in categories:
            (target_dir / split / cat).mkdir(parents=True, exist_ok=True)
            
    print(f"✅ Directory structure prepared at {target_dir}")

def copy_split_to_disk(df: pd.DataFrame, split_name: str, target_dir: Path):
    """Copies images into their respective folders with collision-safe names."""
    print(f"Copying '{split_name}' images...")
    
    for _, row in df.iterrows():
        src_path = Path(row["filepath"])
        label = row["label"]
        
        # 8-character hash prevents file name collision
        unique_id = uuid.uuid4().hex[:8] 
        new_filename = f"{src_path.stem}_{unique_id}{src_path.suffix}"
        
        dest_path = target_dir / split_name / label / new_filename
        shutil.copy2(src_path, dest_path)
    print(f"✅ {split_name.capitalize()} copied successfully!")

# Execute file operations
prepare_directory_structure(PROCESSED_DATA_DIR)

copy_split_to_disk(train_df, "train", PROCESSED_DATA_DIR)
copy_split_to_disk(val_df, "val", PROCESSED_DATA_DIR)
copy_split_to_disk(test_df, "test", PROCESSED_DATA_DIR)

print("\n🚀 All data has been perfectly prepared for model training!")

# %% [markdown]
# ## 6. Final Sanity Check
# Verify that the physical files dynamically copied to the disk match the expected split sizes.

# %%
print("\nFinal File Count Verification:")
print("-" * 50)
for split in ["train", "val", "test"]:
    for label in ["damage", "whole"]:
        folder = PROCESSED_DATA_DIR / split / label
        if folder.exists():
            count = len(list(folder.glob("*")))
            print(f"{split}/{label}: {count} files")
        else:
            print(f"{split}/{label}: 0 files (Directory missing)")

# %% [markdown]
# ### Next Steps:
# With our highly controlled portfolio-ready dataset structure generated inside `data/processed`, we can cleanly move onwards to:
# - Data Augmentation Definitions (e.g. `ImageDataGenerator` / `Albumentations`)
# - PyTorch/Tensorflow Dataset Loader setup
# - Baseline Model experimentation
