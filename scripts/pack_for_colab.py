"""
Packages the minimal required files to run Experiment 2 on Google Colab:
- src/ directory (merging scripts, evaluations, plotting)
- Trained checkpoints: outputs/checkpoints/e_math and outputs/checkpoints/e_pref
- Cached B0 reference log-probs: outputs/results/b0_pref_200_cache.json
- Progress file with Linear & SLERP cached: outputs/results/e2_sweep_progress.json
"""
import os
import zipfile

OUTPUT_ZIP = "cs769_e2_colab.zip"

FILES_TO_INCLUDE = [
    "outputs/results/b0_pref_200_cache.json",
    "outputs/results/e2_sweep_progress.json",
    "outputs/datasets/e_pref_test.jsonl"
]

DIRS_TO_INCLUDE = [
    "configs",
    "src",
    "outputs/checkpoints/e_math",
    "outputs/checkpoints/e_pref"
]

def create_colab_package():
    print(f"Creating Colab deployment package: {OUTPUT_ZIP}...")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add individual files
        for f in FILES_TO_INCLUDE:
            if os.path.exists(f):
                print(f"Adding file: {f}")
                zipf.write(f, arcname=f)
            else:
                print(f"Warning: File not found: {f}")

        # Add directories recursively
        for d in DIRS_TO_INCLUDE:
            if os.path.exists(d):
                print(f"Adding directory: {d}...")
                for root, _, files in os.walk(d):
                    for file in files:
                        filepath = os.path.join(root, file)
                        arcname = os.path.relpath(filepath, ".")
                        zipf.write(filepath, arcname=arcname)
            else:
                print(f"Warning: Directory not found: {d}")

    size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print(f"\nSuccessfully created {OUTPUT_ZIP} ({size_mb:.1f} MB)!")
    print("You can upload this zip directly to Google Drive or Colab.")

if __name__ == "__main__":
    create_colab_package()
