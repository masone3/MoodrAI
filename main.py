from codebase import data_loader, baseline_classifier

AUDIO_DIR = "datasets/DEAM/audio"
ANNOTATIONS_CSV = "datasets/DEAM/static_annotations.csv"

if __name__ == "__main__":
    # Step 1: Load features
    X, y = data_loader.load_deam_dataset_mfcc(AUDIO_DIR, ANNOTATIONS_CSV, n_samples=100)

    # Step 2: Train baseline
    clf, scaler = baseline_classifier.train_baseline(X, y)
