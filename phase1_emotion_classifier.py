import os
import re
import pickle
import warnings
from collections import Counter

import librosa
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

METADATA_FILE = "master_metadata.xlsx"
DATASET_ROOT = "dataset"
AUDIO_SAMPLE_RATE = 22050
RANDOM_STATE = 42
TEST_SIZE = 0.20


EMOTION_MAP = {
    "furious": "Angry",
    "angry": "Angry",
    "anger": "Angry",
    "mad": "Angry",
    "neutral": "Neutral",
    "normal": "Neutral",
    "shocked": "Surprised",
    "shock": "Surprised",
    "surprised": "Surprised",
    "surprise": "Surprised",
    "happy": "Happy",
    "happiness": "Happy",
    "joy": "Happy",
    "sad": "Sad",
    "sadness": "Sad",
}


def find_column(df, candidates):
    lower_map = {str(col).lower().strip(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.lower().strip()
        if key in lower_map:
            return lower_map[key]
    return None


def normalize_emotion(label):
    if pd.isna(label):
        return None
    text = str(label).strip()
    key = text.lower().strip().replace(" ", "_")
    return EMOTION_MAP.get(key, text.capitalize())


def emotion_from_filename(file_name):
    name = os.path.basename(str(file_name)).lower()
    for key, value in EMOTION_MAP.items():
        if re.search(rf"(^|[_\- .]){re.escape(key)}([_\- .]|$)", name):
            return value
    for key, value in EMOTION_MAP.items():
        if key in name:
            return value
    return None


def find_actual_path(path_or_file):
    if pd.isna(path_or_file):
        return None

    raw = str(path_or_file).replace("\\", "/")
    file_name = os.path.basename(raw)

    candidates = [
        raw,
        file_name,
        os.path.join(DATASET_ROOT, raw),
        os.path.join(DATASET_ROOT, file_name),
    ]

    parts = raw.split("/")
    if len(parts) >= 2:
        group_folder = parts[-2]
        candidates.extend([
            os.path.join(group_folder, file_name),
            os.path.join(DATASET_ROOT, group_folder, file_name),
            os.path.join(DATASET_ROOT, group_folder.replace("GRUP_", "GROUP_"), file_name),
            os.path.join(DATASET_ROOT, group_folder.replace("GROUP_", "GRUP_"), file_name),
        ])

    for c in candidates:
        if c and os.path.exists(c):
            return c

    # Final fallback: search by file name under current folder.
    for root, _, files in os.walk("."):
        if file_name in files:
            return os.path.join(root, file_name)

    return None


def safe_stats(values, prefix):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_median": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
    }


def extract_features(file_path, sr=AUDIO_SAMPLE_RATE):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    if len(y) == 0:
        raise ValueError("Empty audio file")

    # Trim silence + normalize. This often helps emotion recognition.
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) == 0:
        raise ValueError("Audio became empty after trimming")
    y = librosa.util.normalize(y)

    features = {}

    # Core time features
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    features.update(safe_stats(rms, "rms"))
    features.update(safe_stats(zcr, "zcr"))

    # MFCC + delta + delta-delta. Stronger than only MFCC mean.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    for i in range(20):
        features.update(safe_stats(mfcc[i], f"mfcc_{i+1}"))
        features.update(safe_stats(delta[i], f"delta_mfcc_{i+1}"))
        features.update(safe_stats(delta2[i], f"delta2_mfcc_{i+1}"))

    # Spectral / timbre features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    features.update(safe_stats(centroid, "spectral_centroid"))
    features.update(safe_stats(bandwidth, "spectral_bandwidth"))
    features.update(safe_stats(rolloff, "spectral_rolloff"))
    features.update(safe_stats(flatness, "spectral_flatness"))

    for i in range(contrast.shape[0]):
        features.update(safe_stats(contrast[i], f"spectral_contrast_{i+1}"))
    for i in range(chroma.shape[0]):
        features.update(safe_stats(chroma[i], f"chroma_{i+1}"))

    # Pitch features. librosa.yin is more robust than a very simple autocorrelation peak.
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0 = f0[np.isfinite(f0)]
        features.update(safe_stats(f0, "f0_yin"))
    except Exception:
        features.update(safe_stats([], "f0_yin"))

    # Tempo can sometimes separate energetic emotions.
    try:
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    except Exception:
        tempo = 0.0
    features["tempo"] = float(tempo)
    features["duration_sec"] = float(librosa.get_duration(y=y, sr=sr))

    return features


def load_metadata():
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"{METADATA_FILE} not found. Put this script next to the Excel file.")

    df = pd.read_excel(METADATA_FILE)

    present_col = find_column(df, ["audio_file_present", "file_present", "present"])
    if present_col is not None:
        df = df[df[present_col] == True].reset_index(drop=True)

    path_col = find_column(df, ["audio_relative_path", "relative_path", "path", "file_path", "audio_path"])
    file_col = find_column(df, ["file_name", "filename", "audio_file", "name"])
    label_col = find_column(df, ["feeling", "emotion", "label", "actual_emotion"])

    if path_col is None and file_col is None:
        raise ValueError("No audio path/file column found in Excel.")

    df["_audio_input"] = df[path_col] if path_col is not None else df[file_col]
    df["real_path"] = df["_audio_input"].apply(find_actual_path)

    # Prefer Excel label. If missing, derive from file name.
    if label_col is not None:
        df["emotion"] = df[label_col].apply(normalize_emotion)
    else:
        df["emotion"] = None

    df["_file_for_label"] = df[file_col] if file_col is not None else df["_audio_input"]
    df["emotion_from_file"] = df["_file_for_label"].apply(emotion_from_filename)
    df["emotion"] = df["emotion"].fillna(df["emotion_from_file"])

    # If Excel labels are weird or empty, use filename labels where available.
    mask_filename_has_label = df["emotion_from_file"].notna()
    df.loc[mask_filename_has_label, "emotion"] = df.loc[mask_filename_has_label, "emotion_from_file"]

    return df


def build_models():
    return {
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(C=10, gamma="scale", kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=700,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=700,
            max_depth=None,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def main():
    print("[BOOSTED PHASE 1] Loading metadata...")
    df = load_metadata()

    found = df["real_path"].notna().sum()
    print(f"[BOOSTED PHASE 1] Files found: {found}/{len(df)}")

    df = df[df["real_path"].notna() & df["emotion"].notna()].reset_index(drop=True)
    if len(df) < 10:
        raise RuntimeError("Too few usable files. Check dataset location and Excel labels.")

    rows = []
    errors = []
    print("[BOOSTED PHASE 1] Extracting stronger audio features...")

    for i, row in df.iterrows():
        try:
            feat = extract_features(row["real_path"])
            feat["file_name"] = os.path.basename(row["real_path"])
            feat["real_path"] = row["real_path"]
            feat["emotion"] = row["emotion"]
            rows.append(feat)
        except Exception as exc:
            errors.append({"file": row["real_path"], "error": str(exc)})

        if (i + 1) % 25 == 0 or i == 0:
            print(f"[BOOSTED PHASE 1] Processed {i + 1}/{len(df)}")

    feature_df = pd.DataFrame(rows)
    feature_df.to_csv("phase1_boosted_feature_table.csv", index=False)

    if errors:
        pd.DataFrame(errors).to_csv("phase1_boosted_errors.csv", index=False)
        print(f"[WARNING] {len(errors)} files failed. See phase1_boosted_errors.csv")

    print("\n[BOOSTED PHASE 1] Label distribution:")
    print(feature_df["emotion"].value_counts())

    feature_cols = [c for c in feature_df.columns if c not in ["file_name", "real_path", "emotion"]]
    X = feature_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_text = feature_df["emotion"].astype(str).values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    class_counts = Counter(y_text)
    stratify_arg = y if min(class_counts.values()) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_arg
    )

    models = build_models()
    results = []
    best_name = None
    best_model = None
    best_acc = -1
    best_pred = None

    print("\n[BOOSTED PHASE 1] Training several models and selecting the best one...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        results.append({"model": name, "accuracy": acc, "accuracy_percent": acc * 100})
        print(f"{name} Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model
            best_pred = pred

    labels = list(range(len(encoder.classes_)))
    class_names = encoder.classes_
    cm = confusion_matrix(y_test, best_pred, labels=labels)

    report_text = classification_report(
        y_test, best_pred, labels=labels, target_names=class_names, zero_division=0
    )
    report_dict = classification_report(
        y_test, best_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
    )

    print("\n================ BOOSTED PHASE 1 RESULTS ================")
    print(f"Best model: {best_name}")
    print(f"Accuracy: {best_acc:.4f}")
    print(f"Accuracy (%): {best_acc * 100:.2f}")
    print("\nClassification Report:")
    print(report_text)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=class_names, columns=class_names))

    pred_df = pd.DataFrame({
        "actual": encoder.inverse_transform(y_test),
        "predicted": encoder.inverse_transform(best_pred),
    })
    pred_df.to_csv("phase1_boosted_results.csv", index=False)

    model_comparison_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    report_df = pd.DataFrame(report_dict).transpose()
    summary_df = pd.DataFrame({
        "Metric": ["Best model", "Accuracy", "Accuracy (%)", "Total valid files", "Train samples", "Test samples", "Feature count"],
        "Value": [best_name, round(best_acc, 4), round(best_acc * 100, 2), len(feature_df), len(X_train), len(X_test), len(feature_cols)],
    })

    with pd.ExcelWriter("phase1_boosted_summary.xlsx") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        model_comparison_df.to_excel(writer, sheet_name="Model_Comparison", index=False)
        cm_df.to_excel(writer, sheet_name="Confusion_Matrix")
        report_df.to_excel(writer, sheet_name="Classification_Report")
        feature_df.to_excel(writer, sheet_name="Feature_Table", index=False)

    package = {
        "model": best_model,
        "model_name": best_name,
        "label_encoder": encoder,
        "feature_columns": feature_cols,
        "accuracy": best_acc,
        "class_names": list(class_names),
    }
    with open("phase1_boosted_model.pkl", "wb") as f:
        pickle.dump(package, f)

    print("\n[SAVED] phase1_boosted_feature_table.csv")
    print("[SAVED] phase1_boosted_results.csv")
    print("[SAVED] phase1_boosted_summary.xlsx")
    print("[SAVED] phase1_boosted_model.pkl")


if __name__ == "__main__":
    main()
