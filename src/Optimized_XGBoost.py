

#%% CWRU Optimized Preprocessing Pipeline (ML + DL)

import os
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt


# ---------------- CONFIGURATION ----------------

DATA_DIR = r"C:\Users\GouthamT\OneDrive - TTCO\Desktop\TensorFlow Files\CWRU_project\data\Extracted_mat_files"
os.chdir(DATA_DIR)

# Sampling frequency for CWRU (Hz)
SAMPLE_RATE = 12000
WINDOW_SIZE = 1024

# Map each file to its class label (your confirmed mapping)
FILE_LABELS = {
    "B007_1_123.mat":       "BallFault_007",
    "B014_1_190.mat":       "BallFault_014",
    "B021_1_227.mat":       "BallFault_021",

    "IR007_1_110.mat":      "InnerRace_007",
    "IR014_1_175.mat":      "InnerRace_014",
    "IR021_1_214.mat":      "InnerRace_021",

    "OR007_6_1_136.mat":    "OuterRace_007",
    "OR014_6_1_202.mat":    "OuterRace_014",
    "OR021_6_1_239.mat":    "OuterRace_021",

    "Time_Normal_1_098.mat":"Normal",
}

# --------------- HELPER FUNCTIONS ---------------

def load_de_fe_signals(mat_path):
    """
    Load first available DE and FE time signals from a .mat file.
    Returns: de_1d, fe_1d (flattened 1D numpy arrays).
    """
    data = sio.loadmat(mat_path)
    keys = list(data.keys())

    de_keys = [k for k in keys if k.endswith('_DE_time')]
    fe_keys = [k for k in keys if k.endswith('_FE_time')]

    if not de_keys or not fe_keys:
        raise ValueError(f"No DE/FE time keys found in {mat_path}")

    # Use the first DE and first FE signal (works for your IR014 etc.)
    de = data[de_keys[0]].flatten()
    fe = data[fe_keys[0]].flatten()

    return de, fe


def max_abs_normalize(signal):
    """Normalize signal to [-1, 1] using max-abs scaling."""
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val


def create_windows(signal, window_size):
    """
    Slice 1D signal into non-overlapping windows of size window_size.
    Returns array of shape (num_windows, window_size).
    """
    num_windows = len(signal) // window_size
    if num_windows == 0:
        return np.empty((0, window_size))
    return signal[:num_windows * window_size].reshape(num_windows, window_size)


def extract_time_freq_features(windows, sample_rate=SAMPLE_RATE):
    """
    Extract 18 time- and frequency-domain features per window.
    Input: windows -> shape (num_windows, window_size)
    Output: features -> shape (num_windows, 18)
    """
    all_features = []

    for window in windows:
        window = np.asarray(window)
        if window.size == 0:
            all_features.append([0.0] * 18)
            continue

        # ----- Time-domain features -----
        mean_val = np.mean(window)
        rms_val  = np.sqrt(np.mean(window**2))
        std_val  = np.std(window)
        var_val  = np.var(window)
        kurtosis_val  = kurtosis(window, fisher=True, bias=False)
        skewness_val  = skew(window, bias=False)

        max_val = np.max(window)
        min_val = np.min(window)
        ptp_val = max_val - min_val

        peak_abs      = np.max(np.abs(window))
        mean_abs      = np.mean(np.abs(window))
        mean_sqrt_abs = np.mean(np.sqrt(np.abs(window)))

        crest_factor    = peak_abs / rms_val        if rms_val       != 0 else 0.0
        shape_factor    = rms_val / mean_abs        if mean_abs      != 0 else 0.0
        impulse_factor  = peak_abs / mean_abs       if mean_abs      != 0 else 0.0
        clearence_factor= peak_abs / (mean_sqrt_abs**2) if mean_sqrt_abs != 0 else 0.0

        # ----- Frequency-domain features -----
        n = window.size
        freq = np.fft.rfftfreq(n, d=1.0/sample_rate)
        fft_values = np.fft.rfft(window)
        mag = np.abs(fft_values)
        power = mag**2

        power_sum = np.sum(power)
        mag_sum   = np.sum(mag)

        if mag.size > 0 and mag_sum != 0:
            dominant_idx       = np.argmax(mag)
            dominant_freq      = float(freq[dominant_idx])
            dominant_amplitude = float(mag[dominant_idx])
        else:
            dominant_freq      = 0.0
            dominant_amplitude = 0.0

        if power_sum != 0:
            spectral_centroid = float(np.sum(freq * power) / power_sum)
            rms_freq          = float(np.sqrt(np.sum((freq**2) * power) / power_sum))
        else:
            spectral_centroid = 0.0
            rms_freq          = 0.0

        spectral_energy = float(power_sum)

        features = [
            mean_val, std_val, rms_val, kurtosis_val, skewness_val, var_val,
            max_val, min_val, ptp_val,
            crest_factor, shape_factor, impulse_factor, clearence_factor,
            dominant_freq, dominant_amplitude, spectral_centroid,
            spectral_energy, rms_freq
        ]
        all_features.append(features)

    return np.array(all_features)


# --------------- MAIN PREPROCESSING LOOP ---------------

all_windows_raw = []   # for Deep Learning(raw windows)
all_labels_raw  = []

all_features    = []   # for Machine Learning (handcrafted features)
all_labels_feat = []

for fname, label in FILE_LABELS.items():
    mat_path = os.path.join(DATA_DIR, fname)
    print(f"Processing file: {fname}  -> label: {label}")

    # 1) Load DE and FE signals
    de_signal, fe_signal = load_de_fe_signals(mat_path)

    # 2) Normalize
    de_norm = max_abs_normalize(de_signal)
    fe_norm = max_abs_normalize(fe_signal)

    # 3) Windowing
    de_windows = create_windows(de_norm, WINDOW_SIZE)
    fe_windows = create_windows(fe_norm, WINDOW_SIZE)

    # 4) Stack DE + FE windows for this file
    file_windows = np.vstack([de_windows, fe_windows])  # shape: (n_de + n_fe, 1024)

    # 5) Labels for these windows
    file_labels = np.array([label] * file_windows.shape[0])

    # 6) Store for raw pipeline (DL)
    all_windows_raw.append(file_windows)
    all_labels_raw.append(file_labels)

    # 7) Extract features for ML pipeline
    file_features = extract_time_freq_features(file_windows, sample_rate=SAMPLE_RATE)
    all_features.append(file_features)
    all_labels_feat.append(file_labels)

# --------------- FINAL DATASETS ---------------

# Raw windows (for deep learning)
X_windows_raw = np.vstack(all_windows_raw)      # shape: (N_samples, 1024)
y_raw         = np.concatenate(all_labels_raw)  # shape: (N_samples,)

# Feature matrix (for classical ML)
X_features    = np.vstack(all_features)         # shape: (N_samples, 18)
y_features    = np.concatenate(all_labels_feat) # shape: (N_samples,)

print("X_windows_raw shape:", X_windows_raw.shape)
print("y_raw shape:        ", y_raw.shape)
print("X_features shape:   ", X_features.shape)
print("y_features shape:   ", y_features.shape)
print("Unique labels:", np.unique(y_raw))

# Optional: feature names for reference
feature_names = [
    'Mean', 'Standard Deviation', 'RMS', 'Kurtosis', 'Skewness', 'Variance',
    'Max', 'Min', 'Peak to Peak',
    'Crest Factor', 'Shape Factor', 'Impulse Factor', 'Clearance Factor',
    'Dominant Frequency', 'Dominant Amplitude', 'Spectral Centroid',
    'Spectral Energy', 'RMS Frequency'
]

df_features = pd.DataFrame(X_features, columns=feature_names)
print(df_features.head())

#Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y_features)

print("label Classes:", le.classes_)
print("Encoded labels:", y_encoded[:10])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print("Training set shape:", X_train.shape, X_test.shape)

# Train XGBoost Classifier
model = XGBClassifier(
    n_estimators=350,
    learning_rate=0.5,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    use_label_encoder=False, 
    eval_metric='mlogloss')
print("Training XGBoost Classifier...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print("Predictions completed.")

#metrics
accuracy= accuracy_score(y_test, y_pred)
print ("\nMODEL ACCURACY:", accuracy)

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
