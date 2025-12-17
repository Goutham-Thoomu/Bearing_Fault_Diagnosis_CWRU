

#%% CWRU Optimized Preprocessing Pipeline (ML + DL)

import os
import scipy.io as sio
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling1D, Input
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from sklearn.utils import shuffle
from tensorflow.keras.layers import Input
# ---------------- CONFIGURATION ----------------

DATA_DIR = r"C:\Users\GouthamT\OneDrive - TTCO\Desktop\TensorFlow Files\CWRU_project\data\Extracted_mat_files"
os.chdir(DATA_DIR)

SAMPLE_RATE = 12000
WINDOW_SIZE = 1024

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
    """Load first available DE and FE time signals from a .mat file."""
    data = sio.loadmat(mat_path)
    keys = list(data.keys())

    de_keys = [k for k in keys if k.endswith('_DE_time')]
    fe_keys = [k for k in keys if k.endswith('_FE_time')]

    if not de_keys or not fe_keys:
        raise ValueError(f"No DE/FE time keys found in {mat_path}")

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
    """Slice 1D signal into non-overlapping windows."""
    num_windows = len(signal) // window_size
    if num_windows == 0:
        return np.empty((0, window_size))
    return signal[:num_windows * window_size].reshape(num_windows, window_size)


# --------------- MAIN PREPROCESSING LOOP ---------------

all_windows_raw = []   # for Deep Learning (raw windows)
all_labels_raw  = []

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
    file_windows = np.vstack([de_windows, fe_windows])  # (n_de + n_fe, 1024)

    # 5) Labels for these windows
    file_labels = np.array([label] * file_windows.shape[0])

    # 6) Store for raw pipeline (DL)
    all_windows_raw.append(file_windows)
    all_labels_raw.append(file_labels)

# --------------- FINAL DATASETS ---------------

X_windows_raw = np.vstack(all_windows_raw)       # (N_samples, 1024)
y_raw         = np.concatenate(all_labels_raw)   # (N_samples,)

print("X_windows_raw shape:", X_windows_raw.shape)
print("y_raw shape:        ", y_raw.shape)
print("Unique labels:", np.unique(y_raw))

# 1) Encode labels as integers
le = LabelEncoder()
y_int = le.fit_transform(y_raw)
num_classes = len(le.classes_)
print("Classes:", le.classes_)

# 2) Build X and reshape for CNN
X = X_windows_raw.astype(np.float32).reshape(-1, WINDOW_SIZE, 1)
print("X shape before shuffle:", X.shape)

# 3) Shuffle X and y_int **together**
from sklearn.utils import shuffle
X, y_int = shuffle(X, y_int, random_state=42)

# 4) Train–test split using integer labels for stratify
from sklearn.model_selection import train_test_split
X_train, X_test, y_train_int, y_test_int = train_test_split(
    X, y_int, test_size=0.2, random_state=42, stratify=y_int
)

print("X_train shape:", X_train.shape)
print("X_test shape: ", X_test.shape)

# 5) One-hot encode AFTER the split
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train_int, num_classes=num_classes)
y_test  = to_categorical(y_test_int,  num_classes=num_classes)
print("y_train shape:", y_train.shape)
print("y_test shape: ", y_test.shape)


# --------------- BUILD SIMPLE CNN-1D MODEL ---------------

model = Sequential([
    # Input
    Input(shape=(WINDOW_SIZE, 1)),

    # large receptive field – capture global patterns
    Conv1D(64, kernel_size=17, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    # still wide, more filters
    Conv1D(128, kernel_size=11, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    # medium kernels
    Conv1D(256, kernel_size=7, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    # dilated conv to catch subtle IR differences
    Conv1D(256, kernel_size=5, activation='relu',
           padding='same', dilation_rate=2),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    # small kernels for fine details
    Conv1D(128, kernel_size=3, activation='relu',
           padding='same', dilation_rate=2),
    BatchNormalization(),

    # Global pooling instead of Flatten--- fewer params, less overfitting
    GlobalAveragePooling1D(),

    # Dense head
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Early Call backs 

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_cnn_model.keras', 
    monitor='val_accuracy', 
    save_best_only=True
)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, lr_scheduler, checkpoint],
    verbose=1
)

# --------------- EVALUATE MODEL ---------------

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy: ", test_accuracy)

# --------------- CONFUSION MATRIX ---------------

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d',
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    cmap='Blues'
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('CNN Confusion Matrix')
plt.show()
