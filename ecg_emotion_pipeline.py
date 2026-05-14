
# FINAL ECG EMOTION + STRESS PIPELINE WITH GRAPHS

import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import butter, filtfilt, iirnotch, welch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, VotingRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

# matplotlib settings (FORCING DISPLAY)
plt.rcParams["figure.figsize"] = (10,4)
plt.rcParams["figure.dpi"] = 120

try:
    from xgboost import XGBClassifier, XGBRegressor
    _xgb_available = True
except:
    _xgb_available = False

np.random.seed(42)

conditions = ["Classical", "Rock", "Favorite"]
n_subjects = 90
duration = 15
sr = 500

signals = []
labels = []
stress_levels = []

# Simulated Data
for cond in conditions:
    for _ in range(n_subjects):

        if cond == "Classical":
            hr = np.random.normal(62, 3)
            stress = np.random.normal(0.28, 0.06)

        elif cond == "Favorite":
            hr = np.random.normal(82, 4)
            stress = np.random.normal(0.50, 0.07)

        else:  # Rock
            hr = np.random.normal(105, 5)
            stress = np.random.normal(0.78, 0.08)

        stress = np.clip(stress, 0, 1)
        ecg = nk.ecg_simulate(duration=duration, sampling_rate=sr, heart_rate=hr)

        # add noise
        ecg += np.random.normal(0, 0.012, len(ecg))
        ecg += 0.02 * np.sin(2*np.pi*0.25*np.arange(len(ecg))/sr)

        signals.append(ecg)
        labels.append(cond)
        stress_levels.append(stress)

print(f"Generated {len(signals)} ECG signals")

# Filters
def butter_bandpass_filter(sig, low=0.5, high=40, fs=500, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, sig)

def notch_filter(sig, freq=50, fs=500, Q=30):
    b, a = iirnotch(freq/(fs/2), Q)
    return filtfilt(b, a, sig)

def adaptive_baseline_remove(sig, window=300):
    baseline = nk.signal_smooth(sig, method="moving_average", size=window)
    return sig - baseline

filtered_signals = []
for s in signals:
    x = butter_bandpass_filter(s, fs=sr)
    x = notch_filter(x, fs=sr)
    x = adaptive_baseline_remove(x)
    filtered_signals.append(x)

print("Filtering done")

idx = 0
plt.figure()
plt.plot(signals[idx][:2000], label="Raw ECG")
plt.plot(filtered_signals[idx][:2000], label="Filtered ECG")
plt.legend()
plt.title("ECG Before vs After Filtering")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()


def extract_features(sig, fs=500):
    f = {}
    try:
        rpeaks, _ = nk.ecg_peaks(sig, sampling_rate=fs)
        r = np.where(rpeaks["ECG_R_Peaks"]==1)[0]
    except:
        r = []

    if len(r)>=2:
        rr = np.diff(r)/fs
        rr_ms = rr*1000
        f["MeanNN"] = np.mean(rr_ms)
        f["MeanBPM"] = 60000/np.mean(rr_ms)
        f["SDNN"] = np.std(rr_ms)
        f["RMSSD"] = np.sqrt(np.mean(np.diff(rr_ms)**2))
        f["pNN50"] = np.mean(np.abs(np.diff(rr_ms))>50)
    else:
        f["MeanNN"] = f["MeanBPM"] = f["SDNN"] = f["RMSSD"] = f["pNN50"] = np.nan

    # entropy
    try:
        f["SampEn"] = float(nk.entropy_sampen(sig))
    except:
        f["SampEn"] = np.nan

    # raw stats
    f["mean"] = np.mean(sig)
    f["std"] = np.std(sig)
    f["max"] = np.max(sig)
    f["min"] = np.min(sig)

    return f

data = []
for s in filtered_signals:
    data.append(extract_features(s, sr))

df = pd.DataFrame(data)
df["Music"] = labels
df["Stress"] = stress_levels

# clean
df = df.replace([np.inf,-np.inf], np.nan)
df = df.dropna(axis=1, how="all")

num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])

# Train/Test split
X = df.drop(["Music","Stress"], axis=1)
y_music = df["Music"]
y_stress = df["Stress"]

Xtrain, Xtest, yM_train, yM_test, yS_train, yS_test = train_test_split(
    X, y_music, y_stress, test_size=0.25, stratify=y_music
)

sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

# models
if _xgb_available:
    m1 = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    r1 = XGBRegressor()
else:
    m1 = GradientBoostingClassifier()
    r1 = GradientBoostingRegressor()

m2 = RandomForestClassifier()
r2 = RandomForestRegressor()

music = VotingClassifier([("a",m1),("b",m2)], voting="soft")
stress = VotingRegressor([("a",r1),("b",r2)])

music.fit(Xtrain, yM_train)
stress.fit(Xtrain, yS_train)

predM = music.predict(Xtest)
predS = np.clip(stress.predict(Xtest),0,1)

# Results
print("\n Music Accuracy:", accuracy_score(yM_test,predM)*100,"%")
print("\nClassification Report:\n", classification_report(yM_test,predM))

# Music Conf Matrix
cm = confusion_matrix(yM_test,predM, labels=conditions)
sns.heatmap(cm,annot=True,xticklabels=conditions,yticklabels=conditions)
plt.title("Music Classification Confusion Matrix")
plt.show()

# Stress Results
mse = mean_squared_error(yS_test,predS)
r2 = r2_score(yS_test,predS)
print(f"\n Stress MSE: {mse:.4f}, R²: {r2:.3f}")

def stress_cat(v):
    return "Low" if v<0.4 else ("Medium" if v<0.65 else "High")

true_c = [stress_cat(i) for i in yS_test]
pred_c = [stress_cat(i) for i in predS]

print("Stress Accuracy:", accuracy_score(true_c,pred_c)*100,"%")

cm2 = confusion_matrix(true_c,pred_c,labels=["Low","Medium","High"])
sns.heatmap(cm2,annot=True,xticklabels=["Low","Medium","High"],yticklabels=["Low","Medium","High"],cmap="Reds")
plt.title("Stress Confusion Matrix")
plt.show()

print("\n Done! Plots + Accuracies working 100%")

# -----------------------------
# MSE & Error Analysis Block
# -----------------------------
errors = yS_test - predS
abs_error = np.abs(errors)
percentage_error = (abs_error / yS_test) * 100

print("\n Error Analysis:")
print(f"Mean Absolute Error: {np.mean(abs_error):.4f}")
print(f"Mean Percentage Error: {np.mean(percentage_error):.2f}%")
print(f"Max Absolute Error: {np.max(abs_error):.4f}")
print(f"Max Percentage Error: {np.max(percentage_error):.2f}%")

# Histogram of error
plt.figure()
plt.hist(errors, bins=20)
plt.title("Error Distribution (Stress Prediction)")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.show()
