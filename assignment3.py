import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")
import warnings, os, time
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, LeaveOneOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

TRAIN_PATH  = "/kaggle/input/competitions/playground-series-s6e4/train.csv"
TEST_PATH   = "/kaggle/input/competitions/playground-series-s6e4/test.csv"
SUBMISSION  = "submission.csv"
TARGET_COL  = "Irrigation_Need"
RANDOM_SEED = 42
N_FOLDS     = 3
RESULTS_DIR = "results"
SAMPLE_SIZE = 50000

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("STEP 1 – Loading Data")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

print(f"Train : {train_df.shape}  |  Test : {test_df.shape}")
print(f"\nTarget distribution:\n{train_df[TARGET_COL].value_counts()}")

print("\n" + "=" * 60)
print("STEP 2 – Feature Engineering")
print("=" * 60)

def engineer_features(df):
    df = df.copy()

    df["Temp_Humidity"] = df["Temperature_C"] * df["Humidity"]
    df["Moisture_Rainfall"] = df["Soil_Moisture"] * df["Rainfall_mm"]
    df["Temp_Moisture_Ratio"] = df["Temperature_C"] / (df["Soil_Moisture"] + 1)
    df["Rain_Evap_Index"] = df["Rainfall_mm"] / (df["Temperature_C"] + 1)
    df["Humidity_Wind"] = df["Humidity"] * df["Wind_Speed_kmh"]
    df["Carbon_pH"] = df["Organic_Carbon"] * df["Soil_pH"]
    df["EC_Moisture"] = df["Electrical_Conductivity"] * df["Soil_Moisture"]
    df["Sun_Wind_Ratio"] = df["Sunlight_Hours"] / (df["Wind_Speed_kmh"] + 1)
    df["Prev_vs_Rainfall"] = df["Previous_Irrigation_mm"] / (df["Rainfall_mm"] + 1)
    df["Area_Moisture"] = df["Field_Area_hectare"] * df["Soil_Moisture"]

    df["Water_Stress"] = (
        (df["Temperature_C"] * df["Sunlight_Hours"]) /
        (df["Rainfall_mm"] + df["Soil_Moisture"] + 1)
    )

    df["Soil_Quality"] = (
        df["Organic_Carbon"] /
        (df["Electrical_Conductivity"] + 1)
    )

    df["Aridity_Index"] = (
        df["Temperature_C"] /
        (df["Rainfall_mm"] + 1)
    )

    return df

train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

print(f"Features after engineering: {train_df.shape[1]}")

print("\n" + "=" * 60)
print("STEP 3 – Preprocessing")
print("=" * 60)

y_raw = train_df[TARGET_COL].copy()
test_ids = test_df["id"] if "id" in test_df.columns else test_df.index

drop_cols = [TARGET_COL] + [c for c in ["id", "Id", "ID"] if c in train_df.columns]

train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=[c for c in ["id", "Id", "ID"] if c in test_df.columns])

le = LabelEncoder()
y = le.fit_transform(y_raw)

print(f"Classes : {le.classes_}")

N_CLASSES = len(le.classes_)

for col in train_df.select_dtypes(include=[np.number]).columns:
    fill = train_df[col].median()
    train_df[col].fillna(fill, inplace=True)
    test_df[col].fillna(fill, inplace=True)

for col in train_df.select_dtypes(exclude=[np.number]).columns:
    fill = train_df[col].mode()[0]
    train_df[col].fillna(fill, inplace=True)
    test_df[col].fillna(fill, inplace=True)

cat_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()

for col in cat_cols:
    col_le = LabelEncoder()

    train_df[col] = col_le.fit_transform(train_df[col].astype(str))

    test_df[col] = test_df[col].astype(str).map(
        lambda x, enc=col_le: x if x in enc.classes_ else enc.classes_[0]
    )

    test_df[col] = col_le.transform(test_df[col])

scaler = StandardScaler()

X_scaled = scaler.fit_transform(train_df)
X_test_scaled = scaler.transform(test_df)

X = train_df.values
X_test_raw = test_df.values

print(f"Final feature matrix : {X.shape}")

X_tr, X_val, Xs_tr, Xs_val, y_tr, y_val = train_test_split(
    X,
    X_scaled,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

sample_idx = np.random.default_rng(RANDOM_SEED).choice(
    len(X_tr),
    min(SAMPLE_SIZE, len(X_tr)),
    replace=False
)

X_tr_s = X_tr[sample_idx]
Xs_tr_s = Xs_tr[sample_idx]
y_tr_s = y_tr[sample_idx]

results_log = {}

def plot_cm(y_true, y_pred, name, classes):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(
        figsize=(max(6, N_CLASSES), max(5, N_CLASSES - 1))
    )

    ConfusionMatrixDisplay(
        cm,
        display_labels=classes
    ).plot(
        ax=ax,
        cmap="Blues",
        colorbar=True
    )

    ax.set_title(f"Confusion Matrix – {name}")

    plt.tight_layout()

    path = f"{RESULTS_DIR}/cm_{name.replace(' ', '_')}.png"

    plt.savefig(path, dpi=100)
    plt.close()

    print(f"  CM saved: {path}")

def evaluate(model, Xtr, Xv, ytr, name):
    print(f"\n--- {name} ---")

    t0 = time.time()

    model.fit(Xtr, ytr)

    preds = model.predict(Xv)
    acc = accuracy_score(y_val, preds)

    print(f"  Val Accuracy : {acc:.4f}  ({time.time()-t0:.1f}s)")
    print(classification_report(y_val, preds, target_names=le.classes_))

    plot_cm(y_val, preds, name, le.classes_)

    results_log[name] = acc

    return model

def quick_cv(model, Xu, yu, name):
    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    scores = cross_val_score(
        model,
        Xu,
        yu,
        cv=skf,
        scoring="accuracy",
        n_jobs=-1
    )

    print(f"  {N_FOLDS}-Fold CV : {scores.mean():.4f} ± {scores.std():.4f}")

    return scores.mean()

print("\n" + "=" * 60)
print("STEP 4 – Required Models (Assignment)")
print("=" * 60)

print("\n[1] Decision Tree")

dt = evaluate(
    DecisionTreeClassifier(
        max_depth=None,
        random_state=RANDOM_SEED
    ),
    X_tr_s,
    X_val,
    y_tr_s,
    "Decision Tree"
)

print("\n[2] Decision Tree (Tuned)")

dt_t = evaluate(
    DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=5,
        random_state=RANDOM_SEED
    ),
    X_tr_s,
    X_val,
    y_tr_s,
    "Decision Tree Tuned"
)

quick_cv(
    DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=5,
        random_state=RANDOM_SEED
    ),
    X_tr_s,
    y_tr_s,
    "Decision Tree Tuned"
)

print("\n[3] Naive Bayes")

nb = evaluate(
    GaussianNB(),
    Xs_tr,
    Xs_val,
    y_tr,
    "Naive Bayes"
)

quick_cv(
    GaussianNB(),
    Xs_tr,
    y_tr,
    "Naive Bayes"
)

print("\n[4] Logistic Regression")

lr = evaluate(
    LogisticRegression(
        max_iter=500,
        solver="saga",
        n_jobs=-1,
        random_state=RANDOM_SEED
    ),
    Xs_tr,
    Xs_val,
    y_tr,
    "Logistic Regression"
)

quick_cv(
    LogisticRegression(
        max_iter=300,
        solver="saga",
        n_jobs=-1
    ),
    Xs_tr,
    y_tr,
    "Logistic Regression"
)

print("\n[5] K-Means Classifier")

class KMeansClassifier:
    def __init__(self, n_clusters, rs=42):
        self.km = KMeans(
            n_clusters=n_clusters,
            random_state=rs,
            n_init=10
        )
        self.map_ = {}

    def fit(self, X, y):
        self.km.fit(X)

        for c in np.unique(self.km.labels_):
            self.map_[c] = np.bincount(
                y[self.km.labels_ == c]
            ).argmax()

        return self

    def predict(self, X):
        return np.array([
            self.map_[c]
            for c in self.km.predict(X)
        ])

km = evaluate(
    KMeansClassifier(N_CLASSES),
    Xs_tr_s,
    Xs_val,
    y_tr_s,
    "K-Means Classifier"
)

print("\n[6] KNN (k=7)")

knn = evaluate(
    KNeighborsClassifier(
        n_neighbors=7,
        n_jobs=-1
    ),
    Xs_tr_s,
    Xs_val,
    y_tr_s,
    "KNN k=7"
)

print("\n[7] Random Forest")

rf = evaluate(
    RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=RANDOM_SEED,
        n_jobs=-1
    ),
    X_tr,
    X_val,
    y_tr,
    "Random Forest"
)

quick_cv(
    RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=RANDOM_SEED
    ),
    X_tr,
    y_tr,
    "Random Forest"
)

print("\n[8] Gradient Boosting (sample)")

gb = evaluate(
    GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_SEED,
        subsample=0.8
    ),
    X_tr_s,
    X_val,
    y_tr_s,
    "Gradient Boosting"
)

print("\n" + "=" * 60)
print("STEP 5 – High Score Models")
print("=" * 60)

if HAS_XGB:
    print("\n[9] XGBoost (tuned)")

    xgb = evaluate(
        XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        X_tr,
        X_val,
        y_tr,
        "XGBoost Tuned"
    )

    quick_cv(
        XGBClassifier(
            n_estimators=200,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1
        ),
        X_tr,
        y_tr,
        "XGBoost"
    )

if HAS_LGBM:
    print("\n[10] LightGBM (tuned)")

    lgbm = evaluate(
        LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=127,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1
        ),
        X_tr,
        X_val,
        y_tr,
        "LightGBM Tuned"
    )

    quick_cv(
        LGBMClassifier(
            n_estimators=300,
            num_leaves=63,
            verbose=-1,
            n_jobs=-1
        ),
        X_tr,
        y_tr,
        "LightGBM"
    )

print("\n" + "=" * 60)
print("STEP 6 – LOOCV (200 samples)")
print("=" * 60)

loo_idx = np.random.default_rng(RANDOM_SEED).choice(
    len(X),
    200,
    replace=False
)

loo_scores = cross_val_score(
    DecisionTreeClassifier(
        max_depth=10,
        random_state=RANDOM_SEED
    ),
    X[loo_idx],
    y[loo_idx],
    cv=LeaveOneOut(),
    scoring="accuracy",
    n_jobs=-1
)

print(
    f"LOOCV Accuracy (DT, n=200): "
    f"{loo_scores.mean():.4f} ± {loo_scores.std():.4f}"
)

print("\n" + "=" * 60)
print("STEP 7 – Model Comparison")
print("=" * 60)

results_df = (
    pd.DataFrame.from_dict(
        results_log,
        orient="index",
        columns=["Val Accuracy"]
    )
    .sort_values("Val Accuracy", ascending=False)
)

print(results_df.to_string())

plt.figure(figsize=(12, 6))

results_df["Val Accuracy"].plot(
    kind="barh",
    color="steelblue",
    edgecolor="black"
)

plt.xlabel("Validation Accuracy")
plt.title("Model Comparison – All Models")
plt.xlim(0, 1.05)

plt.tight_layout()

plt.savefig(
    f"{RESULTS_DIR}/model_comparison.png",
    dpi=100
)

plt.close()

print("\n" + "=" * 60)
print("STEP 8 – Retrain Best Model on ALL Data")
print("=" * 60)

best_name = results_df.index[0]

print(
    f"Best model : {best_name}  "
    f"({results_df.iloc[0,0]:.4f})"
)

scaled_models = {
    "Naive Bayes",
    "Logistic Regression",
    "KNN k=7",
    "K-Means Classifier"
}

use_scaled = best_name in scaled_models

X_final = X_scaled if use_scaled else X
X_test_fin = X_test_scaled if use_scaled else X_test_raw

if "LightGBM" in best_name and HAS_LGBM:
    final = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )

elif "XGBoost" in best_name and HAS_XGB:
    final = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

elif "Random Forest" in best_name:
    final = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

elif "Logistic Regression" in best_name:
    final = LogisticRegression(
        max_iter=500,
        solver="saga",
        n_jobs=-1
    )

elif "Naive Bayes" in best_name:
    final = GaussianNB()

else:
    final = DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=5,
        random_state=RANDOM_SEED
    )

print(f"Fitting on {len(X_final):,} rows...")

t0 = time.time()

final.fit(X_final, y)

print(f"Done in {time.time()-t0:.1f}s")

print("\n" + "=" * 60)
print("STEP 9 – Generating Submission")
print("=" * 60)

preds = le.inverse_transform(
    final.predict(X_test_fin)
)

submission_df = pd.DataFrame({
    "id": test_ids,
    TARGET_COL: preds
})

submission_df.to_csv(
    SUBMISSION,
    index=False
)

print(f"Saved: {SUBMISSION}")
print(submission_df[TARGET_COL].value_counts())

if hasattr(final, "feature_importances_"):
    fi = pd.Series(
        final.feature_importances_,
        index=train_df.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(12, 6))

    fi.head(20).plot(
        kind="bar",
        color="darkorange",
        edgecolor="black"
    )

    plt.title("Top 20 Feature Importances")

    plt.tight_layout()

    plt.savefig(
        f"{RESULTS_DIR}/feature_importance.png",
        dpi=100
    )

    plt.close()

    print(f"\nTop 10 features:\n{fi.head(10)}")

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(results_df.to_string())

print(f"\nBest Model   : {best_name}")
print(f"Val Accuracy : {results_df.iloc[0,0]:.4f}")
print(f"Submission   : {SUBMISSION}")

print("=" * 60)