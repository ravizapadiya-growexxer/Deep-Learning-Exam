import argparse
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os
import sys

# -----------------------
# Model Definition
# -----------------------
class TabularModel(nn.Module):
    def __init__(self, input_dim):
        super(TabularModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# -----------------------
# Load Model
# -----------------------
def load_model(model_path, input_dim):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    model = TabularModel(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


# -----------------------
# Prediction Pipeline
# -----------------------
def predict(input_path, output_path, threshold):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Paths
    preprocessor_path = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
    model_path = os.path.join(BASE_DIR, "models", "model.pth")

    # -----------------------
    # Load Data
    # -----------------------
    print(f"[INFO] Loading test data from {input_path}")

    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)

    # -----------------------
    # Load Preprocessor
    # -----------------------
    print(f"[INFO] Loading preprocessor from {preprocessor_path}")

    if not os.path.exists(preprocessor_path):
        print(f"[ERROR] Preprocessor not found: {preprocessor_path}")
        print("👉 You must save it during training using joblib.dump()")
        sys.exit(1)

    preprocessor = joblib.load(preprocessor_path)

    # -----------------------
    # Transform Data
    # -----------------------
    try:
        X = preprocessor.transform(df)
    except Exception as e:
        print("[ERROR] Preprocessing failed.")
        print("👉 Possible reason: column mismatch between train & test data")
        print(f"Details: {e}")
        sys.exit(1)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    # -----------------------
    # Load Model
    # -----------------------
    print(f"[INFO] Loading model from {model_path}")
    model = load_model(model_path, X.shape[1])

    # -----------------------
    # Prediction
    # -----------------------
    print("[INFO] Running inference...")

    with torch.no_grad():
        probs = model(X_tensor).squeeze().numpy()

    preds = (probs >= threshold).astype(int)

    # -----------------------
    # Save Output
    # -----------------------
    output_df = pd.DataFrame({
        "patient_id": df.index,   # change if real ID exists
        "prediction": preds
    })

    output_dir = os.path.dirname(output_path)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    output_df.to_csv(output_path, index=False)

    print(f"[SUCCESS] Predictions saved to {output_path}")


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on test dataset")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to test.csv (e.g., data/test.csv)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save predictions (e.g., outputs/predictions.csv)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default=0.5)"
    )

    args = parser.parse_args()

    predict(args.input, args.output, args.threshold)