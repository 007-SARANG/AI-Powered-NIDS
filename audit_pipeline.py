import pandas as pd
from ml.preprocessing.pipeline import prepare_data, NIDSPreprocessor
import psutil

df = pd.read_csv('data/raw/dataset.csv', low_memory=False)
print(f"RAM Usage before dedup: {df.memory_usage(deep=True).sum() / (1024**3):.2f} GB")

print(f"Classes before dedup:\n{df['Label'].value_counts()}")

X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df, target_col='Label', holdout_attack='DoS Hulk')

print("\n--- Train Label Dist ---")
print(y_train.value_counts())
print("\n--- Val Label Dist ---")
print(y_val.value_counts())
print("\n--- Test Label Dist ---")
print(y_test.value_counts())

print(f"\nDoS Hulk in Train: {'DoS Hulk' in y_train.values}")
print(f"DoS Hulk in Val: {'DoS Hulk' in y_val.values}")
print(f"DoS Hulk in Test: {'DoS Hulk' in y_test.values}")

# Memory estimation
print(f"\nRAM Usage after split (Train/Val/Test combined roughly): " 
      f"{(X_train.memory_usage(deep=True).sum() + X_val.memory_usage(deep=True).sum() + X_test.memory_usage(deep=True).sum()) / (1024**3):.2f} GB")
