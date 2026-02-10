
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def train_cnn(
    data_path: str,
    output_model_path: str,
    epochs: int = 10,
    batch_size: int = 32,
    validation_split: float = 0.2
):
    """
    Train a CNN for resource allocation.
    
    Args:
        data_path: Path to the parquet dataset.
        output_model_path: Path to save the trained model.
    """
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} records.")
    
    # Check if we have multiple tries per sample
    if "try_index" not in df.columns:
        print("Dataset does not contain 'try_index'. Assuming single allocation per sample (not optimal for training).")
        # In this case, we just train on what we have, but it might just be random allocations.
        # We need "good" allocations. 
        # If the dataset is just random allocations, we can't learn "good" allocation unless we have a quality metric
        # and we weigh samples by quality or filter them.
        # But let's assume the user uses the multi-try generation.
        best_df = df
    else:
        # Group by sample_index and find the best allocation
        # Metric: Minimize BER (and maybe maximize throughput or minimize power)
        # Simple metric: minimize BER.
        # If BER is equal, minimize power?
        # Let's use Utility = (1 - avg_ber)
        
        # Add utility column
        # Using avg_ber across users
        df["utility"] = 1.0 - df["avg_ber"]
        
        # Find best index per sample
        best_indices = df.groupby("sample_index")["utility"].idxmax()
        best_df = df.loc[best_indices].reset_index(drop=True)
        print(f"Selected {len(best_df)} best allocations from {len(df)} trials.")
        
    # Prepare inputs and targets
    # Input: Channel Energy Profile
    # The column contains lists, convert to numpy array
    # Shape: [num_samples, num_ut, fft_size]
    # Use tolist() to ensure nested lists are converted correctly to 3D array
    X_list = best_df["channel_energy"].tolist()
    print(f"DEBUG: len(X_list): {len(X_list)}")
    print(f"DEBUG: len(X_list[0]): {len(X_list[0])}") 
    
    # Robust loading of jagged/object arrays from Pandas/Parquet
    X_data = []
    for item in X_list:
        # item is likely (8,) array of (512,) arrays or similar
        # Convert to standard list of lists of floats
        item_arr = np.array(item) # ensure mostly numpy
        # If it's an object array of arrays, we need to stack/convert inner
        if item_arr.dtype == object or len(item_arr.shape) == 1:
            # Flatten/Stack to (8, 512)
            # Try to convert each element to list/array
             sub_items = [np.array(x) for x in item]
             item_arr = np.stack(sub_items)
        
        X_data.append(item_arr)
        
    X = np.array(X_data, dtype=np.float32)
    print(f"DEBUG: X.shape: {X.shape}")
    print(f"DEBUG: X.dtype: {X.dtype}")
    
    # Check shape
    input_shape = X.shape[1:] # (num_ut, fft_size)
    print(f"Input shape: {input_shape}")
    
    # Targets
    
    # Targets
    # 1. Active UT Mask (Binary)
    y_mask = np.array(best_df["active_ut_mask"].tolist())
    # 2. Per UT Power (Continuous 0-1)
    y_power = np.array(best_df["per_ut_power"].tolist())
    
    output_dim = y_mask.shape[1] # num_ut
    
    # Build Model
    # We use a simple CNN/Dense architecture
    # Input: [num_ut, fft_size]
    # Treat as image? or 1D sequence per user?
    # CNN approach:
    # Reshape to [num_ut, fft_size, 1] for Conv2D? 
    # Or [fft_size, num_ut] for Conv1D?
    # Actually, [num_ut, fft_size] is like an image.
    
    inputs = layers.Input(shape=input_shape)
    
    # Reshape for Conv2D: [num_ut, fft_size, 1]
    last_dim = input_shape[-1]
    input_reshaped = layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    # Conv layers
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_reshaped)
    x = layers.MaxPooling2D((1, 2))(x) # Pool over frequency
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((1, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output Heads
    # Mask Head: Sigmoid for binary classification
    mask_out = layers.Dense(output_dim, activation='sigmoid', name='mask_output')(x)
    
    # Power Head: Sigmoid for 0-1 range
    # Note: If mask is 0, power doesn't matter much, but we train it to predict the chosen power.
    power_out = layers.Dense(output_dim, activation='sigmoid', name='power_output')(x)
    
    model = models.Model(inputs=inputs, outputs=[mask_out, power_out])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss={
            'mask_output': 'binary_crossentropy',
            'power_output': 'mse'
        },
        loss_weights={
            'mask_output': 1.0,
            'power_output': 0.5
        },
        metrics={
            'mask_output': 'accuracy',
            'power_output': 'mae'
        }
    )
    
    model.summary()
    
    # Train
    print("Starting training...")
    history = model.fit(
        X, [y_mask, y_power],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )
    
    # Save
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    model.save(output_model_path)
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN Resource Allocator")
    parser.add_argument("--data", type=str, default="data/dataset.parquet", help="Path to input dataset")
    parser.add_argument("--output", type=str, default="models/cnn_resource_manager.h5", help="Path to save model")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    train_cnn(
        args.data,
        args.output,
        args.epochs,
        args.batch_size
    )
