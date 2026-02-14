
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def load_data(data_path):
    print(f"Loading dataset from {data_path}...")
    df = pd.read_parquet(data_path) # Pandas < 2.0 might be installed
    
    # Extract features
    # 'channel_energy' column contains arrays/lists of shape [num_ut, fft_size]
    # We need to stack them into a big numpy array.
    
    X_list = df["channel_energy"].tolist()
    
    # Robust conversion
    X_data = []
    for i, item in enumerate(X_list):
        try:
            item_arr = np.stack(item).astype(np.float32)
        except Exception as e:
            print(f"Error converting item {i}: {e}")
            print(f"Type: {type(item)}")
            if isinstance(item, np.ndarray):
                print(f"Shape: {item.shape}")
                print(f"Dtype: {item.dtype}")
                # Try to print first element properties
                if item.size > 0:
                     print(f"Item[0] type: {type(item[0])}")
            if isinstance(item, list):
                print(f"Length: {len(item)}")
                if len(item) > 0:
                    print(f"First element type: {type(item[0])}")
            raise e
        X_data.append(item_arr)
        
    X = np.stack(X_data) # [num_samples, num_ut, fft_size]
    
    # Extract labels
    y_mask = np.array(df["active_ut_mask"].tolist(), dtype=np.float32)
    y_power = np.array(df["per_ut_power"].tolist(), dtype=np.float32)
    
    print(f"Data loaded: X={X.shape}, y_mask={y_mask.shape}, y_power={y_power.shape}")
    return X, y_mask, y_power

def create_model(input_shape, output_dim):
    # input_shape: (num_ut, fft_size)
    inputs = layers.Input(shape=input_shape)
    
    # Reshape for Conv2D: [num_ut, fft_size, 1]
    x = layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    # Convolutional layers
    # Convolve over frequency (dim 2) and users (dim 1)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((1, 2))(x) # Pool frequency
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((1, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x) # Pool everything to vector
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Heads
    mask_out = layers.Dense(output_dim, activation='sigmoid', name='mask_output')(x)
    power_out = layers.Dense(output_dim, activation='sigmoid', name='power_output')(x)
    
    model = models.Model(inputs=inputs, outputs=[mask_out, power_out])
    return model

def train_resource_manager(args):
    X, y_mask, y_power = load_data(args.data)
    
    input_shape = X.shape[1:] 
    output_dim = y_mask.shape[1]
    
    model = create_model(input_shape, output_dim)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss={
            'mask_output': 'binary_crossentropy',
            'power_output': 'mse'
        },
        loss_weights={'mask_output': 1.0, 'power_output': 0.5},
        metrics={'mask_output': 'accuracy'}
    )
    
    model.summary()
    
    # Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        args.output, monitor='val_loss', verbose=1, save_best_only=True
    )
    
    history = model.fit(
        X, [y_mask, y_power],
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        callbacks=[checkpoint]
    )
    
    print(f"Training complete. Model saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/rm_training_data.parquet", help="Path to aggregated training data")
    parser.add_argument("--output", type=str, default="models/cnn_resource_manager.h5", help="Output model path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    
    args = parser.parse_args()
    train_resource_manager(args)
