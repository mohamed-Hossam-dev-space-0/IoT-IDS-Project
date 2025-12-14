import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Conv1D, LSTM, GRU, Dropout, 
                                   Flatten, BatchNormalization, Input,
                                   Attention, Bidirectional, MaxPooling1D,
                                   GlobalAveragePooling1D, concatenate)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import warnings
warnings.filterwarnings('ignore')

class NeuralModels:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.models = {}
    
    def create_cnn_model(self, dropout_rate=0.3, learning_rate=0.001):
        """Create a Convolutional Neural Network for sequence classification"""
        model = Sequential([
            # First Conv Block
            Conv1D(64, 3, activation='relu', input_shape=self.input_shape,
                  kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            BatchNormalization(),
            Dropout(dropout_rate),
            MaxPooling1D(2),
            
            # Second Conv Block
            Conv1D(128, 3, activation='relu',
                  kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            BatchNormalization(),
            Dropout(dropout_rate),
            MaxPooling1D(2),
            
            # Third Conv Block
            Conv1D(256, 3, activation='relu',
                  kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Global pooling instead of flatten
            GlobalAveragePooling1D(),
            
            # Dense layers
            Dense(128, activation='relu',
                  kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            BatchNormalization(),
            Dropout(dropout_rate * 1.5),
            
            Dense(64, activation='relu',
                  kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            Dropout(dropout_rate),
            
            Dense(32, activation='relu'),
            Dropout(dropout_rate * 0.5),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')]
        )
        
        self.models['cnn'] = model
        return model
    
    def create_lstm_model(self, dropout_rate=0.3, learning_rate=0.001):
        """Create an LSTM model for sequential data classification"""
        model = Sequential([
            # First LSTM layer
            Bidirectional(LSTM(128, return_sequences=True,
                             kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                         input_shape=self.input_shape),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Second LSTM layer
            Bidirectional(LSTM(64, return_sequences=True,
                             kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Third LSTM layer
            LSTM(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Attention mechanism
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate * 0.5),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')]
        )
        
        self.models['lstm'] = model
        return model
    
    def create_gru_model(self, dropout_rate=0.3, learning_rate=0.001):
        """Create a GRU model (faster alternative to LSTM)"""
        model = Sequential([
            # GRU layers
            Bidirectional(GRU(128, return_sequences=True,
                            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                         input_shape=self.input_shape),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            GRU(64, return_sequences=True,
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            GRU(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Dense layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate * 0.5),
            
            Dense(32, activation='relu'),
            Dropout(dropout_rate * 0.3),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')]
        )
        
        self.models['gru'] = model
        return model
    
    def create_hybrid_model(self, dropout_rate=0.3, learning_rate=0.001):
        """Create a hybrid CNN-LSTM model"""
        inputs = Input(shape=self.input_shape)
        
        # CNN branch
        cnn = Conv1D(64, 3, activation='relu')(inputs)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(dropout_rate)(cnn)
        cnn = MaxPooling1D(2)(cnn)
        
        cnn = Conv1D(128, 3, activation='relu')(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(dropout_rate)(cnn)
        
        # LSTM branch
        lstm = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        lstm = BatchNormalization()(lstm)
        lstm = Dropout(dropout_rate)(lstm)
        
        lstm = LSTM(32)(lstm)
        lstm = BatchNormalization()(lstm)
        lstm = Dropout(dropout_rate)(lstm)
        
        # Combine branches
        combined = concatenate([GlobalAveragePooling1D()(cnn), lstm])
        
        # Dense layers
        x = Dense(128, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate * 1.5)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Dense(32, activation='relu')(x)
        x = Dropout(dropout_rate * 0.5)(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')]
        )
        
        self.models['hybrid'] = model
        return model
    
    def create_autoencoder(self, encoding_dim=16, learning_rate=0.001):
        """Create an autoencoder for anomaly detection"""
        # Encoder
        inputs = Input(shape=self.input_shape)
        
        # Reshape if needed
        if len(self.input_shape) == 2:
            x = Flatten()(inputs)
            original_dim = self.input_shape[0] * self.input_shape[1]
        else:
            x = inputs
            original_dim = self.input_shape[0]
        
        # Encoder layers
        encoded = Dense(128, activation='relu')(x)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dropout(0.1)(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder layers
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dropout(0.1)(decoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dropout(0.2)(decoded)
        
        if len(self.input_shape) == 2:
            decoded = Dense(original_dim, activation='sigmoid')(decoded)
            decoded = tf.keras.layers.Reshape(self.input_shape)(decoded)
        else:
            decoded = Dense(original_dim, activation='sigmoid')(decoded)
        
        # Autoencoder model
        autoencoder = Model(inputs=inputs, outputs=decoded)
        
        # Encoder model
        encoder = Model(inputs=inputs, outputs=encoded)
        
        # Compile autoencoder
        autoencoder.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.models['autoencoder'] = autoencoder
        self.models['encoder'] = encoder
        
        return autoencoder, encoder
    
    def create_attention_model(self, dropout_rate=0.3, learning_rate=0.001):
        """Create a model with attention mechanism"""
        inputs = Input(shape=self.input_shape)
        
        # Feature extraction with Conv1D
        x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Attention mechanism
        attention = Dense(128, activation='tanh')(x)
        attention = Dense(1, activation='linear')(attention)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(128)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        # Apply attention
        x = tf.keras.layers.multiply([x, attention])
        x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Dense(32, activation='relu')(x)
        x = Dropout(dropout_rate * 0.5)(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')]
        )
        
        self.models['attention'] = model
        return model
    
    def create_multimodal_model(self, dropout_rate=0.3, learning_rate=0.001):
        """Create a multimodal model for different feature types"""
        # Assuming input_shape is (timesteps, features)
        timesteps, features = self.input_shape
        
        # Create separate inputs for different feature groups
        statistical_input = Input(shape=(timesteps, 10), name='statistical_input')
        temporal_input = Input(shape=(timesteps, 8), name='temporal_input')
        protocol_input = Input(shape=(timesteps, 6), name='protocol_input')
        
        # Process statistical features
        stat_branch = Conv1D(32, 3, activation='relu')(statistical_input)
        stat_branch = BatchNormalization()(stat_branch)
        stat_branch = Dropout(dropout_rate)(stat_branch)
        stat_branch = GlobalAveragePooling1D()(stat_branch)
        
        # Process temporal features
        temp_branch = LSTM(32, return_sequences=True)(temporal_input)
        temp_branch = BatchNormalization()(temp_branch)
        temp_branch = Dropout(dropout_rate)(temp_branch)
        temp_branch = LSTM(16)(temp_branch)
        temp_branch = BatchNormalization()(temp_branch)
        temp_branch = Dropout(dropout_rate)(temp_branch)
        
        # Process protocol features
        proto_branch = Dense(16, activation='relu')(protocol_input)
        proto_branch = BatchNormalization()(proto_branch)
        proto_branch = Dropout(dropout_rate)(proto_branch)
        proto_branch = GlobalAveragePooling1D()(proto_branch)
        
        # Combine branches
        combined = concatenate([stat_branch, temp_branch, proto_branch])
        
        # Decision layers
        x = Dense(64, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate * 1.5)(x)
        
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = Model(
            inputs=[statistical_input, temporal_input, protocol_input],
            outputs=outputs
        )
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        self.models['multimodal'] = model
        return model
    
    def get_callbacks(self, patience=10, monitor='val_loss'):
        """Get training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, model, X_train, y_train, X_val=None, y_val=None,
                   epochs=50, batch_size=32, verbose=1):
        """Train a neural network model"""
        
        callbacks = self.get_callbacks()
        
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        return history
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        results = model.evaluate(X_test, y_test, verbose=0)
        
        # Create metrics dictionary
        metrics = {}
        for i, metric_name in enumerate(model.metrics_names):
            metrics[metric_name] = results[i]
        
        # Add additional metrics
        y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics['accuracy_sklearn'] = accuracy_score(y_test, y_pred)
        metrics['precision_sklearn'] = precision_score(y_test, y_pred, zero_division=0)
        metrics['recall_sklearn'] = recall_score(y_test, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_test, y_pred, zero_division=0)
        
        return metrics
    
    def get_model_summary(self, model_name):
        """Get summary of a specific model"""
        if model_name in self.models:
            model = self.models[model_name]
            model.summary()
            
            # Count parameters
            trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
            
            return {
                'model_name': model_name,
                'trainable_params': trainable_params,
                'non_trainable_params': non_trainable_params,
                'total_params': trainable_params + non_trainable_params,
                'layers': len(model.layers)
            }
        else:
            print(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
            return None

# Example usage
if __name__ == "__main__":
    # Create neural models with input shape (timesteps, features)
    input_shape = (24, 1)  # 24 features, 1 channel
    
    neural_models = NeuralModels(input_shape)
    
    # Create different models
    cnn_model = neural_models.create_cnn_model()
    lstm_model = neural_models.create_lstm_model()
    hybrid_model = neural_models.create_hybrid_model()
    
    # Get model summaries
    print("CNN Model Summary:")
    neural_models.get_model_summary('cnn')
    
    print("\nLSTM Model Summary:")
    neural_models.get_model_summary('lstm')