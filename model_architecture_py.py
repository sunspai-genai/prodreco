"""
Wide & Deep model architecture
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Dict, List
from loguru import logger


class WideDeepModel:
    """Wide & Deep neural network architecture"""
    
    def __init__(self, config: Dict, category: str):
        """
        Args:
            config: Configuration dictionary
            category: 'deposit' or 'loan'
        """
        self.config = config
        self.category = category
        
        if category == 'deposit':
            self.n_products = len(config['products']['deposits'])
        elif category == 'loan':
            self.n_products = len(config['products']['loans'])
        else:
            raise ValueError(f"Invalid category: {category}")
    
    def build(self, input_dim: int,
             wide_dim: int = None,
             deep_dims: List[int] = None,
             dropout_rate: float = None,
             l2_reg: float = None) -> Model:
        """
        Build Wide & Deep model
        
        Args:
            input_dim: Number of input features
            wide_dim: Dimension of wide component
            deep_dims: List of dimensions for deep component layers
            dropout_rate: Dropout rate
            l2_reg: L2 regularization factor
        
        Returns:
            Keras Model
        """
        # Use config defaults if not specified
        model_config = self.config['model']
        wide_dim = wide_dim or model_config['wide_dim']
        deep_dims = deep_dims or model_config['deep_dims']
        dropout_rate = dropout_rate or model_config['dropout_rate']
        l2_reg = l2_reg or model_config['l2_regularization']
        
        logger.info(f"Building Wide & Deep model for {self.category}")
        logger.info(f"  Input dim: {input_dim}")
        logger.info(f"  Output dim: {self.n_products}")
        logger.info(f"  Wide dim: {wide_dim}")
        logger.info(f"  Deep dims: {deep_dims}")
        logger.info(f"  Dropout: {dropout_rate}")
        logger.info(f"  L2 reg: {l2_reg}")
        
        # Input layer
        input_layer = layers.Input(
            shape=(input_dim,),
            name=f'{self.category}_input'
        )
        
        # ============================================================
        # WIDE COMPONENT (Linear model for memorization)
        # ============================================================
        wide = layers.Dense(
            wide_dim,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name=f'{self.category}_wide'
        )(input_layer)
        
        wide = layers.Dropout(dropout_rate, name=f'{self.category}_wide_dropout')(wide)
        
        # ============================================================
        # DEEP COMPONENT (DNN for generalization)
        # ============================================================
        deep = input_layer
        
        for idx, units in enumerate(deep_dims):
            deep = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(l2_reg),
                name=f'{self.category}_deep_{idx+1}'
            )(deep)
            
            # Batch normalization for training stability
            deep = layers.BatchNormalization(
                name=f'{self.category}_bn_{idx+1}'
            )(deep)
            
            # Dropout for regularization
            deep = layers.Dropout(
                dropout_rate,
                name=f'{self.category}_deep_dropout_{idx+1}'
            )(deep)
        
        # ============================================================
        # COMBINE WIDE & DEEP
        # ============================================================
        combined = layers.concatenate(
            [wide, deep],
            name=f'{self.category}_wide_deep_concat'
        )
        
        # ============================================================
        # OUTPUT LAYER (Multi-label classification)
        # ============================================================
        output = layers.Dense(
            self.n_products,
            activation='sigmoid',  # Sigmoid for multi-label
            name=f'{self.category}_output'
        )(combined)
        
        # Build model
        model = Model(
            inputs=input_layer,
            outputs=output,
            name=f'{self.category}_wide_deep_model'
        )
        
        logger.info(f"Model built successfully")
        logger.info(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def get_weighted_loss(self, pos_weight: float = 10.0):
        """
        Get weighted binary crossentropy loss for imbalanced data
        
        Args:
            pos_weight: Weight multiplier for positive class
        """
        def weighted_binary_crossentropy(y_true, y_pred):
            """
            Weighted BCE to penalize false negatives more heavily
            """
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
            
            # Binary cross entropy with positive class weighting
            bce = -(pos_weight * y_true * tf.math.log(y_pred) +
                   (1 - y_true) * tf.math.log(1 - y_pred))
            
            return tf.reduce_mean(bce)
        
        return weighted_binary_crossentropy
    
    def compile_model(self, model: Model,
                     learning_rate: float = None,
                     use_weighted_loss: bool = False,
                     pos_weight: float = 10.0) -> Model:
        """
        Compile model with optimizer and loss
        
        Args:
            model: Keras model
            learning_rate: Learning rate for Adam optimizer
            use_weighted_loss: Whether to use weighted loss
            pos_weight: Positive class weight
        """
        training_config = self.config['training'][self.category]
        learning_rate = learning_rate or training_config['learning_rate']
        
        # Choose loss function
        if use_weighted_loss:
            logger.info(f"Using weighted BCE loss (pos_weight={pos_weight})")
            loss = self.get_weighted_loss(pos_weight=pos_weight)
        else:
            logger.info("Using standard binary crossentropy loss")
            loss = 'binary_crossentropy'
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.BinaryAccuracy(name='binary_accuracy')
            ]
        )
        
        logger.info(f"Model compiled with learning_rate={learning_rate}")
        
        return model
    
    def get_callbacks(self, model_path: str) -> List:
        """
        Get training callbacks
        
        Args:
            model_path: Path to save best model
        """
        training_config = self.config['training'][self.category]
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=training_config['early_stopping_patience'],
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=training_config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.TerminateOnNaN()
        ]
        
        logger.info(f"Callbacks configured:")
        logger.info(f"  EarlyStopping: patience={training_config['early_stopping_patience']}")
        logger.info(f"  ReduceLROnPlateau: patience={training_config['reduce_lr_patience']}")
        logger.info(f"  ModelCheckpoint: {model_path}")
        
        return callbacks
