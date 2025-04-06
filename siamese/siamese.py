"""
Siamese neural network module for keystroke authentication.
"""

from tensorflow.keras.layers import Input, Dense, Lambda, Add, LayerNormalization, Concatenate, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers

# Add new custom layer for similarity computation
class CosineSimilarity(layers.Layer):
    """Custom layer to compute cosine similarity"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        x1, x2 = inputs
        dot_product = tf.reduce_sum(x1 * x2, axis=-1, keepdims=True)
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=-1, keepdims=True))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=-1, keepdims=True))
        return dot_product / (norm1 * norm2)
    
    def get_config(self):
        return super().get_config()

def contrastive_loss(y_true, y_pred, margin=1.0, same_user_weight=2.0):
    """Modified contrastive loss with higher weight for same-user pairs"""
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    
    # Increase weight for same-user pairs
    weighted_y_true = K.cast(y_true, "float32") * same_user_weight
    return K.mean(weighted_y_true * square_pred + (1 - y_true) * margin_square)

def triplet_loss(margin=1.0):
    def loss(y_true, y_pred):
        # For binary classification output (similar/dissimilar)
        # y_pred is the similarity score between pairs
        # y_true is 1 for positive pairs, 0 for negative pairs
        
        # Convert the binary similarity into a triplet-like loss
        positive_dist = y_pred  # Distance for positive pairs
        negative_dist = 1 - y_pred  # Distance for negative pairs
        
        # Compute triplet loss using the similarity scores
        loss = K.maximum(0., margin + positive_dist - negative_dist)
        return K.mean(loss)
    return loss

class SiameseNetwork:
    """
    A Siamese Network for comparing keystroke dynamics feature vectors.
    """

    def __init__(self, base_model, head_model, loss_type='binary_crossentropy'):
        """
        :param loss_type: Using binary_crossentropy as default and only option
        """
        self.base_model = base_model
        self.head_model = head_model
        self.loss_type = loss_type   # Force binary_crossentropy
        self.input_shape = self.base_model.input_shape[1:]
        self.siamese_model = None
        self.__initialize_siamese_model()

    def compile(self, **kwargs):
        """
        Configures the model for training.
        All arguments are passed to the underlying Keras model compile function,
        but the loss function is determined by self.loss_type
        """
        kwargs['loss'] = self.get_loss_function()
        self.siamese_model.compile(**kwargs)

    def get_loss_function(self):
        """Using only binary_crossentropy loss"""
        return 'binary_crossentropy'

    def fit(self, x_train, y_train, batch_size=32, validation_data=None, **kwargs):
        """
        Trains the model using keystroke feature pairs.

        :param x_train: List of two arrays [input_a, input_b] containing feature pairs
        :param y_train: Labels (1 = same person, 0 = different people)
        :param batch_size: Batch size for training
        :param validation_data: Tuple of ([val_a, val_b], val_labels)
        """
        if validation_data is not None:
            x_val, y_val = validation_data
            validation_data = ([x_val[0], x_val[1]], y_val)
            
        return self.siamese_model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            validation_data=validation_data,
            **kwargs
        )

    def evaluate(self, x, y, batch_size=32, **kwargs):
        """
        Evaluates the Siamese network.

        :param x: Feature pairs (two feature vectors per sample)
        :param y: Labels (1 = same person, 0 = different people)
        :param batch_size: Batch size for evaluation (default: 32)
        :return: Evaluation results
        """
        return self.siamese_model.evaluate(x, y, batch_size=batch_size, **kwargs)

    def predict(self, x1, x2):
        return self.siamese_model.predict([x1, x2])


    def load_weights(self, checkpoint_path):
        """
        Load trained weights into the Siamese network.

        :param checkpoint_path: Path to the checkpoint file.
        """
        self.siamese_model.load_weights(checkpoint_path)

    def __initialize_siamese_model(self):
        """
        Initializes the Siamese Network using the base and head models.
        """
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)

        processed_a = self.base_model(input_a)
        processed_b = self.base_model(input_b)

        head = self.head_model([processed_a, processed_b])
        self.siamese_model = Model([input_a, input_b], head)


def create_residual_block(x, units, dropout_rate=0.3, l2_reg=0.01):
    """Creates a residual block with configurable regularization"""
    shortcut = x
    x = Dense(
        units, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(
        units, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(x)
    x = LayerNormalization()(x)
    
    if shortcut.shape[-1] != units:
        shortcut = Dense(units)(shortcut)
    
    return Add()([shortcut, x])

def create_base_network(input_shape, hidden_layers=[256, 128, 64], dropout_rate=0.3, l2_reg=0.01):
    """Enhanced base network with configurable architecture"""
    inputs = Input(shape=input_shape)
    
    # Initial transformation
    x = Dense(
        hidden_layers[0], 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(inputs)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Create residual blocks for each hidden layer size
    for units in hidden_layers:
        x = create_residual_block(x, units, dropout_rate, l2_reg)
    
    # Output transformation
    outputs = Dense(32, activation='relu')(x)
    
    return Model(inputs, outputs)

def create_head_model(embedding_shape, similarity_metric='cosine'):
    """Head model using custom cosine similarity layer"""
    embedding_a = Input(shape=(embedding_shape[-1],))
    embedding_b = Input(shape=(embedding_shape[-1],))
    
    # Cosine similarity using custom layer
    similarity = CosineSimilarity()([embedding_a, embedding_b])
    
    # Deep comparison network
    x = Dense(64, activation='relu')(similarity)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = LayerNormalization()(x)
    
    output = Dense(1, activation='sigmoid')(x)
    
    return Model([embedding_a, embedding_b], output)
