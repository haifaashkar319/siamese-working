"""
Siamese neural network module for keystroke authentication.
"""

from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers

class SiameseNetwork:
    """
    A Siamese Network for comparing keystroke dynamics feature vectors.
    """

    def __init__(self, base_model, head_model):
        """
        Constructs the Siamese model.

        Structure:
        -------------------------------------------------------------------
        input1 -> base_model |
                             --> embedding --> head_model --> binary output
        input2 -> base_model |
        -------------------------------------------------------------------

        :param base_model: The embedding model (converts input to feature representation).
        :param head_model: The comparator model (determines similarity between two embeddings).
        """
        self.base_model = base_model
        self.head_model = head_model
        self.input_shape = self.base_model.input_shape[1:]

        # Initialize the Siamese model
        self.siamese_model = None
        self.__initialize_siamese_model()

    def compile(self, *args, **kwargs):
        """
        Configures the model for training.
        Passes all arguments to the underlying Keras model compile function.
        """
        self.siamese_model.compile(*args, **kwargs)

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

    def predict(self, x):
        """
        Predicts the similarity score between two keystroke feature vectors.

        :param x: Feature pair (two feature vectors).
        :return: Similarity score (0 = different, 1 = identical).
        """
        return self.siamese_model.predict([x[:, 0], x[:, 1]])

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


def create_base_network(input_shape):
    """
    Creates the base network that transforms raw keystroke feature vectors into embeddings.
    
    :param input_shape: Shape of input feature vectors.
    :return: Keras Model.
    """
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu')
    ])
    return model


def create_head_model(embedding_shape):
    """
    Creates the head model that compares two feature embeddings.

    :param embedding_shape: Shape of feature embeddings.
    :return: Keras Model.
    """
    embedding_a = Input(shape=(embedding_shape[-1],))  
    embedding_b = Input(shape=(embedding_shape[-1],))  

    # 🔹 Define the Lambda function with explicit output shape
    def l1_distance(vectors):
        x, y = vectors
        return K.abs(x - y)

    distance = Lambda(l1_distance, output_shape=lambda input_shape: input_shape[0])([embedding_a, embedding_b])

    # 🔹 Final similarity score
    output = Dense(1, activation='sigmoid')(distance)

    return Model([embedding_a, embedding_b], output)
