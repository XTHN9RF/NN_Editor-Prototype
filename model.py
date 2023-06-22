import tensorflow as tf


class DenseLayer(tf.keras.layers.Layer):
    """Class that represents a custom dense layer."""

    def __init__(self, units):
        """Constructor for the DenseLayer class."""
        super(DenseLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        """Method that builds the layer."""
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        """Method that executes the layer."""
        return tf.matmul(inputs, self.w) + self.b


class CustomModel(tf.keras.Model):
    """Class that represents a custom model."""

    def __init__(self, num_layers, dropout_rate, output_units):
        """Constructor for the CustomModel class."""
        super(CustomModel, self).__init__()

        self.custom_layers = []
        for _ in range(num_layers):
            self.custom_layers.append(DenseLayer(50))

        self.dropout_rate = dropout_rate
        self.output_units = output_units
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.output_layer = DenseLayer(output_units)

    def call(self, input_tensor, training=False):
        """Method that executes the model."""
        x = input_tensor
        for layer in self.custom_layers:
            x = layer(x)
            x = tf.nn.relu(x)
            if training:
                x = self.dropout(x, training=training)
        x = self.output_layer(x)
        x = tf.nn.softmax(x)
        return x

    def set_dropout_rate(self, dropout_rate):
        """Method that sets the dropout rate."""
        self.dropout_rate = dropout_rate
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def set_layer_units(self, layer_index, units):
        """Method that sets the units of a participate layer."""
        self.custom_layers[layer_index] = DenseLayer(units)

    def get_layer_units(self, layer_index):
        """Method that gets the units of a participate layer."""
        return self.custom_layers[layer_index].units

    def add_layer(self, units):
        """Method that adds a layer to the model."""
        self.custom_layers.append(DenseLayer(units))

    def get_config(self):
        """Method that returns the configuration of the model."""
        return {
            "count of layers": len(self.custom_layers),
            "dropout rate": self.dropout_rate,
            "output layer units": self.output_units
        }


model = CustomModel(1, 0.2, 3)
