import tensorflow as tf
import tkinter as tk
import tkinter.simpledialog


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

    def get_config(self):
        """Method that returns the configuration of the layer."""
        return {"Кількість нейронів": self.units}


class CustomModel(tf.keras.Model):
    """Class that represents a custom model."""

    def __init__(self, num_layers, dropout_rate, output_units):
        """Constructor for the CustomModel class."""
        super(CustomModel, self).__init__()

        self.custom_layers = []
        for _ in range(num_layers):
            self.custom_layers.append(DenseLayer(50))
        self.num_layers = num_layers
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
        if dropout_rate is None:
            return
        if self.dropout_rate < 0 or self.dropout_rate > 1:
            self.dropout_rate = 0.2
            tk.simpledialog.messagebox.showerror("Неправильно введені дані",
                                                 "Відновлення значення за замовчуванням, введені дані мають бути в межах від 0 до 1")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def set_layer_units(self, layer_index, units):
        """Method that sets the units of a participate layer."""
        input_units = self.custom_layers[-1].units if self.custom_layers else 50
        if layer_index is None:
            return
        if layer_index - 1 < 0 or layer_index - 1 > self.num_layers:
            tk.simpledialog.messagebox.showerror("Помилка", "Шару не існує, введіть коректний номер")
            return
        if layer_index - 1 == 1:
            if self.custom_layers[0].units != input_units:
                tk.simpledialog.messagebox.showerror("Несумісні розміри вхідного та вихідного шарів")
                return
            if units is None:
                return
        if units <= 0:
            tk.simpledialog.messagebox.showerror("Помилка", "Введіть кількість, більшу за нуль")
            return
        output_units = units
        self.custom_layers[layer_index - 1] = DenseLayer(units)
        self.output_layer = DenseLayer(self.output_units)

    def add_layer(self, units):
        """Method that adds a layer to the model."""
        input_units = self.custom_layers[-1].units if self.custom_layers else 50
        if units is None:
            return
        if units <= 0:
            tk.simpledialog.messagebox.showerror("Помилка", "Введіть кількість, більшу за нуль")
            return
        output_units = units

        if self.custom_layers:
            if self.custom_layers[-1].units != input_units:
                tk.simpledialog.messagebox.showerror("Несумісні розміри вхідного та вихідного шарів")

        self.custom_layers.append(DenseLayer(units))
        self.output_layer = DenseLayer(self.output_units)
        self.num_layers = len(self.custom_layers)

    def remove_layer(self, layer_index):
        """Method that removes a layer from the model."""
        if layer_index is None:
            return
        if layer_index < 0 or layer_index > self.num_layers:
            tk.simpledialog.messagebox.showerror("Помилка", "Шару не існує, введіть коректний номер")
            return
        if layer_index == self.num_layers - 1:
            self.output_layer = DenseLayer(self.custom_layers[-2].units)
        self.custom_layers.pop(layer_index - 1)
        self.num_layers -= 1

    def get_layer(self, index=None):
        """Method that returns a participate layer."""
        if index is None:
            return
        if index == 0:
            return self.output_layer.get_config()
        if index < 0 or index > self.num_layers:
            tk.simpledialog.messagebox.showerror("Помилка", "Шару не існує, введіть коректний номер")
            return
        return self.custom_layers[index - 1].get_config()

    def get_config(self):
        """Method that returns the configuration of the model."""
        return {
            "Кількість шарів": self.num_layers,
            "Швидкість відпадання": self.dropout_rate,
            "Кількість нейронів вихідного шару": self.output_units
        }


model = CustomModel(1, 0.2, 3)
