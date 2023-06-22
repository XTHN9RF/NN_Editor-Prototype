import tensorflow as tf
from dataset import model

test_dataset = tf.convert_to_tensor([
    [5.8, 2.7, 3.9, 1.2],
    [4.7, 3.2, 1.6, 0.2],
    [7.7, 2.6, 6.9, 2.3],
    [4.8, 3., 1.4, 0.1],
    [6.7, 2.5, 5.8, 1.8]
])

Class_Label = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

test_predictions = model(test_dataset, training=False)
test_probabilities = tf.nn.softmax(test_predictions)
predicted_class = tf.argmax(test_probabilities, 1)
predicted_class_label = tf.gather(Class_Label, predicted_class)

for ex, pred in zip(test_dataset, predicted_class_label):
    tf.print(ex, pred)
