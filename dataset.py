from sklearn import datasets
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from model import model

iris = datasets.load_iris()
X = iris.data
y = iris.target

one_hot_Y = to_categorical(y)

X_train, X_val, Y_train, Y_val = train_test_split(X, one_hot_Y, test_size=0.2, random_state=42)

loss_fn = tf.keras.losses.CategoricalCrossentropy()
with tf.GradientTape() as tape:
    logits = model(X_train)
    loss_value = loss_fn(Y_train, logits)

grads = tape.gradient(loss_value, model.trainable_weights)

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

batch_size = 16
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_dataset = val_dataset.batch(batch_size)


def accuracy_scale():
    epochs = 5

    for epoch in range(epochs):

        print("\n")
        print(f"Epoch : {epoch + 1}")

        for step, (X_train, Y_train) in enumerate(train_dataset):
            # Iterate over the batches of the train dataset

            with tf.GradientTape() as tape:
                # During forward pass, Open GradientTape to calculate the gradient
                logits = model(X_train, training=True)  # Calcuate Model's Prediction
                loss_value = loss_fn(Y_train, logits)  # Calculate Loss value

            # Retrieve Gradient Calculation
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run the Optimizer, that update the model parameters to minimize the loss
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training accuracy metric
            train_acc_metric(Y_train, logits)

            # Print Log of loss value at every 5th step
            if step % 5 == 0:
                print(f"Training Loss at step {step} : {loss_value:.3f}")

        print()

        # Print training accuracy at the end of each epoch
        train_acc = train_acc_metric.result()
        print(f"Training Accuracy   : {train_acc:.3f}")
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run model on validation data at the end of each epoch
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val)
            val_acc_metric(y_batch_val, val_logits)

        # Display validation accuracy
        val_acc = val_acc_metric.result()
        # Reset validation metric
        val_acc_metric.reset_states()
        print(f"Validation Accuracy : {val_acc:.3f}")


def training():
    epochs = 5

    for epoch in range(epochs):

        for step, (X_train, Y_train) in enumerate(train_dataset):
            # Iterate over the batches of the train dataset

            with tf.GradientTape() as tape:
                # During forward pass, Open GradientTape to calculate the gradient
                logits = model(X_train, training=True)  # Calcuate Model's Prediction
                loss_value = loss_fn(Y_train, logits)  # Calculate Loss value

            # Retrieve Gradient Calculation
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run the Optimizer, that update the model parameters to minimize the loss
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training accuracy metric
            train_acc_metric(Y_train, logits)

        # Print training accuracy at the end of each epoch
        train_acc = train_acc_metric.result()

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run model on validation data at the end of each epoch
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val)
            val_acc_metric(y_batch_val, val_logits)

        # Display validation accuracy
        val_acc = val_acc_metric.result()
        # Reset validation metric
        val_acc_metric.reset_states()


def test():
    training()
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
