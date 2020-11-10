import tensorflow as tf
from tensorflow.keras import models, layers, datasets


(train_data, train_labels), (test_data, test_labels) = datasets.cifar10.load_data()

train_data , test_data = train_data/255.0 , test_data/255.0


model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3), activation="relu"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer="adam", loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy", "mae"])
history = model.fit(train_data, train_labels, epochs = 10, validation_data=(test_data, test_labels))

model.evaluate(test_data, test_labels, verbose=2)

model.save("model.h5")