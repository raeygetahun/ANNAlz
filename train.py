import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# Constants
CATEGORIES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
SEED = 32
WIDTH = 224
HEIGHT = 224
DEPTH = 3
INPUT_SHAPE = (WIDTH, HEIGHT, DEPTH)

# Paths
dataPath = "./data"
trainPath = os.path.join(dataPath, "train")
testPath = os.path.join(dataPath, "val")
CATEGORIES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
NUM_CATEGORIES = len(CATEGORIES)

# Create DataFrame for training data
trainData = []
for category_id, category in enumerate(CATEGORIES):
    categoryPath = os.path.join(trainPath, category)
    for file in os.listdir(categoryPath):
        filePath = os.path.join(categoryPath, file)
        trainData.append([filePath, category_id, category])
trainDataFrame = pd.DataFrame(trainData, columns=["file", "category_id", "category"])

# Shuffle the data
trainDataFrame = trainDataFrame.sample(frac=1)

# Splitting into train and validation sets
X = trainDataFrame.drop(columns="category_id")
y = trainDataFrame["category_id"]
x_train, x_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=4
)

# Concatenate train and validation sets
train_final = pd.concat([x_train, y_train], axis=1)
validation_final = pd.concat([x_valid, y_valid], axis=1)

# Reset index
train_final = train_final.reset_index(drop=True)
validation_final = validation_final.reset_index(drop=True)


# Data generators
datagen = ImageDataGenerator(rescale=1.0 / 255)

trainGenerator = datagen.flow_from_dataframe(
    dataframe=train_final,
    directory=dataPath,
    x_col="file",
    y_col="category",
    batch_size=32,
    seed=SEED,
    shuffle=True,
    class_mode="categorical",
    target_size=(HEIGHT, WIDTH),
)

validationGenerator = datagen.flow_from_dataframe(
    dataframe=validation_final,
    directory=dataPath,
    x_col="file",
    y_col="category",
    batch_size=32,
    seed=SEED,
    shuffle=True,
    class_mode="categorical",
    target_size=(HEIGHT, WIDTH),
)

# Callback
earlyStop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
callbacks = [earlyStop]

# Optimizer
modelOptimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Model architecture
def createModel():
    # Load ResNet152V2 with pre-trained ImageNet weights
    resnet_model = tf.keras.applications.ResNet152V2(
        weights="imagenet", include_top=False, input_shape=(HEIGHT, WIDTH, DEPTH)
    )

    # Freeze the first 100 layers
    for layer in resnet_model.layers[:100]:
        layer.trainable = False

    # Global average pooling layer
    x = tf.keras.layers.GlobalAveragePooling2D()(resnet_model.output)

    # Additional layers for fine-tuning
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Output layer
    output = tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")(x)

    # Compile the model
    alzModel = tf.keras.Model(inputs=resnet_model.input, outputs=output)
    alzModel.compile(
        loss="categorical_crossentropy", optimizer=modelOptimizer, metrics=["accuracy"]
    )

    return alzModel


# Create and summarize model
alzModel = createModel()

# Train model
outcome = alzModel.fit(
    trainGenerator,
    validation_data=validationGenerator,
    epochs=20,
    batch_size=32,
    callbacks=callbacks,
)

#save the model
alzModel.save('AlzModel.keras')


# plotter
def plotStats(output):
    loss = output.history["loss"]
    val_loss = output.history["val_loss"]

    accuracy = output.history["accuracy"]
    val_accuracy = output.history["val_accuracy"]

    epochs = range(len(output.history["loss"]))

    # Plot loss
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


# Plot training stats
plotStats(outcome)
