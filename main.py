import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.transform import resize

def loadData():
    # Load the oxford_flowers102 dataset
    dataset, info = tfds.load('oxford_flowers102', split='train', with_info=True)
    
    # Extract features and labels from the dataset
    features = []
    labels = []
    for example in dataset:
        image = example['image']  # Extract the image feature
        label = example['label']  # Extract the label
        
        # Preprocess the image (resize to a fixed shape)
        image = resize(image.numpy(), (224, 224))  # Resize to (224, 224), adjust as needed
        
        features.append(image)
        labels.append(label)
    
    # Convert lists to numpy arrays
    x = np.array(features)
    y = np.array(labels)
    
    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    return (x_train, y_train), (x_test, y_test)

def createLinearRegressionModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(224,224,3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu')
        ])
    return model

def resizeImages(images, target_shape):
    resizedImage = tf.image.resize(images, target_shape)
    return resizedImage

def run():
    print("Starting ML.\nLoading data...")
    (x_train, y_train), (x_test, y_test) = loadData()
    print("Creating model...")

    # targetShape = (224, 224)
    # x_train_resized = resizeImages(x_train, targetShape)
    # y_train_resized = resizeImages(y_train, targetShape)

    # Get the input shape after resizing
    # inputShape = targetShape + (3,)  # Assuming RGB images
    model = createLinearRegressionModel()
    model.summary()
    predictions = model(x_train[:1]).numpy()
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss_fn(y_train[:1], predictions).numpy()

    print("Compiling model...")
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])

    print("Training model...")
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    print("Finished training.")

if __name__ == "__main__":
    run()
