from zipfile import ZipFile
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping

# Extract data from the ZIP file
zf = ZipFile('path/to/your/file.zip')
zf.extractall('path/to/extracted/folder/')

# Load the training and test data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Display the first few rows of the training data
train.head()

# Separate features and labels in the training data
x_test = test
x = train.drop(columns=['label'])
y = train['label']

# Print some information about the loaded data
print("Test Data:")
print(x_test.head())
print("\nLabels:")
print(y.head())

# Encode labels using LabelEncoder and convert them to categorical format
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y)
y_train = to_categorical(y_train)

# Display the shape of the encoded labels
print("\nEncoded Labels:")
print(y_train)
print("Shape of Encoded Labels:", y_train.shape)

# Build a convolutional neural network (CNN) model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Specify early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=2, restore_best_weights=True)

# Fit the model with early stopping
history = model.fit(x.values.reshape(-1, 28, 28, 1), y_train, epochs=100, batch_size=256, validation_split=0.2,
                    callbacks=[early_stopping])


# Save the trained model in native Keras format
model.save('digit_recognition_cnn_model.keras')


# Visualize some initial data
plt.figure(figsize=(10, 10))
for i in range(25):  # Change the range to the number of samples you want to visualize
    plt.subplot(5, 5, i + 1)
    plt.imshow(x.values[i].reshape(28, 28))
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
plt.show()


# Visualize training and validation metrics over epochs
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

