from tkinter import *
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('digit_recognition_cnn_model.keras')

root = Tk()
root.title("Digit recognizer")
root.geometry("205x350")


def clear_canvas():
    canvas.delete("all")  # Clear all items on the canvas
    # Reset the canvas_array to all zeros
    global canvas_array
    canvas_array = np.zeros((200, 200), dtype=np.uint8)


def paint(event):
    x = event.x
    y = event.y
    # Draw a white rectangle around the mouse position
    canvas.create_rectangle(x - 10, y - 10, x + 10, y + 10, fill='white', outline='white')
    # Draw white ovals on a black background on the canvas array
    canvas_array[y - 10:y + 10, x - 10:x + 10] = 255


def predict_digit():
    # Convert the canvas content to a NumPy array
    img = Image.fromarray(canvas_array)
    img = img.resize((28, 28))
    # Normalize the image
    img_array = np.array(img)
    # Make prediction
    prediction = model.predict(img_array.reshape(-1, 28, 28, 1))
    # Display the predicted digit on the Tkinter window
    predicted_digit = np.argmax(prediction)
    predicted_label.config(text=f"The predicted digit is: {predicted_digit}")


# Create an array to represent the canvas
canvas_array = np.zeros((200, 200), dtype=np.uint8)

canvas = Canvas(root, height=200, width=200, bg="black")
canvas.grid(row=0, column=0)

# Bind the paint function to the mouse motion event
canvas.bind("<B1-Motion>", paint)

predict = Button(root, text="Predict", command=predict_digit)
predict.grid(row=1, column=0, pady=10)

clearButton = Button(root, text="Clear Canvas", command=clear_canvas)
clearButton.grid(row=2, column=0, pady=10)

# Label to display the predicted digit
predicted_label = Label(root, text="")
predicted_label.grid(row=3, column=0, pady=10)

root.mainloop()
