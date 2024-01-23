import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('handwritten.h5')

digit_number = 0
while os.path.isfile(f"digit/{digit_number}.png"):
    try:
        img = cv2.imread(f"digit/{digit_number}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"Prediction: {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(e)
    finally:
        digit_number += 1


