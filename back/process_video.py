from modeling import transform_video
import numpy as np
import tensorflow as tf


actions = np.array(['Sena 1', 'Sena 2', 'Sena 3','Sena 4','Sena 5'])
model = tf.keras.saving.load_model("./rnn33.keras")

async def predict(frames_list):
    response = "None"
    X = transform_video(frames_list)
    X = np.expand_dims(X, axis=0)
    response = model.predict(X)
    response = actions[np.argmax(response)-1]
    print(response)
    return response