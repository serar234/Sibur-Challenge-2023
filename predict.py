import warnings
import pathlib
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import tensorflow_hub as hub

warnings.filterwarnings("ignore")
#
MODEL_FILE = pathlib.Path(__file__).parent.joinpath("nn30_1.h5")
model1 = load_model(MODEL_FILE)
MODEL_FILE = pathlib.Path(__file__).parent.joinpath("nn30_3.h5")
model2 = load_model(MODEL_FILE)
MODEL_FILE = pathlib.Path(__file__).parent.joinpath("nn30_4.h5")
model3 = load_model(MODEL_FILE)
# MODEL_FILE = pathlib.Path(__file__).parent.joinpath("nn30_5.h5")
# model4 = load_model(MODEL_FILE)
MODEL_FILE = pathlib.Path(__file__).parent.joinpath("nn30_2_2.h5")
model4 = load_model(MODEL_FILE)
# MODEL_FILE = pathlib.Path(__file__).parent.joinpath("nn30_5.h5")
# model = load_model(MODEL_FILE)


# pretrained_model = hub.KerasLayer('https://tfhub.dev/sayakpaul/deit_small_distilled_patch16_224_fe/1', trainable=False)
# input_tensor = tf.keras.layers.Input(shape=(None, None, 3))
# x = tf.keras.layers.Resizing(224, 224, interpolation="bilinear")(input_tensor)
# x = pretrained_model(x)
# output = x[0]
# fts_extract_norm = tf.keras.Model(inputs=input_tensor, outputs=output)



def construct_model_norm1():
    fts_extract = tf.keras.Sequential([
        tf.keras.layers.Resizing(224, 224, interpolation="bilinear"),
        tf.keras.layers.Rescaling(scale=1.0 / 255),
        # tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
        # hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/feature_vector/2", trainable=False)
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5", trainable=False)
    ])
    fts_extract.build([None, 240, 320, 3])
    return fts_extract

def construct_model_norm2():
    fts_extract = tf.keras.Sequential([
        tf.keras.layers.Resizing(260, 260, interpolation="bilinear"),
        tf.keras.layers.Rescaling(scale=1.0 / 255),
        # tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
        hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/feature_vector/2", trainable=False)
    ])
    fts_extract.build([None, 240, 320, 3])
    return fts_extract

def construct_model_move():
    fts_extract = tf.keras.Sequential([
        tf.keras.layers.Resizing(224, 224, interpolation="bilinear"),
        tf.keras.layers.Rescaling(scale=1.0 / 255),
        # tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5", trainable=False)
    ])
    fts_extract.build([None, 240, 320, 3])
    return fts_extract

fts_extract_norm1 = construct_model_norm1()
fts_extract_norm2 = construct_model_norm2()
fts_extract_move = construct_model_move()

def predict(clip: np.ndarray):
    clip = clip[1:]
    mean_frame = clip.mean(axis=0)

    Len_video = 10
    len_clip = len(clip)
    gap = 15
    if len_clip < Len_video * gap:
        gap = len_clip // Len_video
    inds = list(range(0, Len_video * gap, gap))

    thresh = []
    for frame in clip[inds]:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 2)
        thresh.append(frame)
    thresh = np.array(thresh)

    features_norm1 = fts_extract_norm1(clip[inds]).numpy()
    features_norm2 = fts_extract_norm2(clip[inds]).numpy()
    features_move = fts_extract_move(clip[inds] - mean_frame).numpy()
    X_data1 = np.array([features_norm1])
    X_data2 = np.array([features_norm2])
    X_data3 = np.array([features_move])
    X_data4 = np.array([thresh])

    # arg = model.predict([X_data2, X_data3, X_data4])[0].argmax()
    pred1 = model1.predict([X_data2, X_data3, X_data4])[0]
    pred2 = model2.predict([X_data2, X_data3, X_data4])[0]
    pred3 = model3.predict([X_data1, X_data3, X_data4])[0]
    # pred4 = model4.predict([X_data1, X_data2, X_data3, X_data4])[0]
    pred4 = model4.predict([X_data1, X_data3, X_data4])[0]
    # pred5 = model5.predict([X_data1, X_data3, X_data4])[0]
    arg = np.sum([pred1, pred2, pred3, pred4], axis=0).argmax()
    names = ['bridge_down', 'bridge_up', 'no_action', 'train_in_out']
    return names[arg]
