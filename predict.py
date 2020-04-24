from keras.models import load_model
import pickle
from utils import preprocessing, extract_face
from numpy import asarray
from numpy import expand_dims
import matplotlib.pyplot as plt

def predict_image(directory, filename, model_facenet, model, out_encoder):
    face_ori = extract_face(directory + filename)
    face = asarray([face_ori])

    face = preprocessing(face, model_facenet)

    yhat_class = model.predict(face)
    yhat_prob = model.predict_proba(face)

    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)

    # plot
    plt.imshow(face_ori)
    title = '%s (%.3f)' % (predict_names[0], class_probability)
    plt.title(title)
    plt.show()