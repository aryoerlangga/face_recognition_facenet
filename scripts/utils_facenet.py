from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
from numpy import savez_compressed
from numpy import load
from os import listdir
from os.path import isdir
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from numpy import expand_dims
from random import choice
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer

def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    
    image_ori = Image.fromarray(face)
    image_resize = image_ori.resize(required_size)
    face_array = asarray(image_resize)
    
    return face_array, image_ori

def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
        path = directory + filename
        face, image = extract_face(path)
        faces.append(face)
        
    return faces

def load_dataset(directory):
    X, y = list(), list()
    for subdir in listdir(directory):
    # for subdir in ['ben_afflek', 'elton_john']:
        path = directory + subdir + '/'
        if not isdir(path):
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
        
    return asarray(X), asarray(y)

def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)

    return yhat[0]

def preprocessing(X, model_facenet):
    X_prep = list()
    for face_pixels in X:
        embedding = get_embedding(model_facenet, face_pixels)
        X_prep.append(embedding)
    X_prep = asarray(X_prep)

    in_encoder = Normalizer(norm='l2')
    X_prep = in_encoder.transform(X_prep)

    return X_prep

def draw_image_with_boxes(directory, filename, result_list):
    data = plt.imread(directory + filename)
    plt.figure(figsize=(20,10))
    plt.imshow(data)
    ax = plt.gca()
    for result in result_list:
        x, y, width, height = result['box']
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(rect)
        for key, value in result['keypoints'].items():
            dot = Circle(value, radius=2, color='red')
            ax.add_patch(dot)
    plt.show()