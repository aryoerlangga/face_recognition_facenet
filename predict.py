import pickle
from numpy import asarray
from PIL import Image, ImageDraw
from mtcnn.mtcnn import MTCNN
from utils import preprocessing, extract_face

def predict_image(directory, filename, model_facenet, model, out_encoder):

    image = Image.open(directory + filename)
    image.thumbnail((1500, 1500))
    image = image.convert('RGB')
    pixels = asarray(image)

    detector = MTCNN()
    results = detector.detect_faces(pixels)

    draw = ImageDraw.Draw(image)
    for result in results:
        x1, y1, width, height = result['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]

        image_ori = Image.fromarray(face)
        image_resize = image_ori.resize((160,160))
        face_array = asarray(image_resize)
        
        face = asarray([face_array])

        face = preprocessing(face, model_facenet) 

        yhat_class = model.predict(face)
        yhat_prob = model.predict_proba(face)

        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)

        left = x1
        top = y1
        right = x2
        bottom = y2
        
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))
        
        name = str(predict_names[0])
        text_width, text_height = draw.textsize(name)
        draw.text((left, bottom + 5), name, fill=(255, 255, 255, 255))
        
        name = str(round(class_probability,2)) + ' %'
        text_width, text_height = draw.textsize(name)
        draw.text((left, bottom + 15), name, fill=(255, 255, 255, 255))
        
    image.show()