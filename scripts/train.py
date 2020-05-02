from keras.models import load_model
from sklearn.svm import SVC
import pickle
from utils_facenet import load_dataset, preprocessing
from sklearn.preprocessing import LabelEncoder

directory = 'images/train/'
X_train, y_train = load_dataset(directory)

model_facenet = load_model('model/facenet_keras.h5')

X_train = preprocessing(X_train, model_facenet)

out_encoder = LabelEncoder()
out_encoder.fit(y_train)
y_train = out_encoder.transform(y_train)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)


# Save label_encoder and model to pickle
encoder_filename = 'model/target_encoder.pkl'
pickle.dump(out_encoder, open(encoder_filename, 'wb'))

model_filename = 'model/facereco.pkl'
pickle.dump(model, open(model_filename, 'wb'))
