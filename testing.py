
import pandas as pd
import numpy as np
import cv2
import os, sys
from tqdm import tqdm

from keras import applications
from keras.models import Model
from keras.models import Sequential
from keras.models import model_from_json

lbl_idx = {'Checked': 0, 'Colourblock': 1, 'Melange': 2, 'Patterned': 3, 'Printed': 4, 'abstract': 5, 'floral': 6, 'graphic': 7, 'polka dots': 8, 'solid': 9, 'striped': 10, 'typography': 11}
idx_lbl = dict((id,lb) for lb, id in lbl_idx.items())

def read_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (200,200), interpolation = cv2.INTER_AREA)
    img=np.resize(img,(200,200,3))
    return img



json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights/weights-improvement.hdf5")
print("Loaded model from disk")




# for img_path in tqdm(test_img_path):
#     print (os.path.join(TEST_PATH ,img_path))
#     try:
#         test_img.append(read_img(os.path.join(TEST_PATH ,img_path)))
#     except:
#         test_img.append(read_img(os.path.join(TEST_PATH ,img_path)))

def test_csv():
    TEST_PATH = 'images/test/'

    test_img_path=os.listdir(TEST_PATH)
    prediction_output = []
    for img_path in test_img_path:
        output = {}
        test_img = []
        # print (os.path.join(TEST_PATH ,img_path))
        try:
            test_image = test_img.append(read_img(os.path.join(TEST_PATH ,img_path)))
        except:
            test_image = test_img.append(read_img(os.path.join(TEST_PATH, img_path)))

        x_test = np.array(test_img, np.float32) / 255.
        prediction = loaded_model.predict(x_test)
        predicted_label = np.argmax(prediction, axis=1)
        # print(img_path, '-->', predicted_label)
        output['input'] = img_path
        output['predicted_label'] = idx_lbl[predicted_label[0]]
        prediction_output.append(output)


    pd.DataFrame(prediction_output).to_csv('test_predictions.csv')


def predict_label(img):
    test_img = []
    test_img.append(img)
    x_test = np.array(test_img, np.float32) / 255.
    prediction = loaded_model.predict(x_test)
    predicted_label = np.argmax(prediction, axis=1)
    label = idx_lbl[predicted_label[0]]
    return label

# x_test = np.array(test_img, np.float32)/ 255.
# predictions = loaded_model.predict(x_test)
# predictions = np.argmax(predictions, axis=1)
# print(predictions)



