import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

dataDirectory = './data'

data = []
labels = []

for dir_ in os.listdir(dataDirectory):
    for img_path in os.listdir(os.path.join(dataDirectory, dir_)):
        data_temp = []
        img = cv2.imread(os.path.join(dataDirectory, dir_,img_path))
        img_rgb =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    data_temp.append(x)
                    data_temp.append(y)

            data.append(data_temp)
            labels.append(dir_)


#Outputting to data
output = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, output)
output.close()