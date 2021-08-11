import os

import mediapipe as mp
import cv2
import requests
import shutil
from mediapipe.python.solutions.drawing_utils import DrawingSpec, BLUE_COLOR

if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    api_all_url = "https://tweetfilteredrestapi.ey.r.appspot.com/api/tweetfilteredV1/all"
    response = requests.get(api_all_url)
    print(response.json())

    IMAGE_URLS = []

    for post in response.json():
        if post['media'] is not None:
            for m in post['media']:
                IMAGE_URLS.append(m)

    for img in IMAGE_URLS:
        filename = img.split("/")[-1]

        r = requests.get(img, stream = True)

        if r.status_code == 200 and not "?tag=12" in filename:
            filename = "imgs/" + filename
            r.raw.decode_content = True

            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

        elif r.status_code == 200 and "?tag=12" in filename:
            filename = "vids/" + filename.strip("?tag=12")
            r.raw.decode_content = True

            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

        else:
            print("problem")

    IMAGE_FILES = ['test_person.jpg', 'test_people.JPEG']

    for file in os.listdir('imgs'):
        image = 'imgs/' + file
        IMAGE_FILES.append(image)

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=3,
        min_detection_confidence=0.5)

    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#Print and draw facemesh details on faces

        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        filtered_image = image.copy()
        square_image = image.copy()
        ironman_filter = cv2.imread('ironmanfilter.png', -1)
        filter_ratio = ironman_filter.shape[1] / ironman_filter.shape[0]

        #testimg = image.copy()
        height, width, _ = image.shape

        for face_landmarks in results.multi_face_landmarks:
            """for i in range(190,200):
                pt1 = face_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)

                cv2.circle(testimg, (x,y), 2, (100,0,0), -1)
                cv2.putText(testimg, str(i), (x,y), 1, 1, (0,0,0))
            """
            print('face_landmarks:', face_landmarks)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                landmark_drawing_spec=DrawingSpec(color=(100, 100, 0), thickness=2, circle_radius=1)
            )

            top = (int(face_landmarks.landmark[10].x * width), int(face_landmarks.landmark[10].y * height))
            bot = (int(face_landmarks.landmark[152].x * width), int(face_landmarks.landmark[152].y * height))
            left = (int(face_landmarks.landmark[34].x * width), int(face_landmarks.landmark[34].y * height))
            right = (int(face_landmarks.landmark[454].x * width), int(face_landmarks.landmark[454].y * height))
            center = (int(face_landmarks.landmark[195].x * width), int(face_landmarks.landmark[195].y * height))

            print('landmark10(top) coords: ', top)
            print('landmark152(bot) coords: ', bot)
            print('landmark34(left) coords: ', left)
            print('landmark454(right) coords: ', right)
            print('landmark195(center) coords: ', center)
            print(image.shape)

            filter_width = int((right[0] - left[0]) * 1.4)
            filter_height = int(filter_width / filter_ratio)

            top_left = (int(center[0] - filter_width/2), int(center[1] - filter_height/2))
            bot_right = (int(center[0] + filter_width/2), int(center[1] + filter_height/2))

            cv2.rectangle(square_image,
                          pt1 = (left[0], int(top[1] * 0.9)), pt2 = (right[0], bot[1]),
                          color = (255,0,255),
                          thickness = 10
                          )

            ironman_filter = cv2.resize(ironman_filter, (filter_width, filter_height))

            x_offset = int(top_left[0])
            y_offset = int(top_left[1])

            y1, y2 = y_offset, y_offset + ironman_filter.shape[0]
            x1, x2 = x_offset, x_offset + ironman_filter.shape[1]

            alpha_s = ironman_filter[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                filtered_image[y1:y2, x1:x2, c] = (alpha_s * ironman_filter[:, :, c] + alpha_l * filtered_image[y1:y2, x1:x2, c])


            #cv2.imshow('test', filtered_image)
            #cv2.waitKey(0)
            cv2.imwrite('tmp/annotated_image' + str(idx) + '.png', annotated_image)
            cv2.imwrite('tmp/filtered_image' + str(idx) + '.png', filtered_image)
            cv2.imwrite('tmp/square_image' + str(idx) + '.png', square_image)

            """

            cv2.imshow('test', testimg)
            cv2.waitKey(0)
            
            """