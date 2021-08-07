import mediapipe as mp
import cv2

if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    IMAGE_FILES = ['test_person1.jpg']
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=3,
        min_detection_confidence=0.5) as face_mesh:

        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#Print and draw facemesh details on faces

            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                print('face_landmarks:', face_landmarks)
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                cv2.imshow('test_person1.jpg', annotated_image)
                cv2.waitKey(0)
                cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)