import cv2
import mediapipe as mp
import numpy as np
import time



# Initialize mediapipe modules for face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize mediapipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Open the video capture device
cap = cv2.VideoCapture(0)


while cap.isOpened():
    success, image = cap.read()
    start = time.time()
    #The start variable is used to measure the time taken for processing the frame.

    # Flip the image horizontally for selfie-view display and convert color space
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # converts the color space from BGR to RGB, which is the format expected by mediapipe.
    image.flags.writeable = False
    # When the writable flag is set to False, OpenCV does not need to make a copy of the image array 
    # whenever it is processed.

    # Process face mesh on the image
    results = face_mesh.process(image)

    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
    # initialize empty lists to store the 3D and 2D coordinates of the face landmarks
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                # Have 468 landmarks
                # Select specific facial landmarks of interest
                # 1 is tip of nose ,33 islefmost left eye , 263 is righmost right eye, 199 -chin centre,
                # (291,61)=rightmost, leftmost lips 
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        # Store 3D coordinates of the nose
                        nose_2d = (lm.x*img_w, lm.y*img_h)
                        nose_3d = (lm.x*img_w, lm.y*img_h, lm.z)

                    x, y = int(lm.x*img_w), int(lm.y*img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
                
               

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_length = 1 * img_w

            # Define camera matrix for projection
            cam_matrix = np.array([[focal_length, 0, img_h/2],
                                   [0, focal_length, img_w/2],
                                   [0, 0, 1]])
            
            # The destorion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve Perspective-n-Point (PnP) to estimate rotation and translation vectors
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix)

            # Convert rotation vector to rotation matrix and decompose into Euler angles
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # Determine the direction of gaze based on angles
            
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -5:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            # Project 3D nose point onto 2D image plane
            nose_3d_projection, jacobian = cv2.projectPoints(
                nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0]+y*10), int(nose_2d[1]-x*10))

            # Draw line representing gaze direction and display text
            cv2.line(image, p1, p2, (255, 0, 0), 3)
            cv2.putText(image, text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x:"+str(np.round(x, 2)), (500, 50+30*0),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y:"+str(np.round(y, 2)), (500, 50+30*1),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z:"+str(np.round(z, 2)), (500, 50+30*2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        
        end = time.time()
        totalTime = end-start
        if totalTime != 0:
            fps = 1/totalTime
            print("FPS: ", fps)

        # Display FPS on the image
            cv2.putText(image, f'FPS: {int(fps)}', (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # Draw face landmarks on the image
            mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=drawing_spec,
                                      connection_drawing_spec=drawing_spec)

    # Display the final image with head pose estimation
    cv2.imshow('Head Pose Estimation', image)

    # Break the loop on 'Esc' key press
    if (cv2.waitKey(5) & 0xFF == 27):
        break
    


# Release the video capture device and destroy windows
cap.release()
cv2.destroyAllWindows()
