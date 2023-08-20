import mediapipe as mp
import cv2
import pyautogui as pg


mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

faces = mp_face.FaceMesh(refine_landmarks=True)
cam = cv2.VideoCapture(1)

cam.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

com_width, com_height = pg.size()
compare = [(17,4,2),(0,8,5),(0,12,9),(0,16,13),(0,20,17)]
past_delta_x, past_delta_y = 0,0

while(True):
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    if not ret: continue
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = faces.process(image)
    if result.multi_face_landmarks:
        #mp_drawing.draw_landmarks(frame,result.multi_face_landmarks[0],mp_face.FACEMESH_TESSELATION)
        lm = result.multi_face_landmarks[0].landmark

        left_eye_center = (int(lm[468].x*w),int(lm[468].y*h))
        left_eye_upper = (int(lm[470].x*w),int(lm[470].y*h))
        radius1 = int((((left_eye_center[0]-left_eye_upper[0])**2+(left_eye_center[1]-left_eye_upper[1])**2)**0.5))
        cv2.circle(frame,left_eye_center,radius1,(255,0,255),1)
        eye_right = (int(lm[133].x*w),int(lm[133].y*h))
        eye_left = (int(lm[33].x*w),int(lm[33].y*h))
        eye_center = ((eye_left[0]+eye_right[0])//2,(eye_left[1]+eye_right[1])//2)
        cv2.circle(frame,eye_center,30,(255,255,255),1)
        
        delta_x = left_eye_center[0] - eye_center[0]
        delta_y = left_eye_center[1] - eye_center[1]
        if delta_x > 3 and past_delta_x == -1:
            past_delta_x = 1
            pg.keyUp('left')
            pg.keyDown('right')
        elif delta_x < -3 and past_delta_x == 1:
            past_delta_x = -1
            pg.keyUp('right')
            pg.keyDown('left')
        elif -3<delta_x <3 and past_delta_x != 0:
            past_delta_x = 0
            pg.keyUp('left')
            pg.keyUp('right')
            
        if delta_y > 3 and past_delta_y == -1:
            past_delta_y = 1
            pg.keyUp('down')
            pg.keyDown('up')
            
        elif delta_y > -3 and past_delta_y == 1:
            past_delta_y = -1
            pg.keyUp('up')
            pg.keyDown('down')
            
        elif -3<delta_y<3 and past_delta_y != 0:
            past_delta_y = 0
            pg.keyUp('up')
            pg.keyUp('down')
        

    cv2.imshow("cv2 test",frame) #화면 띄우기
    
    if cv2.waitKey(delay=50) == 27: break
    
cam.release()
cv2.destroyAllWindows()


