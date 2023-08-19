import mediapipe as mp
import cv2
import pyautogui as pg

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands()
cam = cv2.VideoCapture(1)

cam.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

com_width, com_height = pg.size()

compare = [(17,4,2),(0,8,5),(0,12,9),(0,16,13),(0,20,17)]

while(True):
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    if not ret: continue
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    if result.multi_handedness:
        
        
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks,result.multi_handedness):
            a = hand_landmarks.landmark
            
            finger_point = (int(a[8].x*com_width),int(a[8].y*com_height))
            pg.moveTo(finger_point[0], finger_point[1])
            
            label = handedness.classification[0].label
            coord = (int(a[0].x * width), int(a[0].y * height))
            cord = (int(a[0].x * width), int(a[0].y * height+50))
            cv2.putText(frame, label, coord,cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255), 1)
            mp_drawing.draw_landmarks(frame, hand_landmarks,mp_hands.HAND_CONNECTIONS)
            folding = ""
            for base,tip,mcp in compare:
                base_tip = (a[base].x-a[tip].x)**2 + (a[base].y-a[tip].y)**2 #점과 점사이 공식
                base_mcp = (a[base].x-a[mcp].x)**2 + (a[base].y-a[mcp].y)**2 #점과 점사이 공식
                folding += str(int(base_tip < base_mcp))
            cv2.putText(frame,folding,cord,cv2.FONT_HERSHEY_DUPLEX,2,(255,36,125),1) #1,0, 텍스트 출력
    cv2.imshow("cv2 test",frame) #화면 띄우기
    
    if cv2.waitKey(delay=50) == 27: break
    
cam.release()
cv2.destroyAllWindows()


