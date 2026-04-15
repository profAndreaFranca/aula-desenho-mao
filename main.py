import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_draw = mp.solutions.drawing_utils
canvas = None
prev_x = None
prev_y = None

def clear_canvas(canvas):
    canvas[:] = 0

while True:
    success, frame = cap.read()

    if not success:
        print("Erro ao capturar imagem da webcam.")
        break

    frame = cv2.flip(frame, 1)
    
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    display = np.zeros_like(frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(
                display,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
        )
        
        h, w, _ = frame.shape
        
        index_tip = hand_landmarks.landmark[8]
        
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)
        #print(x, y)
        cv2.circle(display,(x,y),10,(0,255,0),-1)
        
        if prev_x is None or prev_y is None:
            prev_x, prev_y = x, y
        
        cv2.line(canvas,(prev_x, prev_y),(x, y),(255, 255, 0), 3)
        prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None

    output = cv2.add(display, canvas)
    cv2.imshow("Desenho com Gestos", output)

    key = cv2.waitKey(1) & 0xFF 
    
    if key == ord('c'):
        clear_canvas(canvas)
        prev_x, prev_y = None, None
    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
