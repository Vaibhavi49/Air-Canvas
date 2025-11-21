import cv2
import mediapipe as mp
import numpy as np
import time


CAM_ID = 0
DRAW_THICKNESS = 8
ERASE_THICKNESS = 50
SELECTION_COOLDOWN = 0.4  

PALETTE = [
    (0, 100, (0, 0, 255), "Red"),
    (100, 200, (255, 0, 0), "Blue"),
    (200, 300, (0, 255, 0), "Green"),
    (300, 400, (0, 255, 255), "Yellow"),
    (400, 500, (0, 0, 0), "Eraser"),
]


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.65, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(CAM_ID)
time.sleep(0.5)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not open webcam")

frame = cv2.flip(frame, 1)
h, w, c = frame.shape

canvas = np.zeros_like(frame)

prev_x, prev_y = 0, 0
current_color = (0, 0, 255)  
is_eraser = False
last_selection_time = 0

def draw_palette(img, selected_color, highlight_idx=None):
    """Draws the palette UI on top of img (in-place)."""
    for i, (x1, x2, col, label) in enumerate(PALETTE):
    
        cv2.rectangle(img, (x1, 0), (x2, 60), col if label != "Eraser" else (255,255,255), -1)
    
        cv2.rectangle(img, (x1, 0), (x2, 60), (50,50,50), 2)
    
        text = label if label != "Eraser" else "Eraser"
        cv2.putText(img, text, (x1+8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0) if label!="Eraser" else (0,0,0), 2)
    
        if label != "Eraser" and col == selected_color:
            cv2.rectangle(img, (x1, 0), (x2, 60), (255,255,255), 3)
        if highlight_idx is not None and highlight_idx == i:
            cv2.rectangle(img, (x1+3, 3), (x2-3, 57), (0,255,255), 4)

def fingers_up(hand_landmarks):
   
    i_tip_y = hand_landmarks.landmark[8].y
    i_pip_y = hand_landmarks.landmark[6].y
    m_tip_y = hand_landmarks.landmark[12].y
    m_pip_y = hand_landmarks.landmark[10].y
    index_up = i_tip_y < i_pip_y
    middle_up = m_tip_y < m_pip_y
    return index_up, middle_up

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    draw = frame.copy()

    draw_palette(draw, current_color)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    highlight_index = None

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
       
        mp_draw.draw_landmarks(draw, handLms, mp_hands.HAND_CONNECTIONS)

       
        ix = int(handLms.landmark[8].x * w)
        iy = int(handLms.landmark[8].y * h)
        mx = int(handLms.landmark[12].x * w)
        my = int(handLms.landmark[12].y * h)

        index_up, middle_up = fingers_up(handLms)

     
        if index_up and middle_up:
            prev_x, prev_y = 0, 0  
            sel_x, sel_y = ix, iy
            cv2.circle(draw, (sel_x, sel_y), 10, (0,255,255), cv2.FILLED)

         
            if sel_y < 60:
               
                for i, (x1, x2, col, label) in enumerate(PALETTE):
                    if x1 <= sel_x < x2:
                        highlight_index = i
                       
                        now = time.time()
                        if now - last_selection_time > SELECTION_COOLDOWN:
                            last_selection_time = now

                            if label == "Eraser":
                                is_eraser = True
                                current_color = (0,0,0)
                            else:
                                is_eraser = False
                                current_color = col

            cv2.putText(draw, "Selection Mode", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,200), 2)

 
        elif index_up and not middle_up:

            cv2.circle(draw, (ix, iy), 8, current_color if not is_eraser else (0,0,0), cv2.FILLED)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = ix, iy

            if is_eraser:
                cv2.line(canvas, (prev_x, prev_y), (ix, iy), (0,0,0), ERASE_THICKNESS)
            else:
                cv2.line(canvas, (prev_x, prev_y), (ix, iy), current_color, DRAW_THICKNESS)

            prev_x, prev_y = ix, iy
            cv2.putText(draw, "Drawing Mode", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)

        else:
            prev_x, prev_y = 0, 0

    else:
        prev_x, prev_y = 0, 0

 
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(draw, draw, mask=mask_inv)
    fg = cv2.bitwise_and(canvas, canvas, mask=mask)

    output = cv2.add(bg, fg)

    if highlight_index is not None:
        draw_palette(output, current_color, highlight_index)

    cv2.imshow("Air Paint", output)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: 
        break
    elif key == ord('c'): 
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()
