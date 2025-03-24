import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands #initialize mediapipe hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

colors = [(255, 0, 255), (0, 255, 0), (0, 0, 255), (255, 0, 0)] #colors for drawing (pink, green, red, blue)
color_names = ['PINK', 'GREEN', 'RED', 'BLUE']
current_color = colors[0]

button_width, button_height = 150, 100 #button dimensions
button_positions = [(20, 50 + i * (button_height + 30)) for i in range(7)]
button_labels = ['CLEAR', 'ERASER'] + color_names + ['SAVE']

drawing = False #drawing parameters
prev_x, prev_y = 0, 0
thickness = 5
canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255 
SMOOTHING_FACTOR = 5 #adaptive pinch detection(may vaires)
smoothing_buffer = []

history = [] 
redo_stack = []

def is_thumb_index_touching(hand_landmarks): #check if thumb touches index finger (own)
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    palm_base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    hand_size = np.sqrt((thumb_tip.x - palm_base.x) ** 2 + (thumb_tip.y - palm_base.y) ** 2)

    distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

    return distance < hand_size * 0.25


def check_button(x, y):
    for i, (bx, by) in enumerate(button_positions):
        if bx < x < bx + button_width and by < y < by + button_height:
            return i
    return -1

def smooth_coordinates(x, y):
    global smoothing_buffer

    if len(smoothing_buffer) >= SMOOTHING_FACTOR:
        smoothing_buffer.pop(0)

    smoothing_buffer.append((x, y))

    avg_x = int(np.mean([p[0] for p in smoothing_buffer]))
    avg_y = int(np.mean([p[1] for p in smoothing_buffer]))

    return avg_x, avg_y

def save_canvas(): #save canvas as .png
    cv2.imwrite("air_canvas.png", canvas)

def save_history(): #asve history of redo/undo
    history.append(canvas.copy())

def undo(): #undo
    if history:
        redo_stack.append(canvas.copy())
        return history.pop()
    return canvas

def redo(): #redo
    if redo_stack:
        save_history()
        return redo_stack.pop()
    return canvas

# Main
def draw_air_canvas():
    global current_color, drawing, prev_x, prev_y, canvas, thickness

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # main 2
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])
                x, y = smooth_coordinates(x, y)

                if is_thumb_index_touching(hand_landmarks):
                    if not drawing:
                        save_history()
                        prev_x, prev_y = x, y
                        drawing = True

                    # Erasering tool
                    if current_color == (255, 255, 255):
                        cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, thickness * 3)
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, thickness)

                    prev_x, prev_y = x, y
                else:
                    drawing = False

                button_index = check_button(x, y)
                if button_index == 0:  # Clear button
                    save_history()
                    canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
                elif button_index == 1:  # Eraser button
                    current_color = (255, 255, 255)
                elif button_index in range(2, 6):  # Color buttons
                    current_color = colors[button_index - 2]
                elif button_index == 6:  # Save button
                    save_canvas()

        # UI buttons
        for i, (bx, by) in enumerate(button_positions):
            color = (0, 0, 0) if i in [0, 1, 6] else colors[i - 2]
            cv2.rectangle(frame, (bx, by), (bx + button_width, by + button_height), color, 2)
            cv2.putText(frame, button_labels[i], (bx + 10, by + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        canvas_resized = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))
        combined = np.hstack((frame, canvas_resized))

        cv2.imshow("saf", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # press 'ESC' to exit
            break
        elif key == ord('u'):  # undo
            canvas = undo()
        elif key == ord('r'):  # redo
            canvas = redo()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    draw_air_canvas()
