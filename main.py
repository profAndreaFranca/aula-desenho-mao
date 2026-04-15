import cv2
import mediapipe as mp
import numpy as np
import math

# Indices dos landmarks usados pelo MediaPipe Hands.
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
INDEX_FINGER_PIP = 6
MIDDLE_FINGER_TIP = 12
MIDDLE_FINGER_PIP = 10
RING_FINGER_TIP = 16
RING_FINGER_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18

# Cores em BGR para um efeito neon/laser sobre o fundo preto.
LASER_CORE_COLOR = (255, 255, 255)
DRAWING_COLOR_PALETTE = [
    ((255, 80, 0), (255, 255, 0)),
    ((255, 0, 180), (255, 120, 255)),
    ((0, 180, 255), (0, 255, 255)),
    ((0, 255, 80), (160, 255, 160)),
    ((180, 0, 255), (230, 120, 255)),
    ((255, 255, 0), (255, 255, 180)),
]
WINDOW_NAME = "Desenho com Gestos"
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
DISPLAY_WINDOW_WIDTH = 1400
DISPLAY_WINDOW_HEIGHT = 900
DRAWING_SELECTION_RADIUS = 24

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_draw = mp.solutions.drawing_utils

drawings = []
active_drawing_index = None
selected_drawing_index = None
last_drawn_index = None
previous_point = None
previous_move_point = None

def clear_canvas(canvas):
    """Preenche a area de desenho com preto."""
    canvas[:] = 0

def finger_is_up(hand_landmarks, tip_id, pip_id):
    """
    Considera o dedo levantado quando a ponta esta acima da articulacao PIP.
    A imagem usa origem no topo, entao menor Y significa mais alto.
    """
    return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y
    
def landmark_point(hand_landmarks, landmark_id, frame_width, frame_height):
    """Converte um landmark normalizado do MediaPipe para ponto em pixels."""
    landmark = hand_landmarks.landmark[landmark_id]
    return (int(landmark.x * frame_width), int(landmark.y * frame_height))

def thumb_index_touching(hand_landmarks, frame_width, frame_height, threshold=35):
    """
    Detecta se polegar e indicador estao encostando com base na distancia
    entre suas pontas em pixels.
    """
    thumb_point = landmark_point(hand_landmarks, THUMB_TIP, frame_width, frame_height)
    index_point = landmark_point(hand_landmarks, INDEX_FINGER_TIP, frame_width, frame_height)

    distance = math.hypot(index_point[0] - thumb_point[0], index_point[1] - thumb_point[1])
    return distance < threshold

def thumb_index_midpoint(hand_landmarks, frame_width, frame_height):
    """Usa o meio da pinca como ponto de selecao e movimento."""
    thumb_point = landmark_point(hand_landmarks, THUMB_TIP, frame_width, frame_height)
    index_point = landmark_point(hand_landmarks, INDEX_FINGER_TIP, frame_width, frame_height)
    return (
        (thumb_point[0] + index_point[0]) // 2,
        (thumb_point[1] + index_point[1]) // 2,
    )

def create_empty_canvas(shape):
    """Cria um canvas vazio com o tamanho do frame atual."""
    return np.zeros(shape, dtype=np.uint8)

def move_canvas(canvas, delta_x, delta_y):
    """
    Move todo o desenho existente no canvas sem fazer o conteudo reaparecer
    do outro lado da tela.
    """
    translation_matrix = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
    return cv2.warpAffine(
        canvas,
        translation_matrix,
        (canvas.shape[1], canvas.shape[0]),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

#funções visuais do laser
def drawing_colors(drawing_index):
    """Retorna as cores do efeito laser para um desenho especifico."""
    return DRAWING_COLOR_PALETTE[drawing_index % len(DRAWING_COLOR_PALETTE)]

def draw_laser_line(canvas, start_point, end_point, colors):
    """
    Desenha um traco com varias camadas para simular um feixe de laser.
    """
    outer_color, inner_color = colors
    cv2.line(canvas, start_point, end_point, outer_color, 10)
    cv2.line(canvas, start_point, end_point, inner_color, 5)
    cv2.line(canvas, start_point, end_point, LASER_CORE_COLOR, 2) 
    
def render_laser_canvas(canvas):
    """
    Cria um brilho suave a partir do desenho para destacar o traco.
    """
    glow = cv2.GaussianBlur(canvas, (0, 0), sigmaX=10, sigmaY=10)
    glow = cv2.addWeighted(glow, 1.2, canvas, 0.3, 0)
    return cv2.add(glow, canvas)

#funções de composição e seleção de desenhos
def compose_drawings(drawings, frame_shape):
    """Combina todos os desenhos independentes em um unico canvas para exibir."""
    combined = create_empty_canvas(frame_shape)

    for drawing in drawings:
        combined = cv2.add(combined, drawing)

    return combined

def point_over_drawing(drawing, point, radius=DRAWING_SELECTION_RADIUS):
    """Verifica se o ponto esta sobre pixels do desenho, com uma pequena folga."""
    x, y = point
    height, width = drawing.shape[:2]

    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(width, x + radius + 1)
    y2 = min(height, y + radius + 1)

    if x1 >= x2 or y1 >= y2:
        return False

    region = drawing[y1:y2, x1:x2]
    return np.any(region)

def select_drawing(drawings, point, preferred_index=None):
    """
    Escolhe primeiro o desenho sob a mao; se nao houver nenhum, usa o mais
    recente como fallback.
    """
    for index in range(len(drawings) - 1, -1, -1):
        if point_over_drawing(drawings[index], point):
            return index

    if preferred_index is not None and 0 <= preferred_index < len(drawings):
        return preferred_index

    return None

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
