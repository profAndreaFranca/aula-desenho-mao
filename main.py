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

def main():
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    if not cap.isOpened():
        print("Nao foi possivel acessar a webcam.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WINDOW_WIDTH, DISPLAY_WINDOW_HEIGHT)
    
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

    while True:
        success, frame = cap.read()
        if not success:
            print("Falha ao capturar frame da webcam.")
            break

        # Espelha a imagem para um comportamento mais natural.
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        display = np.zeros_like(frame)

        current_mode = "Aguardando gesto"
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                display,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                None,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            )

            index_up = finger_is_up(hand_landmarks, INDEX_FINGER_TIP, INDEX_FINGER_PIP)
            middle_up = finger_is_up(hand_landmarks, MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP)
            ring_up = finger_is_up(hand_landmarks, RING_FINGER_TIP, RING_FINGER_PIP)
            pinky_up = finger_is_up(hand_landmarks, PINKY_TIP, PINKY_PIP)
            pinch_move = thumb_index_touching(hand_landmarks, frame_width, frame_height)
            open_hand = index_up and middle_up and ring_up and pinky_up

            current_point = landmark_point(
                hand_landmarks,
                INDEX_FINGER_TIP,
                frame_width,
                frame_height,
            )
            pinch_point = thumb_index_midpoint(hand_landmarks, frame_width, frame_height)
            
            if open_hand:
                current_mode = "Movendo mao sem desenhar"
                active_drawing_index = None
                selected_drawing_index = None
                previous_point = None
                previous_move_point = None
            elif pinch_move:
                active_drawing_index = None
                previous_point = None

                if previous_move_point is None:
                    selected_drawing_index = select_drawing(
                        drawings,
                        pinch_point,
                        preferred_index=last_drawn_index,
                    )

                if previous_move_point is not None:
                    delta_x = pinch_point[0] - previous_move_point[0]
                    delta_y = pinch_point[1] - previous_move_point[1]
                    if selected_drawing_index is not None:
                        drawings[selected_drawing_index] = move_canvas(
                            drawings[selected_drawing_index],
                            delta_x,
                            delta_y,
                        )

                previous_move_point = pinch_point
                if selected_drawing_index is not None:
                    current_mode = f"Movendo desenho {selected_drawing_index + 1}"
                else:
                    current_mode = "Nenhum desenho selecionado"
            elif pinky_up and not index_up and not middle_up and not ring_up:
                drawings.clear()
                active_drawing_index = None
                selected_drawing_index = None
                last_drawn_index = None
                previous_point = None
                previous_move_point = None
                current_mode = "Tela limpa por mindinho"
            elif index_up and not middle_up and not ring_up and not pinky_up:
                current_mode = "Desenhando"
                previous_move_point = None
                selected_drawing_index = None

                if previous_point is None:
                    drawings.append(create_empty_canvas(frame.shape))
                    active_drawing_index = len(drawings) - 1
                    last_drawn_index = active_drawing_index
                    previous_point = current_point

                draw_laser_line(
                    drawings[active_drawing_index],
                    previous_point,
                    current_point,
                    drawing_colors(active_drawing_index),
                )
                previous_point = current_point
            else:
                active_drawing_index = None
                selected_drawing_index = None
                previous_point = None
                previous_move_point = None
        else:
            active_drawing_index = None
            selected_drawing_index = None
            previous_point = None
            previous_move_point = None
        # Mantem o fundo preto e sobrepoe o desenho do usuario.
        combined_canvas = compose_drawings(drawings, frame.shape)
        display = cv2.add(display, render_laser_canvas(combined_canvas))
        
        cv2.putText(
            display,
            current_mode,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        cv2.putText(
            display,
            "Indicador: desenha | Pinca: move desenho | Mindinho: limpa | Mao aberta: move mao",
            (10, frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            drawings.clear()
            active_drawing_index = None
            selected_drawing_index = None
            last_drawn_index = None
            previous_point = None
            previous_move_point = None
        elif key == ord("q"):
            break
    hands.close()
    cap.release()
    cv2.destroyAllWindows()


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

if __name__ == "__main__":
    main()
