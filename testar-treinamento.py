import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import os
import math

# Configurar TensorFlow para usar apenas a CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Configuração MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicialização de vídeo
cap = cv2.VideoCapture(0)
cv2.namedWindow('Detecção de Gestos e Emoções', cv2.WND_PROP_FULLSCREEN)  # Nome da janela e modo fullscreen
cv2.setWindowProperty('Detecção de Gestos e Emoções', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Parâmetros de zoom
zoom_factor = 1.0
zoom_step = 0.05  # Quanto o zoom aumenta ou diminui com o gesto de pinça
pinch_threshold = 50  # Distância mínima entre polegar e indicador para considerar uma pinça
zoom_center = None  # Centro do zoom inicializado como None

# Função para configurar o modelo CNN para expressões
def criar_modelo_cnn():
    modelo = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')  # Feliz, Triste, Normal
    ])
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo

# Tenta carregar o modelo salvo, caso exista
if os.path.exists("modelo_emocoes.h5"):
    modelo = load_model("modelo_emocoes.h5")
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Recriar o otimizador
    print("Modelo carregado com sucesso!")
else:
    modelo = criar_modelo_cnn()

dados_imagens, dados_labels = [], []
modelo_pronto = False  # Indica se o modelo inicial foi treinado

# Mapear emoções para números
emocao_map = {'f': 0, 't': 1, 'n': 2}
emocao_texto = {0: "Feliz", 1: "Triste", 2: "Neutro"}

def calcular_centroide(landmarks, largura, altura):
    x_coords = [int(lm.x * largura) for lm in landmarks]
    y_coords = [int(lm.y * altura) for lm in landmarks]
    centro_x = sum(x_coords) // len(x_coords)
    centro_y = sum(y_coords) // len(y_coords)
    return (centro_x, centro_y)

def detectar_numero_mao(landmarks):
    dedos_levantados = 0
    pontos_dedos = [8, 12, 16, 20]
    if landmarks[4].x > landmarks[3].x:
        dedos_levantados += 1
    for ponto in pontos_dedos:
        if landmarks[ponto].y < landmarks[ponto - 2].y:
            dedos_levantados += 1
    return dedos_levantados

def predizer_emocao(frame):
    rosto_img = cv2.resize(frame, (64, 64))
    rosto_img = np.expand_dims(rosto_img, axis=0) / 255.0
    predicao = modelo.predict(rosto_img)
    emocao_index = np.argmax(predicao)
    porcentagem = predicao[0][emocao_index] * 100
    return emocao_texto[emocao_index], porcentagem

def calcular_distancia(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# Laço principal para captura e treino em tempo real
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
     mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            break

        # Processamento da imagem
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados_maos = hands.process(frame_rgb)
        resultados_face = face_mesh.process(frame_rgb)
        resultados_corpo = pose.process(frame_rgb)

        # Prever e exibir emoção com precisão no topo da tela
        if resultados_face.multi_face_landmarks:
            emocao, porcentagem = predizer_emocao(frame)
            cv2.putText(frame, f"Emocao: {emocao} - Precisao: {porcentagem:.1f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Detectar e desenhar esqueleto do corpo
        if resultados_corpo.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                resultados_corpo.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Detectar e desenhar esqueleto das mãos
        if resultados_maos.multi_hand_landmarks:
            for mao_landmarks in resultados_maos.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    mao_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )

                # Detectar gesto de pinça para zoom in e zoom out
                polegar = mao_landmarks.landmark[4]
                indicador = mao_landmarks.landmark[8]
                distancia_pinca = calcular_distancia(polegar, indicador) * frame.shape[1]  # Convertendo para pixels

                if distancia_pinca < pinch_threshold:
                    # Ajustar o centro do zoom para a posição do gesto de pinça
                    zoom_center = (int(indicador.x * frame.shape[1]), int(indicador.y * frame.shape[0]))

                    # Zoom in
                    zoom_factor = min(zoom_factor + zoom_step, 3.0)  # Limita o zoom máximo a 3x
                elif distancia_pinca > pinch_threshold + 20:
                    # Zoom out
                    zoom_factor = max(zoom_factor - zoom_step, 1.0)  # Limita o zoom mínimo a 1x

        # Aplicar zoom e mover o centro da câmera
        if zoom_factor > 1 and zoom_center:
            h, w, _ = frame.shape
            new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
            center_x, center_y = zoom_center

            # Ajuste o recorte para garantir que ele não ultrapasse os limites da imagem
            start_x = max(0, min(center_x - new_w // 2, w - new_w))
            start_y = max(0, min(center_y - new_h // 2, h - new_h))
            frame_zoomed = frame[start_y:start_y+new_h, start_x:start_x+new_w]
            frame = cv2.resize(frame_zoomed, (w, h))

        # Exibir imagem com zoom e previsões
        cv2.imshow('Detecção de Gestos e Emoções', frame)

        # Controle de teclas para capturar emoção e treinar o modelo
        tecla = cv2.waitKey(10) & 0xFF
        if tecla == ord('g') and dados_imagens:
            # Treinar o modelo com as imagens capturadas e rótulos
            X = np.array(dados_imagens) / 255.0
            y = to_categorical(dados_labels, num_classes=3)
            modelo.fit(X, y, epochs=5)
            print("Modelo treinado com novas emoções!")
            modelo_pronto = True  # Atualizar o estado do modelo para treinado
            
            # Salvar o modelo treinado em um arquivo .h5
            modelo.save("modelo_emocoes.h5")
            print("Modelo salvo como 'modelo_emocoes.h5'")
            
        elif tecla in [ord('f'), ord('t'), ord('n')]:
            # Captura a emoção correspondente e armazena a imagem com o rótulo
            label = emocao_map[chr(tecla)]
            imagem_rostro = cv2.resize(frame, (64, 64))
            dados_imagens.append(imagem_rostro)
            dados_labels.append(label)
            print(f"Imagem capturada para '{emocao_texto[label]}'")
        elif tecla == ord('q'):
            break

# Liberação de recursos e fechamento de janela
cap.release()
cv2.destroyAllWindows()

