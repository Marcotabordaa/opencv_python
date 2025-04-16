import cv2
from keras.models import load_model  # TensorFlow es requerido para Keras
from PIL import Image, ImageOps
import numpy as np

# Configurar Keras
np.set_printoptions(suppress=True)
model = load_model("keras_model5.h5", compile=False)
class_names = open("labels5.txt", "r").readlines()

# Inicializar la cámara
cap = cv2.VideoCapture(0)  # Usa 0 para la cámara principal

while True:
    ret, frame = cap.read()  # Capturar frame
    if not ret:
        print("Error al acceder a la cámara")
        break
    
    # Convertir la imagen de OpenCV (BGR) a PIL (RGB)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Redimensionar y normalizar
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Preparar los datos para el modelo
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Hacer la predicción
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Mostrar la predicción en pantalla
    cv2.putText(frame, f"{class_name}: {confidence_score:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar el video en vivo
    cv2.imshow("Reconocimiento en Vivo", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()