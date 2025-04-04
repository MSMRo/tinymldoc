
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="TinyML con Arduino Nano 33 BLE Sense",
    layout="wide"
)

st.markdown("""
    <style>
        .main {
            font-family: 'Segoe UI', sans-serif;
        }
        .stCodeBlock {
            background-color: #f0f0f0 !important;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📘 Documentación TinyML con Arduino Nano 33 BLE Sense")

section = st.sidebar.radio("📂 Navegación", [
    "Introducción",
    "IMU",
    "Micrófono",
    "Sensor de Gestos (APDS9960)",
    "Sensor de Temperatura y Humedad (HTS221)",
    "Sensor de Presión Barométrica (LPS22HB)",
    "Diagrama UML",
    "Diagrama UML2",
    "Sensor Genérico",
    "Cámara (Kit TinyML)",
    "Clasificación con YOLO (Cámara TinyML)",
    "Análisis Temporal con LSTM (IMU)"
])

st.header(section)

if section == "Introducción":
    st.markdown("""TensorFlow Lite Micro para Arduino""")
    st.markdown("""La biblioteca TensorFlow Lite Micro Library for Arduino permite ejecutar modelos de aprendizaje automático en microcontroladores compatibles con Arduino. Proporciona ejemplos y código necesario para integrar modelos de TensorFlow Lite en proyectos de Arduino.

Las Librerias de los sensores se encuentra en: [https://docs.arduino.cc/hardware/nano-33-ble-sense/#suggested-libraries](https://docs.arduino.cc/hardware/nano-33-ble-sense/#suggested-libraries)


### Tabla de Contenidos
- Estructura de la Biblioteca
- Clases y Métodos Principales
- Ejemplos Incluidos
- Instalación y Configuración
- Compatibilidad
- Licencia
- Contribuciones
### Estructura de la Biblioteca
- `examples/`: Contiene proyectos de ejemplo que demuestran el uso de TensorFlow Lite en Arduino.
- `src/`: Incluye el código fuente principal de la biblioteca.
- `docs/`: Documentación adicional sobre la biblioteca y su uso.
- `scripts/`: Scripts útiles para tareas como sincronización y configuración.
- `library.properties`: Archivo que define las propiedades de la biblioteca para el Arduino IDE.
- `README.md`: Archivo principal de documentación con información general y guías de uso.
### Clases y Métodos Principales
**`tflite::MicroInterpreter`**: Esta clase es fundamental para la ejecución de modelos en dispositivos con recursos limitados.
- `Invoke()`: Ejecuta el modelo con los datos de entrada proporcionados.
- `input(index)`: Accede al tensor de entrada en la posición especificada.
- `output(index)`: Accede al tensor de salida en la posición especificada.

**`tflite::MicroMutableOpResolver`**: Gestiona y registra las operaciones (kernels) que el intérprete puede utilizar durante la inferencia.
- `AddBuiltin(builtin_op, registration)`: Registra una operación incorporada.

**`tflite::ErrorReporter`**: Maneja la salida de errores y mensajes de depuración.
- `Report(format, ...)`: Informa de un error o mensaje según el formato especificado.
### Ejemplos Incluidos
- `Hello World`: Conceptos básicos de TensorFlow Lite para microcontroladores.
- `Micro Speech`: Captura audio para detectar las palabras 'sí' y 'no'.
- `Magic Wand`: Clasifica gestos físicos con datos de acelerómetro.
Todos están en el directorio `examples/` con su respectivo `README.md`.
### Instalación y Configuración
Para instalar la versión en desarrollo desde GitHub:
```bash
git clone https://github.com/tensorflow/tflite-micro-arduino-examples Arduino_TensorFlowLite
```
Verifica en el IDE de Arduino en `Archivo -> Ejemplos` que aparezca `Arduino_TensorFlowLite`.
### Compatibilidad
Esta biblioteca está diseñada principalmente para la placa Arduino Nano 33 BLE Sense. También puede usarse en placas con procesadores Arm Cortex M como la Raspberry Pi Pico. Sin embargo, el acceso a sensores está específicamente diseñado para el Nano 33 BLE Sense.""")
    st.markdown("""TensorFlow Lite Micro para Arduino""")
    st.markdown("""La biblioteca TensorFlow Lite Micro Library for Arduino permite ejecutar modelos de aprendizaje automático en microcontroladores compatibles con Arduino. Proporciona ejemplos y código necesario para integrar modelos de TensorFlow Lite en proyectos de Arduino.
### Tabla de Contenidos
- Estructura de la Biblioteca
- Clases y Métodos Principales
- Ejemplos Incluidos
- Instalación y Configuración
- Compatibilidad
- Licencia
- Contribuciones
### Estructura de la Biblioteca
- `examples/`: Contiene proyectos de ejemplo que demuestran el uso de TensorFlow Lite en Arduino.
- `src/`: Incluye el código fuente principal de la biblioteca.
- `docs/`: Documentación adicional sobre la biblioteca y su uso.
- `scripts/`: Scripts útiles para tareas como sincronización y configuración.
- `library.properties`: Archivo que define las propiedades de la biblioteca para el Arduino IDE.
- `README.md`: Archivo principal de documentación con información general y guías de uso.
### Clases y Métodos Principales
**`tflite::MicroInterpreter`**: Esta clase es fundamental para la ejecución de modelos en dispositivos con recursos limitados.
- `Invoke()`: Ejecuta el modelo con los datos de entrada proporcionados.
- `input(index)`: Accede al tensor de entrada en la posición especificada.
- `output(index)`: Accede al tensor de salida en la posición especificada.

**`tflite::MicroMutableOpResolver`**: Gestiona y registra las operaciones (kernels) que el intérprete puede utilizar durante la inferencia.
- `AddBuiltin(builtin_op, registration)`: Registra una operación incorporada.

**`tflite::ErrorReporter`**: Maneja la salida de errores y mensajes de depuración.
- `Report(format, ...)`: Informa de un error o mensaje según el formato especificado.
### Ejemplos Incluidos
- `Hello World`: Conceptos básicos de TensorFlow Lite para microcontroladores.
- `Micro Speech`: Captura audio para detectar las palabras 'sí' y 'no'.
- `Magic Wand`: Clasifica gestos físicos con datos de acelerómetro.
Todos están en el directorio `examples/` con su respectivo `README.md`.
### Instalación y Configuración
Para instalar la versión en desarrollo desde GitHub:
```bash
git clone https://github.com/tensorflow/tflite-micro-arduino-examples Arduino_TensorFlowLite
```
Verifica en el IDE de Arduino en `Archivo -> Ejemplos` que aparezca `Arduino_TensorFlowLite`.
### Compatibilidad
Esta biblioteca está diseñada principalmente para la placa Arduino Nano 33 BLE Sense. También puede usarse en placas con procesadores Arm Cortex M como la Raspberry Pi Pico. Sin embargo, el acceso a sensores está específicamente diseñado para el Nano 33 BLE Sense.""")
    st.markdown("# Código de ejemplo de python")
    
    st.markdown("## Descargar el datset desde GitHub")

    github_file_url = "https://raw.githubusercontent.com/MSMRo/tinymldoc/refs/heads/main/data.csv"

    st.markdown(f"[Haz clic aquí para descargar el dataset 📄]({github_file_url})", unsafe_allow_html=True)

    st.markdown("## Creación del modelo")

    st.code("""
#!pip install "tensorflow[and-cuda]" --upgrade --force-reinstall --no-cache-dir

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("data.csv")
df.head()

X = df[["x1","x2"]].values
y = df[["y"]].values

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

model = models.Sequential([
    layers.Input(shape=(2,)), 
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax'),
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, validation_data=(X_test, y_test))

import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Save the model
model.save("model.h5")

import seaborn as sb
sb.scatterplot(x=X_test[:,0], y=X_test[:,1], hue=y_test.flatten(), palette="deep")

import numpy as np
val = np.array([[700,400]])
print("Clase predicha es:", int(np.argmax(model.predict(val))))

!pip install pydot
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='./model.png')

# Exportar el modelo a formato TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo TFLite en un archivo
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# Convertir a C array para usar en Arduino
#import binascii

#hex_data = binascii.hexlify(tflite_model).decode('utf-8')
#c_array = ','.join(f'0x{hex_data[i:i+2]}' for i in range(0, len(hex_data), 2))

#with open("model.h", "w") as f:
#    f.write('#include <cstdint>\\n\\n')
#    f.write('const unsigned char g_model[] = {\\n')
#    f.write(c_array)
#    f.write('\\n};\\n')
#    f.write(f'const int g_model_len = {len(tflite_model)};\\n')

# Convertir el modelo TFLite a un archivo .h
with open("model.tflite", "rb") as f:
    tflite_data = f.read()

with open("model.h", "w") as f:
    f.write("#ifndef MODEL_H\\n")
    f.write("#define MODEL_H\\n\\n")
    f.write(f"const unsigned char model_tflite[] = {{\\n")
    f.write(", ".join(f"0x{byte:02x}" for byte in tflite_data))
    f.write("\\n};\\n\\n")
    f.write(f"const unsigned int model_tflite_len = {len(tflite_data)};\\n\\n")
    f.write("#endif // MODEL_H\\n")

""", language='python')
    
    st.markdown("# Ejemplo de código en arduino")

    st.markdown("## Descargar el modelo desde GitHub")

    github_file_url = "https://raw.githubusercontent.com/MSMRo/tinymldoc/refs/heads/main/model.h"

    st.markdown(f"[Haz clic aquí para descargar el dataset 📄]({github_file_url})", unsafe_allow_html=True)

    st.markdown("## Inferencia del modelo en arduino")

    st.code("""
#include <TensorFlowLite.h>
#include "model.h"  // tu modelo en formato .h convertido con xxd

#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

// Configuración de memoria
constexpr int kTensorArenaSize = 21 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Instancias globales
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(100);

  // Cargar modelo
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Modelo incompatible con TFLite Micro");
    return;
  }

  // Resolver de operaciones
  static tflite::AllOpsResolver resolver;

  // Crear intérprete
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Asignar tensores
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Fallo al asignar tensores");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Modelo cargado correctamente 🎉");
}

void loop() {
  // ✅ Ejemplo: llenar input con dos valores
  input->data.f[0] = 200;
  input->data.f[1] = 305;

  // Ejecutar inferencia
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Fallo al ejecutar inferencia");
    return;
  }

  // Mostrar salida
  Serial.print("Resultados: ");
  for (int i = 0; i < output->dims->data[1]; ++i) {
    Serial.print(output->data.f[i], 5);
    Serial.print(" ");
  }
  Serial.println();

  float* resultados = output->data.f;
  int num_clases = output->dims->data[1];
  //int num_clases = output->dims->data[1];
  //Serial.print("Número de clases detectadas por el modelo: ");
  //Serial.println(num_clases);

  int pred = 0;
  float confianza = resultados[0];

  for (int i = 1; i < num_clases; i++) {
    if (resultados[i] > confianza) {
      confianza = resultados[i];
      pred = i;
    }
  }

  Serial.print("Movimiento detectado: Clase ");
  Serial.print(pred);
  Serial.print(" | Confianza: ");
  Serial.println(confianza);

  delay(2000);
}

    """, language='c')
    st.markdown("# Explicación del código de arduino")

    st.title("Descargar el modelo desde GitHub")

    github_file_url = "https://raw.githubusercontent.com/MSMRo/tinymldoc/refs/heads/main/model.h"

    st.markdown(f"[Haz clic aquí para descargar el archivo 📄]({github_file_url})", unsafe_allow_html=True)

    st.markdown("## Importación de las librerias de arduino")
    st.code(""" 
// Inclusión de la cabecera principal de TensorFlow Lite para microcontroladores
#include <TensorFlowLite.h>

// Se incluye el modelo convertido a un arreglo C (formato .h generado con xxd o script en Python)
#include "model.h"  // Este archivo contiene: const unsigned char model_tflite[] = {...};

// Incluye clases para reportar errores de TFLite
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"

// Intérprete para modelos TFLite en dispositivos con recursos limitados
#include "tensorflow/lite/micro/micro_interpreter.h"

// Para interpretar el modelo cargado (estructura interna del archivo .tflite)
#include "tensorflow/lite/schema/schema_generated.h"

// Resolver de operaciones: incluye todas las operaciones posibles de TFLite (puede ser pesado)
#include "tensorflow/lite/micro/all_ops_resolver.h"

""", language='c')
    st.markdown("## Configuración de memoria para tensores")
    st.code(""" 
// Tamaño del buffer de memoria donde se almacenarán los tensores del modelo
constexpr int kTensorArenaSize = 21 * 1024;  // 21 KB
uint8_t tensor_arena[kTensorArenaSize];      // Memoria estática para los tensores

""", language='c')
    
    st.markdown("## Declaración de variables globales")
    st.code(""" 
tflite::MicroErrorReporter micro_error_reporter;     // Manejador de errores
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* model = nullptr;                // Puntero al modelo
tflite::MicroInterpreter* interpreter = nullptr;     // Puntero al intérprete
TfLiteTensor* input = nullptr;                       // Puntero al tensor de entrada
TfLiteTensor* output = nullptr;                      // Puntero al tensor de salida

""", language='c')
    
    st.markdown("## Función setup()")
    st.code(""" 
void setup() {
  Serial.begin(115200);            // Inicia la comunicación serial
  while (!Serial) delay(100);      // Espera hasta que el puerto esté listo

  // Cargar el modelo desde el array en memoria
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Modelo incompatible con TFLite Micro");
    return;
  }

  // Resolver: se inicializa con todas las operaciones disponibles (puede ser reemplazado por MicroMutableOpResolver)
  static tflite::AllOpsResolver resolver;

  // Crear el intérprete del modelo, asignando el modelo, las operaciones, el área de memoria, y el manejador de errores
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Asignar los tensores internos (reserva memoria en el tensor_arena)
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Fallo al asignar tensores");
    return;
  }

  // Obtener punteros directos al tensor de entrada y salida
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Modelo cargado correctamente 🎉");
}

""", language='c')
    
    st.markdown("## Función loop()")
    st.code(""" 
void loop() {
  // Asignar valores de entrada al modelo (en este caso, 2 características de entrada)
  input->data.f[0] = 200;   // Primer valor de entrada
  input->data.f[1] = 305;   // Segundo valor de entrada

  // Ejecutar la inferencia
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Fallo al ejecutar inferencia");
    return;
  }

  // Mostrar los resultados de salida
  Serial.print("Resultados: ");
  for (int i = 0; i < output->dims->data[1]; ++i) {
    Serial.print(output->data.f[i], 5);  // Imprime cada valor con 5 decimales
    Serial.print(" ");
  }
  Serial.println();

  // Obtener el índice con mayor probabilidad (argmax)
  float* resultados = output->data.f;
  int num_clases = output->dims->data[1];
  int pred = 0;
  float confianza = resultados[0];

  for (int i = 1; i < num_clases; i++) {
    if (resultados[i] > confianza) {
      confianza = resultados[i];
      pred = i;
    }
  }

  // Mostrar clase predicha y su nivel de confianza
  Serial.print("Movimiento detectado: Clase ");
  Serial.print(pred);
  Serial.print(" | Confianza: ");
  Serial.println(confianza);

  delay(2000);  // Esperar 2 segundos antes de la siguiente inferencia
}

""", language='c')

    
elif section == "IMU":

    st.markdown("""1. Clasificación de Gestos con IMU
Este ejemplo utiliza el sensor IMU del Arduino Nano 33 BLE Sense para detectar gestos como 'arriba', 'abajo', 'izquierda' y 'derecha'.
Consta de tres partes: adquisición de datos desde el sensor, entrenamiento del modelo en Python, y despliegue del modelo en la placa.""")
    st.subheader("Código Arduino para adquisición de datos IMU")
    st.markdown("Pueden usar este código usando millis y controlando el tiempo de muestreo:")
    st.code("""
#include <Arduino_LSM9DS1.h> 
// Incluye la librería para controlar el sensor de movimiento LSM9DS1 (acelerómetro y giroscopio)
// Esta librería es compatible con la placa Arduino Nano 33 BLE Sense

int fs = 10;       // Frecuencia de muestreo en Hz (cuántas veces por segundo se tomará una medición)
float T = 1/fs;    // Periodo de muestreo en segundos (tiempo entre mediciones)

// Variable para guardar el tiempo del último muestreo
unsigned long lastMillis = 0;

void setup() {
  Serial.begin(115200); // Inicializa la comunicación serial a 115200 baudios
  while (!Serial)
    ; // Espera a que se abra el monitor serial (importante para algunas placas)

  // Intenta inicializar el sensor IMU (acelerómetro + giroscopio)
  if (!IMU.begin()) {
    Serial.println("Error al iniciar IMU"); // Si falla, imprime error
    while (1)
      ; // Se detiene en un bucle infinito
  }
}

void loop() {
  float ax, ay, az; // Variables para la aceleración en los ejes X, Y, Z
  float gx, gy, gz; // Variables para la velocidad angular (giroscopio) en los ejes X, Y, Z

  // Verifica si ha pasado el tiempo suficiente desde la última lectura
  if (millis() - lastMillis > T) {
    lastMillis = millis(); // Actualiza el tiempo de la última lectura

    // Verifica que haya datos disponibles de acelerómetro y giroscopio
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      // Lee los datos del acelerómetro y los guarda en ax, ay, az
      IMU.readAcceleration(ax, ay, az);

      // Lee los datos del giroscopio y los guarda en gx, gy, gz
      IMU.readGyroscope(gx, gy, gz);

      // Imprime los datos separados por comas (útil para registrar en CSV o visualizar en tiempo real)
      Serial.print(ax);
      Serial.print(",");
      Serial.print(ay);
      Serial.print(",");
      Serial.print(az);
      Serial.print(",");
      Serial.print(gx);
      Serial.print(",");
      Serial.print(gy);
      Serial.print(",");
      Serial.println(gz); // Cambia de línea después de imprimir gz
    }
  }
}

            """, language='c')
    st.markdown("Tambien pueden usar un código mas simple, pero deben de asegurar el tiempo de muestreo:")
    st.code("""#include <Arduino_LSM9DS1.h>

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  if (!IMU.begin()) {
    Serial.println("Error al iniciar IMU");
    while (1);
  }
}

void loop() {
  float ax, ay, az, gx, gy, gz;
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);
    Serial.print(ax); Serial.print(",");
    Serial.print(ay); Serial.print(",");
    Serial.print(az); Serial.print(",");
    Serial.print(gx); Serial.print(",");
    Serial.print(gy); Serial.print(",");
    Serial.println(gz);
  }
  delay(100);
}""", language='c')

    st.subheader('Adquirir datos desde la PC')
    st.code("""
pip install pyserial tqdm numpy pandas tensorflow
""", language='bash')
    st.markdown("Se debe crear un archivo llamado utils.txt en el mismo folder del script de adquisición")
    st.code("""
{'count': '0', 'file_name': 'ex1'}
""", language='bash')

    st.code("""
import serial
import time
import sys
import os
import numpy as np
from tqdm import tqdm

# Configuración del puerto serie
PATH_ROOT = "./dataset/mov3/"
PUERTO_COM = "COM4"            # Cambiar si es necesario (por ejemplo "COM3", "COM6" o "/dev/ttyUSB0" en Linux)
BAUD_RATE = 115200             # Debe coincidir con Serial.begin del Arduino
FILE_UTILS = "utils.txt"       # Archivo auxiliar que guarda el nombre base y el contador

# Leer archivo auxiliar para obtener nombre base y contador de archivos
with open(FILE_UTILS, 'r') as f:
    lines = eval(f.read())          # Convierte el contenido en un diccionario
    count = lines["count"]          # Número de archivo actual
    file_name = lines["file_name"]  # Nombre base del archivo

# Construir el nombre del archivo de salida
FILE_NAME = file_name + "." + count + ".txt"

# Parámetros de adquisición
t = 1        # Duración total de adquisición (en segundos)
Fs = 10      # Frecuencia de muestreo (Hz)
T = 1 / Fs   # Periodo de muestreo
n = int(t / T)  # Número total de muestras

# Abrir conexión serial con Arduino
ser = serial.Serial(PUERTO_COM, BAUD_RATE)

# Captura de datos
try:
    with open(PATH_ROOT+FILE_NAME, "w") as f:
        for i in tqdm(range(n)):
            line = ser.readline().decode('utf-8', errors='ignore')  # Leer línea del puerto serial
            f.write(line.strip() + "\\n")  # Guardar en archivo sin espacios en blanco

            time.sleep(T)  # Esperar entre lecturas para respetar la frecuencia

        print("Datos guardados en", FILE_NAME)
        ser.close()  # Cerrar puerto serial
except:
    ser.close()  # Cerrar puerto serial en caso de error

# Actualizar contador en el archivo auxiliar
with open(FILE_UTILS, 'w') as f:
    lines["count"] = str(int(count) + 1)  # Incrementar contador
    f.write(str(lines))                   # Guardar nueva versión
    print("Archivo actualizado:", lines["count"])

""", language='python')

    st.markdown("""
## Dataset de Clasificación de Movimiento

Este dataset contiene datos del sensor IMU (acelerómetro y giroscopio) recolectados desde un Arduino Nano 33 BLE Sense.

## 📁 Estructura del Dataset
El dataset debe tener una estructura, cada clase de movimiento se almacena en una carpeta separada.
EL archivo debe llamarse por ejem: mov1.0.txt donde contendrá los valores de los acelerometros y giroscopios.
                
""")
    imagen = Image.open("img/dataset_folder_tinyml3.png")
    st.image(imagen, caption='Estructura del Dataset', use_container_width=False)

    img2 = Image.open("img/estructura_contenido.png")
    st.image(img2, caption='Contenido de los files: mov1.x.png', use_container_width=False)


    st.subheader("Código Python para entrenamiento del modelo IMU")

    st.code("""
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

DATASET_PATH = "./dataset"
SEQUENCE_LENGTH = 30  # número fijo de muestras por secuencia
NUM_FEATURES = 6       # ax, ay, az, gx, gy, gz

def cargar_y_preprocesar(dataset_path=DATASET_PATH, seq_len=SEQUENCE_LENGTH):
    X_raw, y_raw = [], []

    for clase in os.listdir(dataset_path):
        clase_path = os.path.join(dataset_path, clase)
        if not os.path.isdir(clase_path): continue
        for archivo in os.listdir(clase_path):
            if archivo.endswith(".txt"):
                with open(os.path.join(clase_path, archivo)) as f:
                    fila = []
                    for linea in f:
                        if "," in linea:
                            datos = [float(x) for x in linea.strip().split(",")]
                            if len(datos) == NUM_FEATURES:
                                fila.append(datos)
                    if len(fila) == 0:
                        continue
                    # Padding o corte para longitud fija
                    if len(fila) > seq_len:
                        fila = fila[:seq_len]
                    elif len(fila) < seq_len:
                        pad = [[0.0] * NUM_FEATURES] * (seq_len - len(fila))
                        fila += pad
                    X_raw.append(fila)
                    y_raw.append(clase)

    X = np.array(X_raw)
    y = np.array(y_raw)

    # Normalización (por característica)
    X_reshape = X.reshape(-1, NUM_FEATURES)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_reshape)
    X_norm = X_norm.reshape(-1, seq_len, NUM_FEATURES)

    # Codificar clases
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_cat = tf.keras.utils.to_categorical(y_encoded)

    return X_norm, y_cat, encoder.classes_

# PREPROCESAR
X, y, clases = cargar_y_preprocesar()
print("Formas:", X.shape, y.shape, "Clases:", clases)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout

# model = Sequential([
#     SimpleRNN(32, input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
#     Dropout(0.3),
#     Dense(32, activation='relu'),
#     Dense(len(clases), activation='softmax')
# ])
model = Sequential([
    Conv1D(16, kernel_size=3, activation='relu', input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
    Conv1D(32, kernel_size=3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(len(clases), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=8)


loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Pérdida en el conjunto de prueba: {loss:.4f}")
print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")

model.save("modelo_movimientos_lstm3.keras")

# Convertir el modelo a formato TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo convertido en un archivo
with open("modelo_conv1d.tflite", "wb") as f:
    f.write(tflite_model)

def convertir_tflite_a_header(nombre_tflite, nombre_header):
    with open(nombre_tflite, "rb") as f:
        contenido = f.read()

    # Abrimos el archivo header para escribir el arreglo
    with open(nombre_header, "w") as f:
        array_name = nombre_tflite.split('.')[0] + "_tflite"
        f.write(f"const unsigned char {array_name}[] = {{\\n")

        for i, byte in enumerate(contenido):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{byte:02x}")
            if i < len(contenido) - 1:
                f.write(", ")
            if (i + 1) % 12 == 0:
                f.write("\\n")

        f.write("\\n};\\n")
        f.write(f"const unsigned int {array_name}_len = {len(contenido)};\\n")

    print(f"✅ Archivo '{nombre_header}' generado con éxito.")


# 👉 Reemplaza con el nombre de tu modelo
convertir_tflite_a_header("modelo_conv1d.tflite", "modelo_conv1d.h")""", language='python')

    st.markdown('**🔄 Conversión del modelo TFLite a .h**')
    st.code("""
def convertir_tflite_a_header(nombre_tflite, nombre_header):
    with open(nombre_tflite, "rb") as f:
        contenido = f.read()

    # Abrimos el archivo header para escribir el arreglo
    with open(nombre_header, "w") as f:
        array_name = nombre_tflite.split('.')[0] + "_tflite"
        f.write(f"const unsigned char {array_name}[] = {{\\n")

        for i, byte in enumerate(contenido):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{byte:02x}")
            if i < len(contenido) - 1:
                f.write(", ")
            if (i + 1) % 12 == 0:
                f.write("\\n")

        f.write("\\n};\\n")
        f.write(f"const unsigned int {array_name}_len = {len(contenido)};\\n")

    print(f"✅ Archivo '{nombre_header}' generado con éxito.")


# 👉 Reemplaza con el nombre de tu modelo
convertir_tflite_a_header("modelo_conv1d.tflite", "modelo_conv1d.h")
""", language='bash')

    st.subheader("Código Arduino para inferencia del modelo IMU")
    st.markdown('Debes incluir la libreria modelo_conv1d.h')
    st.code("""
#include "modelo_conv1d.h"  // Tu modelo convertido
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <Arduino_LSM9DS1.h>

#define SEQUENCE_LENGTH 50
#define NUM_FEATURES 6
#define TENSOR_ARENA_SIZE 10 * 1024  // Ajusta si hace falta

uint8_t tensor_arena[TENSOR_ARENA_SIZE];

tflite::MicroErrorReporter error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = tflite::GetModel(modelo_conv1d_tflite);

// ✅ Esta es la forma correcta para tu versión:
//tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE, &error_reporter);
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);


TfLiteTensor* input;
TfLiteTensor* output;

void setup() {
  Serial.begin(115200);
  while (!Serial)
    ;

  if (!IMU.begin()) {
    Serial.println("Error al inicializar el IMU");
    while (1)
      ;
  }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Modelo incompatible.");
    while (1)
      ;
  }

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Error asignando tensores");
    while (1)
      ;
  }

  input = interpreter.input(0);
  output = interpreter.output(0);

  Serial.println("Todo listo. Iniciando inferencia...");
}

void loop() {
  // Recolectar SEQUENCE_LENGTH muestras antes de inferencia
  // for (int i = 0; i < SEQUENCE_LENGTH; i++) {
  //   float ax, ay, az, gx, gy, gz;
  //   while (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable());

  //   IMU.readAcceleration(ax, ay, az);
  //   IMU.readGyroscope(gx, gy, gz);

  //   // Asignar al tensor de entrada
  //   input->data.f[i * NUM_FEATURES + 0] = ax;
  //   input->data.f[i * NUM_FEATURES + 1] = ay;
  //   input->data.f[i * NUM_FEATURES + 2] = az;
  //   input->data.f[i * NUM_FEATURES + 3] = gx;
  //   input->data.f[i * NUM_FEATURES + 4] = gy;
  //   input->data.f[i * NUM_FEATURES + 5] = gz;

  //   delay(10);  // ~10 Hz
  // }
  int muestras = 0;

  Serial.println("Recolectando datos del sensor...");
  while (muestras < SEQUENCE_LENGTH) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      float ax, ay, az, gx, gy, gz;
      if (IMU.readAcceleration(ax, ay, az) && IMU.readGyroscope(gx, gy, gz)) {
        input->data.f[muestras * NUM_FEATURES + 0] = ax;
        input->data.f[muestras * NUM_FEATURES + 1] = ay;
        input->data.f[muestras * NUM_FEATURES + 2] = az;
        input->data.f[muestras * NUM_FEATURES + 3] = gx;
        input->data.f[muestras * NUM_FEATURES + 4] = gy;
        input->data.f[muestras * NUM_FEATURES + 5] = gz;
        muestras++;
      } else {
        Serial.println("Lectura fallida del IMU. Reintentando...");
      }
    }
    delay(100);  // Frecuencia ~100 Hz
  }

  if (interpreter.Invoke() != kTfLiteOk) {
    Serial.println("Error durante la inferencia");
    return;
  }

  // float* resultados = output->data.f;
  // int num_clases = output->dims->data[1];
  // int max_index = 0;
  // float max_score = resultados[0];

  // for (int i = 1; i < num_clases; i++) {
  //   if (resultados[i] > max_score) {
  //     max_score = resultados[i];
  //     max_index = i;
  //   }
  // }

  // Serial.print("Predicción: Clase ");
  // Serial.print(max_index);
  // Serial.print(" - Confianza: ");
  // Serial.println(max_score);

  float* resultados = output->data.f;
  int num_clases = output->dims->data[1];
  //int num_clases = output->dims->data[1];
  Serial.print("Número de clases detectadas por el modelo: ");
  Serial.println(num_clases);

  int pred = 0;
  float confianza = resultados[0];

  for (int i = 1; i < num_clases; i++) {
    if (resultados[i] > confianza) {
      confianza = resultados[i];
      pred = i;
    }
  }

  Serial.print("Movimiento detectado: Clase ");
  Serial.print(pred);
  Serial.print(" | Confianza: ");
  Serial.println(confianza);
  delay(500);
}""", language='c')

elif section == "Micrófono":
    st.markdown("""2. Reconocimiento de Palabras Clave con Micrófono
Este ejemplo utiliza el micrófono digital del Arduino Nano 33 BLE Sense para detectar palabras clave como 'sí' y 'no'.
Incluye captura de audio, entrenamiento con CNN y despliegue del modelo en la placa.""")
    st.subheader("Código Arduino para adquisición de datos del micrófono")
    st.code("""#include <PDM.h>

void setup() {
  Serial.begin(115200);
  while (!Serial);
  if (!PDM.begin(1, 16000)) {
    Serial.println("Error al iniciar PDM");
    while (1);
  }
  PDM.onReceive(onPDMdata);
}

short buffer[512];
volatile int samplesRead = 0;

void loop() {}

void onPDMdata() {
  samplesRead = PDM.read(buffer, sizeof(buffer));
  for (int i = 0; i < samplesRead; i++) {
    Serial.println(buffer[i]);
  }
}""", language='c')
    st.subheader("Código Python para entrenamiento del modelo de palabras clave")
    st.code("""import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Supongamos que X_train son espectrogramas de audio preprocesados
# X_train.shape = (n_samples, height, width, 1)

model = Sequential([
    Conv2D(8, (3,3), activation='relu', input_shape=(32,32,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("modelo_audio.tflite", "wb") as f:
    f.write(tflite_model)""", language='c')
    st.markdown('**🔄 Conversión del modelo TFLite a .h**')
    st.code("""xxd -i modelo_nombre.tflite > modelo_nombre.h""", language='bash')
    st.subheader("Código Arduino para inferencia de palabras clave")
    st.code("""#include <PDM.h>
#include "modelo_audio.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int kTensorArenaSize = 8 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroErrorReporter error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = tflite::GetModel(modelo_audio_tflite);
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
TfLiteTensor* input;
TfLiteTensor* output;

void setup() {
  Serial.begin(115200);
  while (!Serial);
  PDM.begin(1, 16000);
  PDM.onReceive(onPDMdata);
  interpreter.AllocateTensors();
  input = interpreter.input(0);
  output = interpreter.output(0);
}

short buffer[512];
volatile bool dataReady = false;

void loop() {
  if (dataReady) {
    dataReady = false;
    // Preprocesamiento y copia de datos en input->data.f...
    interpreter.Invoke();
    float* prediction = output->data.f;
    int resultado = std::distance(prediction, std::max_element(prediction, prediction + 2));
    if (resultado == 0) Serial.println("Detectado: Sí");
    else if (resultado == 1) Serial.println("Detectado: No");
  }
}

void onPDMdata() {
  PDM.read(buffer, sizeof(buffer));
  dataReady = true;
}""", language='c')

elif section == "Sensor de Gestos (APDS9960)":
    st.subheader("🎯 Objetivo")
    st.markdown("Entrenar un modelo que reconozca gestos de mano como: derecha, izquierda, arriba y abajo, usando el sensor APDS9960, y realizar inferencias directamente en el Arduino Nano 33 BLE Sense.")

    st.subheader("🟢 a. Adquisición de Datos (Arduino)")
    st.markdown("Usamos la biblioteca oficial `Arduino_APDS9960` para capturar los gestos detectados por el sensor.")
    st.markdown("📦 **Librería necesaria**: Arduino_APDS9960")

    st.code("""#include <Arduino_APDS9960.h>

void setup() {
  Serial.begin(115200);
  while (!Serial);
  if (!APDS.begin()) {
    Serial.println("¡Error al inicializar el sensor APDS9960!");
    while (1);
  }
  Serial.println("Sensor APDS9960 inicializado.");
}

void loop() {
  if (APDS.gestureAvailable()) {
    int gesture = APDS.readGesture();
    switch (gesture) {
      case GESTURE_UP: Serial.println("ARRIBA"); break;
      case GESTURE_DOWN: Serial.println("ABAJO"); break;
      case GESTURE_LEFT: Serial.println("IZQUIERDA"); break;
      case GESTURE_RIGHT: Serial.println("DERECHA"); break;
      default: Serial.println("GESTO DESCONOCIDO"); break;
    }
  }
}
""", language='c')

    st.markdown("📝 **¿Qué hacer con esto?**\n- Abre el Monitor Serial del IDE de Arduino.\n- Mueve la mano frente al sensor.\n- Copia los datos y guárdalos en un archivo `.csv` como este:")

    st.code("""proximidad,gesture_label
123,ARRIBA
129,ABAJO
110,IZQUIERDA
118,DERECHA
...""", language='text')

    st.subheader("🧠 b. Creación del modelo en Python")
    st.code("""import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("gestos_apds.csv")
X = df[["proximidad"]].values
y = df["gesture_label"].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = tf.keras.utils.to_categorical(y_encoded)

model = Sequential([
    Dense(8, input_shape=(1,), activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_cat, epochs=30)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("modelo_gestos_apds.tflite", "wb") as f:
    f.write(tflite_model)
""", language='python')

    st.markdown("✅ **Conversión a archivo `.h`**")
    st.code("xxd -i modelo_gestos_apds.tflite > modelo_gestos_apds.h", language='bash')

    st.subheader("🔁 c. Inferencia en Arduino")
    st.code("""#include <Arduino_APDS9960.h>
#include "modelo_gestos_apds.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

constexpr int kTensorArenaSize = 4 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = tflite::GetModel(modelo_gestos_apds_tflite);
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
TfLiteTensor* input;
TfLiteTensor* output;

void setup() {
  Serial.begin(115200);
  while (!Serial);
  if (!APDS.begin()) {
    Serial.println("Error al iniciar el sensor APDS9960");
    while (1);
  }
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Error al asignar tensores");
    while (1);
  }
  input = interpreter.input(0);
  output = interpreter.output(0);
}

void loop() {
  if (APDS.gestureAvailable()) {
    int gesture = APDS.readGesture();
    int proximity = APDS.readProximity();
    input->data.f[0] = proximity;
    if (interpreter.Invoke() == kTfLiteOk) {
      float* pred = output->data.f;
      int pred_label = std::distance(pred, std::max_element(pred, pred + 4));
      switch (pred_label) {
        case 0: Serial.println("ARRIBA (modelo)"); break;
        case 1: Serial.println("ABAJO (modelo)"); break;
        case 2: Serial.println("IZQUIERDA (modelo)"); break;
        case 3: Serial.println("DERECHA (modelo)"); break;
      }
    } else {
      Serial.println("Error en la inferencia");
    }
  }
  delay(200);
}
""", language='c')

elif section == "Sensor de Temperatura y Humedad (HTS221)":
    st.subheader("🎯 Objetivo")
    st.markdown("Entrenar un modelo que clasifique condiciones ambientales (por ejemplo: 'calor seco', 'ambiente templado', 'ambiente húmedo') usando los datos del sensor HTS221.")

    st.subheader("🟢 a. Adquisición de Datos (Arduino)")
    st.markdown("📦 **Librería necesaria**: Arduino_HTS221")

    st.code("""#include <Arduino_HTS221.h>

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!HTS.begin()) {
    Serial.println("No se pudo encontrar el sensor HTS221!");
    while (1);
  }
}

void loop() {
  float temperatura = HTS.readTemperature(); // en °C
  float humedad = HTS.readHumidity();       // en %

  Serial.print(temperatura);
  Serial.print(",");
  Serial.println(humedad);

  delay(2000);
}
""", language='c')

    st.markdown("📋 **Formato de salida esperado:**")
    st.code("""temperatura,humedad
29.5,20.1
25.1,40.2
18.2,65.3
...""", language='text')

    st.markdown("🧠 Etiquetar los datos según reglas como:\n- `calor seco`: temp > 28 & humedad < 30\n- `templado`: temp 20–28 & humedad 30–60\n- `húmedo`: humedad > 60")

    st.subheader("🧠 b. Creación del modelo en Python")
    st.code("""import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("ambiente_humedad.csv")

X = df[["temperatura", "humedad"]].values
y = df["clase"].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = tf.keras.utils.to_categorical(y_encoded)

model = Sequential([
    Dense(8, input_shape=(2,), activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_cat, epochs=30)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("modelo_ambiente.tflite", "wb") as f:
    f.write(tflite_model)
""", language='python')

    st.markdown("🔁 **Conversión a .h**")
    st.code("xxd -i modelo_ambiente.tflite > modelo_ambiente.h", language='bash')

    st.subheader("🔁 c. Inferencia en Arduino")
    st.code("""#include <Arduino_HTS221.h>
#include "modelo_ambiente.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

constexpr int kTensorArenaSize = 4 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = tflite::GetModel(modelo_ambiente_tflite);
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);

TfLiteTensor* input;
TfLiteTensor* output;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!HTS.begin()) {
    Serial.println("Sensor HTS221 no disponible");
    while (1);
  }

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Error al asignar tensores");
    while (1);
  }

  input = interpreter.input(0);
  output = interpreter.output(0);
}

void loop() {
  float temperatura = HTS.readTemperature();
  float humedad = HTS.readHumidity();

  input->data.f[0] = temperatura;
  input->data.f[1] = humedad;

  if (interpreter.Invoke() == kTfLiteOk) {
    float* pred = output->data.f;
    int clase = std::distance(pred, std::max_element(pred, pred + 3));

    switch (clase) {
      case 0: Serial.println("Ambiente seco y caliente"); break;
      case 1: Serial.println("Ambiente templado"); break;
      case 2: Serial.println("Ambiente húmedo"); break;
    }
  } else {
    Serial.println("Error en la inferencia");
  }

  delay(2000);
}
""", language='c')

elif section == "Sensor de Presión Barométrica (LPS22HB)":
    st.markdown("""5. Clasificación de Altitud con Sensor de Presión (LPS22HB)
Este ejemplo utiliza el sensor LPS22HB para clasificar altitud estimada en base a la presión: nivel del mar, media y montaña.
Incluye lectura de datos con Arduino, entrenamiento de modelo de clasificación y despliegue para inferencia.""")
    st.subheader("Código Arduino para adquisición de presión")
    st.code("""#include <Arduino_LPS22HB.h>

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!BARO.begin()) {
    Serial.println("Sensor de presión no disponible");
    while (1);
  }
}

void loop() {
  float presion = BARO.readPressure(); // en hPa
  Serial.println(presion);
  delay(1000);
}""", language='c')
    st.subheader("Código Python para entrenamiento de modelo de altitud")
    st.code("""import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("presion.csv")
X = df[["presion"]].values
y = df["altitud"].values  # clases: "mar", "media", "montaña"

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = tf.keras.utils.to_categorical(y_encoded)

model = Sequential([
    Dense(8, input_shape=(1,), activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_cat, epochs=30)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("modelo_presion.tflite", "wb") as f:
    f.write(tflite_model)""", language='c')
    st.markdown('**🔄 Conversión del modelo TFLite a .h**')
    st.code("""xxd -i modelo_nombre.tflite > modelo_nombre.h""", language='bash')
    st.subheader("Código Arduino para inferencia de altitud")
    st.code("""#include <Arduino_LPS22HB.h>
#include "modelo_presion.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int kTensorArenaSize = 4 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = tflite::GetModel(modelo_presion_tflite);
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
TfLiteTensor* input;
TfLiteTensor* output;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!BARO.begin()) {
    Serial.println("Sensor de presión no disponible");
    while (1);
  }

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Error al asignar tensores");
    while (1);
  }

  input = interpreter.input(0);
  output = interpreter.output(0);
}

void loop() {
  float presion = BARO.readPressure();
  input->data.f[0] = presion;

  if (interpreter.Invoke() == kTfLiteOk) {
    float* pred = output->data.f;
    int altitud = std::distance(pred, std::max_element(pred, pred + 3));

    switch (altitud) {
      case 0: Serial.println("Altitud: Nivel del mar"); break;
      case 1: Serial.println("Altitud: Media"); break;
      case 2: Serial.println("Altitud: Montaña"); break;
    }
  } else {
    Serial.println("Error en la inferencia");
  }

  delay(1000);
}""", language='c')
    st.markdown("""5. Clasificación de Altitud con Sensor de Presión (LPS22HB)
Este ejemplo utiliza el sensor LPS22HB para clasificar altitud estimada en base a la presión: nivel del mar, media y montaña.
Incluye lectura de datos con Arduino, entrenamiento de modelo de clasificación y despliegue para inferencia.""")
    st.subheader("Código Arduino para adquisición de presión")
    st.code("""#include <Arduino_LPS22HB.h>

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!BARO.begin()) {
    Serial.println("Sensor de presión no disponible");
    while (1);
  }
}

void loop() {
  float presion = BARO.readPressure(); // en hPa
  Serial.println(presion);
  delay(1000);
}""", language='c')
    st.subheader("Código Python para entrenamiento de modelo de altitud")
    st.code("""import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("presion.csv")
X = df[["presion"]].values
y = df["altitud"].values  # clases: "mar", "media", "montaña"

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = tf.keras.utils.to_categorical(y_encoded)

model = Sequential([
    Dense(8, input_shape=(1,), activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_cat, epochs=30)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("modelo_presion.tflite", "wb") as f:
    f.write(tflite_model)""", language='c')
    st.markdown('**🔄 Conversión del modelo TFLite a .h**')
    st.code("""xxd -i modelo_nombre.tflite > modelo_nombre.h""", language='bash')
    st.subheader("Código Arduino para inferencia de altitud")
    st.code("""#include <Arduino_LPS22HB.h>
#include "modelo_presion.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int kTensorArenaSize = 4 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = tflite::GetModel(modelo_presion_tflite);
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
TfLiteTensor* input;
TfLiteTensor* output;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!BARO.begin()) {
    Serial.println("Sensor de presión no disponible");
    while (1);
  }

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Error al asignar tensores");
    while (1);
  }

  input = interpreter.input(0);
  output = interpreter.output(0);
}

void loop() {
  float presion = BARO.readPressure();
  input->data.f[0] = presion;

  if (interpreter.Invoke() == kTfLiteOk) {
    float* pred = output->data.f;
    int altitud = std::distance(pred, std::max_element(pred, pred + 3));

    switch (altitud) {
      case 0: Serial.println("Altitud: Nivel del mar"); break;
      case 1: Serial.println("Altitud: Media"); break;
      case 2: Serial.println("Altitud: Montaña"); break;
    }
  } else {
    Serial.println("Error en la inferencia");
  }

  delay(1000);
}""", language='c')

elif section == "Diagrama UML":
    st.markdown("""Anexo: Diagrama UML del Proceso de Inferencia
El siguiente diagrama muestra el flujo típico de ejecución de un modelo de TensorFlow Lite en un microcontrolador Arduino Nano 33 BLE Sense.

Documentación Completa: Uso de Sensores del Arduino Nano 33 BLE Sense con TensorFlow Lite for Microcontrollers""")

elif section == "Diagrama UML":
    st.image("uml_inferencia_tflite.png", caption="Flujo de inferencia en Arduino con TFLite", use_column_width=True)

elif section == "Sensor Genérico":
    st.subheader("🎯 Objetivo")
    st.markdown("Este módulo permite adaptar la guía de TinyML para cualquier sensor que no esté listado específicamente, como sensores de luz, gas, sonido ambiental, presión arterial, entre otros.")

    st.subheader("🟢 a. Adquisición de Datos (Arduino)")
    st.markdown("Modifica el siguiente ejemplo para capturar los datos relevantes de tu sensor. Asegúrate de usar Serial.print() para exportar los datos.")

    st.code("""// Reemplaza con la librería de tu sensor
#include <SensorPersonalizado.h>

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Sensor.begin();  // Ajusta según el sensor
}

void loop() {
  float valor = Sensor.read();  // Reemplaza con el método de lectura
  Serial.println(valor);
  delay(1000);
}
""", language='c')

    st.subheader("📋 Ejemplo de archivo CSV generado:")
    st.code("""valor,label
0.52,NORMAL
0.81,ALTO
0.12,BAJO
...""", language='text')

    st.subheader("🧠 b. Entrenamiento del Modelo en Python")
    st.code("""import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("sensor_generico.csv")
X = df[["valor"]].values
y = df["label"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = tf.keras.utils.to_categorical(y_encoded)

model = Sequential([
    Dense(8, input_shape=(1,), activation='relu'),
    Dense(len(y_cat[0]), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_cat, epochs=30)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("modelo_sensor_generico.tflite", "wb") as f:
    f.write(tflite_model)
""", language='python')

    st.markdown("🔁 **Conversión a .h para incluir en Arduino**")
    st.code("xxd -i modelo_sensor_generico.tflite > modelo_sensor_generico.h", language='bash')

    st.subheader("🔁 c. Inferencia en Arduino")
    st.code("""#include <SensorPersonalizado.h>
#include "modelo_sensor_generico.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int kTensorArenaSize = 4 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = tflite::GetModel(modelo_sensor_generico_tflite);
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);

TfLiteTensor* input;
TfLiteTensor* output;

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Sensor.begin();

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Error al asignar tensores");
    while (1);
  }

  input = interpreter.input(0);
  output = interpreter.output(0);
}

void loop() {
  float valor = Sensor.read();
  input->data.f[0] = valor;

  if (interpreter.Invoke() == kTfLiteOk) {
    float* pred = output->data.f;
    int resultado = std::distance(pred, std::max_element(pred, pred + output->dims->data[1]));
    Serial.print("Predicción: ");
    Serial.println(resultado);
  } else {
    Serial.println("Error en la inferencia");
  }

  delay(1000);
}
""", language='c')

elif section == "Cámara (Kit TinyML)":
    st.subheader("🎯 Objetivo")
    st.markdown("Capturar imágenes desde el módulo de cámara del Kit TinyML (versión OV7675 para Arduino Nano 33 BLE Sense Rev1), visualizar las imágenes en Python o Processing, y preparar los datos para entrenar un modelo de clasificación o detección.")

    st.subheader("🟢 a. Captura de imágenes con Arduino")
    st.markdown("Este ejemplo utiliza el ejemplo oficial 'CameraCaptureRawBytes' del Kit TinyML, que transmite los datos binarios de la imagen por el puerto serie.")
    
    st.code("""#include "camera.h"

void setup() {
  Serial.begin(115200);
  while (!Serial);
  if (!Camera.begin()) {
    Serial.println("Camera failed to begin!");
    while (1);
  }
  Camera.setResolution(QQVGA); // 160x120
  Camera.setImageFormat(BITMAP);
}

void loop() {
  Camera.readFrame();
  Serial.write(Camera.getRGB565(), Camera.width() * Camera.height() * 2); // Enviar por serial en formato RGB565
  delay(100);
}
""", language='c')

    st.subheader("👁️ b. Visualización de imágenes en Python")
    st.markdown("Capturamos los bytes del puerto serie y reconstruimos la imagen usando Python:")

    st.code("""import serial
import numpy as np
import cv2

ser = serial.Serial('COM3', 115200)  # Cambia por tu puerto
while True:
    raw = ser.read(160 * 120 * 2)  # 2 bytes por píxel RGB565
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((120, 160, 2))

    def rgb565_to_rgb888(p):
        return [
            (p[0] & 0xF8),                       # R
            ((p[0] & 0x07) << 5) | ((p[1] & 0xE0) >> 3),  # G
            (p[1] & 0x1F) << 3                   # B
        ]

    img = np.array([[rgb565_to_rgb888(pixel) for pixel in row] for row in arr], dtype=np.uint8)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Camera", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
ser.close()
""", language='python')

    st.subheader("🖼️ c. Visualización alternativa en Processing")
    st.markdown("También puedes usar Processing para visualizar imágenes desde el puerto serie:")

    st.code("""import processing.serial.*;

Serial myPort;
PImage img;

void setup() {
  size(160, 120);
  img = createImage(160, 120, RGB);
  myPort = new Serial(this, "COM3", 115200);  // Cambia por tu puerto
  myPort.buffer(160 * 120 * 2);
}

void draw() {
  if (myPort.available() >= 160 * 120 * 2) {
    for (int i = 0; i < 160 * 120; i++) {
      int byte1 = myPort.read();
      int byte2 = myPort.read();
      int r = byte1 & 0xF8;
      int g = ((byte1 & 0x07) << 5) | ((byte2 & 0xE0) >> 3);
      int b = (byte2 & 0x1F) << 3;
      img.pixels[i] = color(r, g, b);
    }
    img.updatePixels();
    image(img, 0, 0);
  }
}
""", language='java')

elif section == "Clasificación con YOLO (Cámara TinyML)":
    st.subheader("🎯 Objetivo")
    st.markdown("Implementar una solución de clasificación de objetos con la cámara del Kit TinyML, utilizando un modelo TinyYOLO o una variante simplificada compatible con microcontroladores.")

    st.subheader("🧠 a. Entrenamiento del modelo en Python")
    st.markdown("Utilizaremos una red convolucional compacta inspirada en YOLO, entrenada con imágenes recolectadas desde la cámara (por ejemplo, para clasificar entre 'luz encendida' y 'luz apagada').")

    st.code("""import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import cv2
import os

# Cargar imágenes
def load_data(path):
    X, y = [], []
    for label in os.listdir(path):
        for img_name in os.listdir(os.path.join(path, label)):
            img_path = os.path.join(path, label, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (96, 96))  # Tamaño pequeño para microcontrolador
            X.append(img / 255.0)
            y.append(label)
    return np.array(X), np.array(y)

X, y = load_data("dataset")
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = tf.keras.utils.to_categorical(y_encoded)

model = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape=(96, 96, 3)),
    MaxPooling2D(),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(len(y_cat[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_cat, epochs=20)

# Convertir a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("modelo_yolo_camara.tflite", "wb") as f:
    f.write(tflite_model)
""", language='python')

    st.markdown("🔁 **Conversión a .h para Arduino**")
    st.code("xxd -i modelo_yolo_camara.tflite > modelo_yolo_camara.h", language='bash')

    st.subheader("🔁 b. Inferencia en Arduino con cámara")
    st.markdown("Este ejemplo usa una arquitectura muy ligera basada en visión para inferencia en vivo.")

    st.code("""#include "modelo_yolo_camara.h"
#include "camera.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int kTensorArenaSize = 100 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = tflite::GetModel(modelo_yolo_camara_tflite);
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);

TfLiteTensor* input;
TfLiteTensor* output;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!Camera.begin()) {
    Serial.println("No se pudo inicializar la cámara");
    while (1);
  }
  Camera.setResolution(QQVGA);  // 160x120
  Camera.setImageFormat(GRAYSCALE);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Error al asignar tensores");
    while (1);
  }

  input = interpreter.input(0);
  output = interpreter.output(0);
}

void loop() {
  Camera.readFrame();
  memcpy(input->data.uint8, Camera.getRGB565(), Camera.width() * Camera.height());  // simplificado

  if (interpreter.Invoke() == kTfLiteOk) {
    float* prediction = output->data.f;
    int clase = std::distance(prediction, std::max_element(prediction, prediction + output->dims->data[1]));
    Serial.print("Clase detectada: ");
    Serial.println(clase);
  } else {
    Serial.println("Error en la inferencia");
  }

  delay(1000);
}
""", language='c')

elif section == "Análisis Temporal con LSTM (IMU)":
    st.subheader("🎯 Objetivo")
    st.markdown("Implementar una red LSTM para reconocer patrones en señales temporales como aceleración o giroscopio del sensor IMU (LSM9DS1) del Arduino Nano 33 BLE Sense.")

    st.subheader("📦 a. Adquisición de Secuencias en Arduino")
    st.markdown("Recolectamos múltiples muestras consecutivas del acelerómetro para formar secuencias temporales.")

    st.code("""#include <Arduino_LSM9DS1.h>

#define SEQ_LEN 20

void setup() {
  Serial.begin(115200);
  while (!Serial);
  if (!IMU.begin()) {
    Serial.println("Error al iniciar IMU");
    while (1);
  }
}

void loop() {
  for (int i = 0; i < SEQ_LEN; i++) {
    float x, y, z;
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(x, y, z);
      Serial.print(x); Serial.print(",");
      Serial.print(y); Serial.print(",");
      Serial.println(z);
    }
    delay(100);
  }
  Serial.println("---");  // delimitador de secuencia
}
""", language='c')

    st.subheader("📋 b. Formato de Datos en CSV")
    st.markdown("Después de grabar secuencias en el monitor serial, agrúpalas con etiquetas:")

    st.code("""x1,y1,z1
x2,y2,z2
...
---
x1,y1,z1
x2,y2,z2
...
label
""", language='text')

    st.subheader("🧠 c. Entrenamiento con LSTM en Python")
    st.markdown("Transformamos las secuencias en tensores 3D para entrenar un modelo LSTM.")

    st.code("""import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder

# Cargar y segmentar datos
def cargar_secuencias(file):
    X, y = [], []
    with open(file) as f:
        seq = []
        for line in f:
            if line.strip() == "---":
                continue
            elif "," in line:
                seq.append([float(v) for v in line.strip().split(",")])
            elif len(line.strip()) > 0:
                X.append(seq)
                y.append(line.strip())
                seq = []
    return np.array(X), np.array(y)

X, y = cargar_secuencias("secuencias_lstm.csv")
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = tf.keras.utils.to_categorical(y_encoded)

model = Sequential([
    LSTM(32, input_shape=(X.shape[1], X.shape[2])),
    Dense(len(y_cat[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_cat, epochs=30)

# Exportar modelo
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("modelo_lstm_imu.tflite", "wb") as f:
    f.write(tflite_model)
""", language='python')

    st.markdown("🔁 **Conversión a .h**")
    st.code("xxd -i modelo_lstm_imu.tflite > modelo_lstm_imu.h", language='bash')

    st.subheader("🔁 d. Inferencia en Arduino (Secuencia)")
    st.markdown("Se recomienda guardar la secuencia en un búfer y normalizar los datos antes de enviarlos al modelo.")

    st.code("""# (Ejemplo básico de estructura)
float sequence[SEQ_LEN][3];  // Llenar con valores IMU en tiempo real
for (int i = 0; i < SEQ_LEN; i++) {
  input->data.f[i * 3 + 0] = sequence[i][0];  // x
  input->data.f[i * 3 + 1] = sequence[i][1];  // y
  input->data.f[i * 3 + 2] = sequence[i][2];  // z
}
// luego hacer: interpreter.Invoke(); y leer output
""", language='c')
