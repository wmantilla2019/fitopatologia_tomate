# Aplicación para la detección de enfermedades en plantas de Tomate
# Creado por: Edinson Cortes - William Mantilla
# Feb. 21 de 2026
# Funciona Ok

import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QLabel, QFileDialog, QWidget, QFrame)
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtCore import Qt

# --- CONFIGURACIÓN ---
# Nombre del archivo del modelo que debe estar en la misma carpeta del script
NOMBRE_MODELO = 'CNN-7_Model64.h5' 

class DetectorTomates(QMainWindow):
    def __init__(self):
        """
        Constructor de la clase: Inicializa las constantes, 
        diccionarios de traducción y configura la interfaz base.
        """
        super().__init__()

        # Dimensiones de imagen requeridas por la red neuronal (64x64 píxeles)
        self.IMG_DIMENSIONS = (64, 64)
        
        # Mapeo de índices numéricos a nombres de clase originales del dataset
        self.CLASS_INDICES = {
            'Tomato___Bacterial_spot': 0,
            'Tomato___Early_blight': 1,
            'Tomato___Late_blight': 2,
            'Tomato___Septoria_leaf_spot': 3,
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 4,
            'Tomato___Tomato_mosaic_virus': 5,
            'Tomato___healthy': 6
        }
        
        # Traducciones amigables para mostrar al usuario final (agricultor)
        self.TRADUCCIONES = {
            'Tomato___Bacterial_spot': 'Mancha Bacteriana',
            'Tomato___Early_blight': 'Tizón Temprano',
            'Tomato___Late_blight': 'Tizón Tardío',
            'Tomato___Septoria_leaf_spot': 'Mancha Foliar (Septoria)',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Virus de la Cuchara (Rizado Amarillo)',
            'Tomato___Tomato_mosaic_virus': 'Virus del Mosaico',
            'Tomato___healthy': 'Planta Sana'
        }

        # Genera un diccionario inverso (número -> nombre) para facilitar la búsqueda tras la predicción
        self.indice_a_clave = {v: k for k, v in self.CLASS_INDICES.items()}
        
        # Variables para almacenar el modelo y la ruta de la imagen seleccionada
        self.modelo = None
        self.ruta_imagen = None

        # Llamada a los métodos de construcción de UI y carga de modelo
        self.initUI()
        self.cargar_modelo_al_inicio()

    def initUI(self):
        """
        Define la estructura visual de la aplicación: botones, etiquetas y layouts.
        """
        self.setWindowTitle("IA Fitopatología: Diagnóstico de Tomate")
        self.resize(700, 850)

        # Contenedor principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout_principal = QVBoxLayout(central_widget)

        # 1. Título con formato específico
        titulo = QLabel("Panel de Diagnóstico Fitopatológico")
        font_titulo = QFont("Segoe UI", 20)
        font_titulo.setBold(True)
        titulo.setFont(font_titulo)
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet("color: #1b5e20; margin: 10px;")
        layout_principal.addWidget(titulo)

        # 2. Área de visualización de la imagen cargada
        self.lbl_imagen = QLabel("Seleccione una imagen para analizar")
        self.lbl_imagen.setAlignment(Qt.AlignCenter)
        self.lbl_imagen.setFrameShape(QFrame.StyledPanel)
        self.lbl_imagen.setMinimumSize(450, 450)
        self.lbl_imagen.setStyleSheet("background-color: #f1f8e9; border: 2px dashed #a5d6a7; color: #555;")
        layout_principal.addWidget(self.lbl_imagen)

        # 3. Etiquetas de resultados (Español, Técnico y Porcentaje)
        self.lbl_resultado_es = QLabel("Resultado: -")
        font_es = QFont("Segoe UI", 18)
        font_es.setBold(True)
        self.lbl_resultado_es.setFont(font_es)
        self.lbl_resultado_es.setAlignment(Qt.AlignCenter)
        layout_principal.addWidget(self.lbl_resultado_es)

        self.lbl_resultado_original = QLabel("ID Modelo: -")
        self.lbl_resultado_original.setFont(QFont("Consolas", 10))
        self.lbl_resultado_original.setAlignment(Qt.AlignCenter)
        self.lbl_resultado_original.setStyleSheet("color: #78909c;")
        layout_principal.addWidget(self.lbl_resultado_original)

        self.lbl_confianza = QLabel("Confianza: -")
        self.lbl_confianza.setFont(QFont("Segoe UI", 12))
        self.lbl_confianza.setAlignment(Qt.AlignCenter)
        layout_principal.addWidget(self.lbl_confianza)

        # --- BOTONES DE ACCIÓN ---
        layout_botones = QHBoxLayout()
        
        # Botón para buscar imagen en el equipo
        btn_cargar_img = QPushButton("📸 Seleccionar Imagen")
        btn_cargar_img.setMinimumHeight(50)
        btn_cargar_img.clicked.connect(self.cargar_imagen)
        
        # Botón para ejecutar la red neuronal
        self.btn_predecir = QPushButton("🔬 Iniciar Diagnóstico")
        self.btn_predecir.setMinimumHeight(50)
        self.btn_predecir.setEnabled(False) # Se activa solo si hay imagen y modelo
        self.btn_predecir.setStyleSheet("""
            QPushButton { background-color: #2e7d32; color: white; font-weight: bold; border-radius: 8px; font-size: 14px; }
            QPushButton:disabled { background-color: #cfd8dc; color: #90a4ae; }
            QPushButton:hover { background-color: #1b5e20; }
        """)
        self.btn_predecir.clicked.connect(self.ejecutar_diagnostico)

        # Botón para resetear la pantalla
        btn_limpiar = QPushButton("🧹 Limpiar datos")
        btn_limpiar.setMinimumHeight(50)
        btn_limpiar.setStyleSheet("""
            QPushButton { background-color: #546e7a; color: white; font-weight: bold; border-radius: 8px; font-size: 14px; }
            QPushButton:hover { background-color: #455a64; }
        """)
        btn_limpiar.clicked.connect(self.limpiar_datos)

        layout_botones.addWidget(btn_cargar_img)
        layout_botones.addWidget(self.btn_predecir)
        layout_botones.addWidget(btn_limpiar)
        layout_principal.addLayout(layout_botones)
        
        self.statusBar().showMessage("Cargando sistema...")

    def cargar_modelo_al_inicio(self):
        """
        Intenta cargar el archivo de modelo .h5 de forma automática.
        Si no existe, informa al usuario mediante etiquetas.
        """
        if os.path.exists(NOMBRE_MODELO):
            try:
                self.modelo = load_model(NOMBRE_MODELO)
                self.statusBar().showMessage("IA lista para el agricultor.")
            except Exception as e:
                self.lbl_resultado_es.setText("❌ Error de carga")
                print(f"Error técnico: {e}")
        else:
            self.lbl_resultado_es.setText(f"❌ Falta {NOMBRE_MODELO}")

    def limpiar_datos(self):
        """
        Restaura la aplicación a su estado de bienvenida, 
        borrando variables de imagen y etiquetas de texto.
        """
        self.ruta_imagen = None
        self.lbl_imagen.clear()
        self.lbl_imagen.setText("Seleccione una imagen para analizar")
        self.lbl_resultado_es.setText("Resultado: -")
        self.lbl_resultado_es.setStyleSheet("color: black; font-weight: bold;")
        self.lbl_resultado_original.setText("ID Modelo: -")
        self.lbl_confianza.setText("Confianza: -")
        self.btn_predecir.setEnabled(False)
        self.statusBar().showMessage("Datos limpiados. Esperando nueva imagen.")

    def cargar_imagen(self):
        """
        Abre un explorador de archivos para seleccionar una imagen.
        Muestra la imagen en el visor central si es válida.
        """
        archivo, _ = QFileDialog.getOpenFileName(self, "Abrir Imagen", "", "Imágenes (*.jpg *.jpeg *.png)")
        if archivo:
            self.ruta_imagen = archivo
            pixmap = QPixmap(archivo)
            # Escala la imagen para que quepa en el visor manteniendo su proporción
            self.lbl_imagen.setPixmap(pixmap.scaled(self.lbl_imagen.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # Solo permite predecir si el modelo fue cargado previamente
            if self.modelo:
                self.btn_predecir.setEnabled(True)
                self.lbl_resultado_es.setText("Imagen cargada correctamente")

    def ejecutar_diagnostico(self):
        """
        Realiza el proceso de Machine Learning:
        1. Pre-procesa la imagen (Resize, Normalización, Batch).
        2. Realiza la predicción con el modelo cargado.
        3. Traduce el resultado y actualiza la interfaz con colores.
        """
        if not self.modelo or not self.ruta_imagen: return

        try:
            self.statusBar().showMessage("Procesando imagen...")
            
            # --- 1. PRE-PROCESAMIENTO ---
            # Carga y redimensiona a 64x64
            img = image.load_img(self.ruta_imagen, target_size=self.IMG_DIMENSIONS)
            # Convierte imagen a arreglo numérico (NumPy)
            img_array = image.img_to_array(img)
            # Normalización (de 0-255 a 0-1)
            img_array /= 255.0
            # Expande dimensiones para simular un "lote" de imágenes (1, 64, 64, 3)
            img_batch = np.expand_dims(img_array, axis=0)

            # --- 2. PREDICCIÓN ---
            predicciones = self.modelo.predict(img_batch)
            # Toma el índice con el valor de probabilidad más alto
            indice = np.argmax(predicciones[0])
            # Obtiene el porcentaje de confianza
            confianza = np.max(predicciones[0]) * 100

            # --- 3. INTERPRETACIÓN ---
            nombre_original = self.indice_a_clave[indice]
            nombre_agricultor = self.TRADUCCIONES.get(nombre_original, "Desconocido")

            # Actualiza las etiquetas de texto
            self.lbl_resultado_es.setText(f"Diagnóstico: {nombre_agricultor}")
            self.lbl_resultado_original.setText(f"Etiqueta del Modelo: {nombre_original}")
            self.lbl_confianza.setText(f"Nivel de Confianza: {confianza:.2f}%")
            
            # Cambia el color del texto: Verde si está sano, Rojo si tiene enfermedad
            color = "#2e7d32" if "healthy" in nombre_original else "#c62828"
            self.lbl_resultado_es.setStyleSheet(f"color: {color}; font-weight: bold;")
            
            self.statusBar().showMessage("Análisis finalizado.")

        except Exception as e:
            self.lbl_resultado_es.setText(f"Error: {str(e)}")

# Punto de entrada de la aplicación
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Estilo Fusion para una apariencia moderna y uniforme en Windows/Linux/Mac
    app.setStyle("Fusion")
    ventana = DetectorTomates()
    ventana.show()
    sys.exit(app.exec())
