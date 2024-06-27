import cv2
import mediapipe as mp
import numpy as np

# Inicializar componentes de MediaPipe para la detección de rostros y el dibujo.
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

# Función para procesar y dibujar la malla facial en la imagen.
def process_and_draw(image):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        # Procesar la imagen.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Dibujar las mallas faciales en la imagen.
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Dibujar malla facial
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
                
                # Dibujar más grillas (opcional)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())

                # Dibujar líneas de iris
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
        
        return image, results

# Función para guardar solo la malla facial como PNG.
def save_face_mesh_only(image_path, output_path):
    # Cargar la imagen.
    image = cv2.imread(image_path)
    
    # Verificar si la imagen se cargó correctamente.
    if image is None:
        print(f"Error: No se pudo cargar la imagen desde '{image_path}'.")
        return
    
    # Procesar y dibujar la malla facial en la imagen original.
    processed_image, results = process_and_draw(image)
    
    # Mostrar la imagen resultante con la malla facial dibujada.
    cv2.imshow('MediaPipe FaceMesh', processed_image)
    cv2.waitKey(0)
    
    # Guardar la imagen con la malla facial dibujada encima.
    cv2.imwrite(output_path, processed_image)
    print(f"Imagen del grafo con fondo guardada en '{output_path}'.")

    # Crear una imagen vacía para dibujar solo la malla facial.
    blank_image = np.zeros_like(image)
    
    # Dibujar las mallas faciales en la imagen vacía.
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dibujar malla facial
            mp_drawing.draw_landmarks(
                image=blank_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
            
            # Dibujar más grillas (opcional)
            mp_drawing.draw_landmarks(
                image=blank_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())

            # Dibujar líneas de iris
            mp_drawing.draw_landmarks(
                image=blank_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

    # Guardar solo la malla facial dibujada como PNG sin fondo.
    cv2.imwrite('images/grafo_sin_fondo.png', blank_image)
    print("Imagen del grafo sin fondo guardada en 'images/grafo_sin_fondo.png'.")

# Función principal.
def main():
    # Rutas de entrada y salida.
    image_path = 'images/yo.jpg'
    output_path = 'images/grafo_con_fondo.png'
    
    # Procesar y guardar el grafo con la malla facial dibujada encima.
    save_face_mesh_only(image_path, output_path)

if __name__ == '__main__':
    main()
