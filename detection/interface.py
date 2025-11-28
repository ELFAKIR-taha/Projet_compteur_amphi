import cv2
import tkinter as tk
from tkinter import filedialog
import sys

def choisir_source(parent_root):
    user_choice = None

    def select_webcam():
        nonlocal user_choice
        user_choice = 'webcam'
        window.destroy()

    def select_import():
        nonlocal user_choice
        user_choice = 'import'
        window.destroy()

    def on_close():
        # Si l'utilisateur ferme la croix, on ne fait rien (user_choice reste None)
        window.destroy()

    window = tk.Toplevel(parent_root)
    window.title("Source")
    window.protocol("WM_DELETE_WINDOW", on_close)

    label = tk.Label(window, text="Comment voulez-vous charger l'image ?", 
                     font=("Arial", 14), pady=20)
    label.pack()

    webcam_button = tk.Button(window, text="Activer la webcam", 
                              font=("Arial", 12), width=30, height=2, 
                              command=select_webcam)
    webcam_button.pack(pady=10)

    import_button = tk.Button(window, text="Importer un fichier image", 
                              font=("Arial", 12), width=30, height=2, 
                              command=select_import)
    import_button.pack(pady=10)
    
    # Important : on attend que cette fenêtre soit fermée avant de continuer le code
    window.wait_window()
    
    return user_choice


def obtenir_image(source, parent_root):

    if source == 'webcam':
        # 1. Tentative sur l'index 1 (caméra externe)
        print("Tentative d'ouverture de la webcam (Index 1)...")
        cap = cv2.VideoCapture(1) 

        # 2. Si l'index 1 ne s'ouvre pas, on tente l'index 0 (caméra intégrée)
        if not cap.isOpened():
            print("Echec index 1. Tentative sur l'index 0...")
            cap = cv2.VideoCapture(0)

        # 3. Si aucun des deux ne fonctionne
        if not cap.isOpened():
            print("Erreur critique : Impossible d'ouvrir une webcam (ni index 1, ni index 0).")
            return None

        # --- BOUCLE DE CAPTURE ---
        img = None
        while True:
            success, frame = cap.read()
            if not success:
                print("Erreur de lecture du flux vidéo.")
                break
            
            # Miroir pour confort utilisateur
            frame = cv2.flip(frame, 1)
            preview = frame.copy()
            cv2.putText(preview, "Appuyez sur 'c' pour capturer, 'q' pour quitter.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Previsualisation", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                img = frame
                break

        cap.release()
        cv2.destroyAllWindows()
        return img

    elif source == 'import':
        path = filedialog.askopenfilename(parent=parent_root, title="Choisir une image")

        if not path:
            return None # L'utilisateur a annulé

        return cv2.imread(path)

    return None