import cv2
from ultralytics import YOLO
import os
from datetime import datetime


class CompteurAmphi:

    def __init__(self, model_path='yolo_head_test.pt', class_id_head=0):
        self.model = YOLO(model_path)
        self.CLASS_ID_HEAD = class_id_head
        self.image = None
        self.resultats = None
        self.count = 0
        self.image_annotee = None

    def charger_image(self, img):
        self.image = img

    def sauvegarder_nombre_etudiants(self, fichier="historique.txt"):
        # Créer le fichier si besoin
        if not os.path.exists(fichier):
            with open(fichier, "w", encoding="utf-8") as f:
                f.write("")

        # Format date & heure
        date_heure = datetime.now().strftime("%d/%m/%y à %Hh%M")

        # Ligne à écrire
        ligne = f"[C]: {date_heure}, Il y a {self.count} étudiants\n"

        # Écriture
        with open(fichier, "a", encoding="utf-8") as f:
            f.write(ligne)

    def compter(self, seuil=0.3):
        if self.image is None: return 0
        
        results = self.model(self.image)
        self.resultats = results
        self.count = 0
        self.image_annotee = self.image.copy()

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if cls_id == self.CLASS_ID_HEAD and confidence > seuil:
                    self.count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(self.image_annotee, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return self.count

    def exporter_predictions(self, output_path): 
        if self.resultats is None:
            raise ValueError("Appeler compter() avant exporter_predictions().")
        
        h, w = self.image.shape[:2]
        preds = []

        # Récupération des boîtes
        for r in self.resultats:
            for box in r.boxes:
                cls = int(box.cls[0])
                # On ne garde que la classe qui nous intéresse
                if cls != self.CLASS_ID_HEAD: continue
                    
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Conversion en format YOLO normalisé (xc, yc, w, h)
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                
                preds.append((cls, xc, yc, bw, bh))

        # Écriture directe dans le fichier de sortie
        with open(output_path, "w") as f:
            for p in preds:
                f.write(f"{p[0]} {p[1]:.6f} {p[2]:.6f} {p[3]:.6f} {p[4]:.6f}\n")

    def annoter(self):
        if self.image_annotee is None: return
        texte = f"Nombre total d'etudiants : {self.count}"
        cv2.putText(self.image_annotee, texte, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    def afficher(self):
        #Affiche l'image annotée et rouvre le menu quand on ferme la fenêtre
        if self.image_annotee is None:
            raise ValueError("Aucune image annotée à afficher.")

        cv2.imshow("Resultat du comptage", self.image_annotee)
        cv2.destroyAllWindows()
        cv2.waitKey(1)