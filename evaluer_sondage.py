import os
import cv2
import numpy as np
import sys

# Ajout du chemin pour trouver les modules si lancé depuis la racine
sys.path.append(os.path.abspath("detection"))

# IMPORTATION DE LA CLASSE SONDAGE
try:
    from detection.sondage import SondageDetector
except ImportError:
    # Fallback si le nom de la classe est différent ou import direct
    from detection.sondage import HandDetector as SondageDetector

# --- CONFIGURATION DES CHEMINS ---
DATASET_PATH = "dataset"
IMG_DIR = os.path.join(DATASET_PATH, "images")
LBL_DIR = os.path.join(DATASET_PATH, "labels")

# Racine du dossier d'évaluation
EVAL_ROOT = "evaluation"
# Dossier spécifique pour les prédictions "sondage"
OUT_DIR = os.path.join(EVAL_ROOT, "predictions_sondage")
# Dossier commun pour les résultats
RES_DIR = os.path.join(EVAL_ROOT, "resultats")

# Création des dossiers
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# Fichier de résultat final pour le sondage
RESULT_FILE = os.path.join(RES_DIR, "resultats_sondage.txt")

# Fichiers modèles
HEAD_MODEL = "yolo_head_test.pt"
POSE_MODEL = "yolov8x-pose-p6.pt"

# --- PARAMÈTRES D'ÉVALUATION ---
MATCHING_RADIUS_RATIO = 1.0

def charger_labels_gt(label_path, w_img, h_img):
    #Charge les classes 0 et 1 (MAINS) et retourne les pixels (x, y).
    gt_hands = []
    if not os.path.exists(label_path): return []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.split()))
            if len(parts) >= 3:
                cls = int(parts[0])
                # Classes 0 et 1 = Mains
                if cls in [0, 1]: 
                    xc, yc = parts[1], parts[2]
                    abs_x = xc * w_img
                    abs_y = yc * h_img
                    gt_hands.append((abs_x, abs_y))
                
    return gt_hands

def main():
    if not os.path.exists(HEAD_MODEL) or not os.path.exists(POSE_MODEL):
        print(f"ATTENTION : Modèles introuvables ({HEAD_MODEL} ou {POSE_MODEL})")
    
    detector = SondageDetector(head_model_path=HEAD_MODEL, pose_model_path=POSE_MODEL)
    
    if not os.path.exists(IMG_DIR):
        print(f"Erreur : Dossier images introuvable : {IMG_DIR}")
        return

    image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    total_TP = 0
    total_FP = 0
    total_FN = 0
    
    print(f"Début de l'évaluation SONDAGE sur {len(image_files)} images...")
    print(f"Critère de succès : Distance < {MATCHING_RADIUS_RATIO} * Hauteur_Tête")

    for img_name in image_files:
        img_path = os.path.join(IMG_DIR, img_name)
        lbl_path = os.path.join(LBL_DIR, os.path.splitext(img_name)[0] + ".txt")
        
        img = cv2.imread(img_path)
        if img is None: continue
        h_img, w_img = img.shape[:2]

        # --- A. DÉTECTION ---
        heads_results = detector.detect_sondage(img)
        
        # On doit "aplatir" la structure pour récupérer la liste de toutes les mains détectées
        # pour la comparaison spatiale avec le Ground Truth.
        predictions = []
        for head in heads_results:
            # Chaque 'head' contient une liste 'hands' avec les infos (x, y, conf, head_h...)
            predictions.extend(head['hands'])
        
        # --- B. EXPORTATION (Format YOLO) ---
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        with open(os.path.join(OUT_DIR, txt_name), "w") as f:
            for p in predictions:
                xn = p['x'] / w_img
                yn = p['y'] / h_img
                # On utilise head_h pour estimer une taille de boite main approximative
                hn = (p['head_h'] * 0.5) / h_img 
                wn = hn 
                # Classe 1 arbitraire pour "Main" dans l'output
                f.write(f"1 {xn:.6f} {yn:.6f} {wn:.6f} {hn:.6f}\n")

        # --- C. COMPARAISON ---
        gt_hands = charger_labels_gt(lbl_path, w_img, h_img)
        
        img_TP = 0
        matched_gt_indices = set()
        
        for pred in predictions:
            px, py = pred['x'], pred['y']
            # head_h est stocké dans chaque main par le détecteur vote.py
            dynamic_threshold = pred['head_h'] * MATCHING_RADIUS_RATIO
            
            best_dist = float('inf')
            best_gt_idx = -1
            
            for idx_gt, (gx, gy) in enumerate(gt_hands):
                if idx_gt in matched_gt_indices: continue
                
                dist = np.hypot(px - gx, py - gy)
                if dist < best_dist:
                    best_dist = dist
                    best_gt_idx = idx_gt
            
            if best_gt_idx != -1 and best_dist <= dynamic_threshold:
                img_TP += 1
                matched_gt_indices.add(best_gt_idx)
        
        img_FP = len(predictions) - img_TP
        img_FN = len(gt_hands) - img_TP
        
        total_TP += img_TP
        total_FP += img_FP
        total_FN += img_FN
        
        print(f"{img_name:<15} | TP={img_TP} FP={img_FP} FN={img_FN} | Preds:{len(predictions)} GT:{len(gt_hands)}")

    # --- D. RÉSULTATS GLOBAUX ---
    prec = total_TP / (total_TP + total_FP + 1e-9)
    rec = total_TP / (total_TP + total_FN + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    
    res_txt = (
        "\n=== RÉSULTATS DÉTECTION SONDAGE (Basé sur vote.py) ===\n"
        f"Images : {len(image_files)}\n"
        f"TP : {total_TP}\n"
        f"FP : {total_FP}\n"
        f"FN : {total_FN}\n"
        f"Precision : {prec:.4f}\n"
        f"Rappel    : {rec:.4f}\n"
        f"F1-Score  : {f1:.4f}\n"
    )
    
    print(res_txt)
    
    with open(RESULT_FILE, "w") as f:
        f.write(res_txt)

    print(f"Rapport sauvegardé dans {RESULT_FILE}")

if __name__ == "__main__":
    main()