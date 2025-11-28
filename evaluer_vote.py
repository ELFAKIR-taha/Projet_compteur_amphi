import os
import cv2
import numpy as np
import sys

# Ajout du chemin pour trouver les modules si lancé depuis la racine
sys.path.append(os.path.abspath("detection"))

# --- IMPORT DU MODULE DE VOTE ---
try:
    from detection.vote import HandDetector
except ImportError:
    from detection.vote import HandDetector

# --- CONFIGURATION DES CHEMINS ---
DATASET_PATH = "dataset"
IMG_DIR = os.path.join(DATASET_PATH, "images")
LBL_DIR = os.path.join(DATASET_PATH, "labels")

# Racine du dossier d'évaluation
EVAL_ROOT = "evaluation"
# Dossier spécifique pour les prédictions "vote"
OUT_DIR = os.path.join(EVAL_ROOT, "predictions_vote")
# Dossier commun pour les résultats
RES_DIR = os.path.join(EVAL_ROOT, "resultats")

# Création des dossiers
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# Fichier de résultat final
RESULT_FILE = os.path.join(RES_DIR, "resultats_vote.txt")

# Fichiers modèles
HEAD_MODEL = "yolo_head_test.pt"
POSE_MODEL = "yolov8x-pose-p6.pt"

# --- PARAMÈTRES D'ÉVALUATION ---
# Rayon de tolérance pour valider une détection (proportionnel à la taille de la tête)
MATCHING_RADIUS_RATIO = 1.0

def charger_labels_vote(label_path, w_img, h_img):
    """
    Charge les labels pour le vote.
    Hypothèse standard : 
      - Classe 0 = Main GAUCHE
      - Classe 1 = Main DROITE
    """
    gt_hands = []
    if not os.path.exists(label_path): return []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.split()))
            if len(parts) < 5: continue
            
            cls = int(parts[0])
            
            # On ne garde que les classes 0 et 1 pour le vote
            if cls in [0, 1]: 
                xc, yc = parts[1], parts[2]
                abs_x = xc * w_img
                abs_y = yc * h_img
                gt_hands.append({'cls': cls, 'x': abs_x, 'y': abs_y, 'matched': False})
                
    return gt_hands

def main():
    # Instanciation du détecteur de VOTE
    print("Chargement des modèles pour le Vote...")
    if not os.path.exists(HEAD_MODEL) or not os.path.exists(POSE_MODEL):
        print(f"ATTENTION : Modèles introuvables ({HEAD_MODEL} ou {POSE_MODEL})")

    detector = HandDetector(head_model_path=HEAD_MODEL, pose_model_path=POSE_MODEL)
    
    image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Compteurs globaux
    global_TP = 0
    global_FP = 0
    global_FN = 0
    
    # Compteurs spécifiques par côté
    stats_gauche = {'TP': 0, 'FP': 0, 'FN': 0} # Classe 0 (Gauche)
    stats_droite = {'TP': 0, 'FP': 0, 'FN': 0} # Classe 1 (Droite)

    print(f"Début de l'évaluation VOTE sur {len(image_files)} images...")
    
    for img_name in image_files:
        img_path = os.path.join(IMG_DIR, img_name)
        lbl_path = os.path.join(LBL_DIR, os.path.splitext(img_name)[0] + ".txt")
        
        img = cv2.imread(img_path)
        if img is None: continue
        h_img, w_img = img.shape[:2]

        # --- A. DÉTECTION ---
        # detector.detect retourne une liste de TÊTES.
        heads_results = detector.detect(img)
        
        # APLATISSEMENT : On extrait toutes les mains de toutes les têtes
        predictions = []
        for head in heads_results:
            # head['hands'] contient la liste des dictionnaires mains
            predictions.extend(head['hands'])
        
        # --- B. EXPORTATION (Format YOLO) ---
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        with open(os.path.join(OUT_DIR, txt_name), "w") as f:
            for p in predictions:
                xn = p['x'] / w_img
                yn = p['y'] / h_img
                # Utilisation de head_h (qui doit être présent dans le dict main)
                hn = (p['head_h'] * 0.5) / h_img 
                wn = hn 
                
                # Conversion Side -> Class ID
                # 'G' (Gauche) -> 0
                # 'D' (Droite) -> 1
                cls_id = 0 if p['side'] == 'G' else 1
                
                f.write(f"{cls_id} {xn:.6f} {yn:.6f} {wn:.6f} {hn:.6f}\n")

        # --- C. COMPARAISON ---
        gt_hands = charger_labels_vote(lbl_path, w_img, h_img)
        
        img_TP = 0
        matched_pred_indices = set()
        
        for gt in gt_hands:
            best_dist = float('inf')
            best_pred_idx = -1
            
            for idx_p, pred in enumerate(predictions):
                if idx_p in matched_pred_indices: continue
                
                dist = np.hypot(gt['x'] - pred['x'], gt['y'] - pred['y'])
                # Seuil dynamique basé sur la taille de la tête associée à la main prédite
                threshold = pred['head_h'] * MATCHING_RADIUS_RATIO
                
                if dist < threshold and dist < best_dist:
                    # Vérification de la CLASSE (Gauche vs Droite)
                    pred_cls = 0 if pred['side'] == 'G' else 1
                    
                    if pred_cls == gt['cls']:
                        best_dist = dist
                        best_pred_idx = idx_p
            
            if best_pred_idx != -1:
                # MATCH VALIDE
                gt['matched'] = True
                matched_pred_indices.add(best_pred_idx)
                img_TP += 1
                
                if gt['cls'] == 0: stats_gauche['TP'] += 1
                else: stats_droite['TP'] += 1
            else:
                # PAS DE MATCH (FN)
                if gt['cls'] == 0: stats_gauche['FN'] += 1
                else: stats_droite['FN'] += 1

        img_FP = len(predictions) - len(matched_pred_indices)
        img_FN = len(gt_hands) - img_TP
        
        global_TP += img_TP
        global_FP += img_FP
        global_FN += img_FN
        
        # Attribution des FP aux classes (basé sur la prédiction)
        for idx_p, pred in enumerate(predictions):
            if idx_p not in matched_pred_indices:
                pred_cls = 0 if pred['side'] == 'G' else 1
                if pred_cls == 0: stats_gauche['FP'] += 1
                else: stats_droite['FP'] += 1

        print(f"{img_name:<20} | TP={img_TP} FP={img_FP} FN={img_FN} | Preds:{len(predictions)} GT:{len(gt_hands)}")

    # --- D. CALCUL DES MÉTRIQUES ---
    def calc_metrics(tp, fp, fn):
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        return prec, rec, f1

    g_prec, g_rec, g_f1 = calc_metrics(global_TP, global_FP, global_FN)
    l_prec, l_rec, l_f1 = calc_metrics(stats_gauche['TP'], stats_gauche['FP'], stats_gauche['FN'])
    r_prec, r_rec, r_f1 = calc_metrics(stats_droite['TP'], stats_droite['FP'], stats_droite['FN'])

    res_txt = (
        "=== RÉSULTATS DÉTECTION VOTE (Gauche/Droite) ===\n\n"
        f"Images traitées : {len(image_files)}\n\n"
        "--- GLOBAL ---\n"
        f"TP : {global_TP}\n"
        f"FP : {global_FP}\n"
        f"FN : {global_FN}\n"
        f"Precision : {g_prec:.4f}\n"
        f"Rappel    : {g_rec:.4f}\n"
        f"F1-Score  : {g_f1:.4f}\n\n"
        "--- DÉTAIL GAUCHE (Classe 0/Bleu) ---\n"
        f"TP: {stats_gauche['TP']} | FP: {stats_gauche['FP']} | FN: {stats_gauche['FN']}\n"
        f"F1-Score  : {l_f1:.4f}\n\n"
        "--- DÉTAIL DROITE (Classe 1/Rouge) ---\n"
        f"TP: {stats_droite['TP']} | FP: {stats_droite['FP']} | FN: {stats_droite['FN']}\n"
        f"F1-Score  : {r_f1:.4f}\n"
    )
    
    print("\n" + res_txt)
    
    with open(RESULT_FILE, "w") as f:
        f.write(res_txt)

    print(f"Rapport sauvegardé dans {RESULT_FILE}")
    print(f"Prédictions détaillées dans {OUT_DIR}")

if __name__ == "__main__":
    main()