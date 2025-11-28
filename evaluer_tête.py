import os
import cv2
from detection.comptage import CompteurAmphi

# --- CONFIGURATION DES CHEMINS ---
DATASET_PATH = "dataset"
IMG_DIR = os.path.join(DATASET_PATH, "images")
LBL_DIR = os.path.join(DATASET_PATH, "labels")

# Racine du dossier d'évaluation
EVAL_ROOT = "evaluation"
# Dossier spécifique pour les prédictions "comptage"
OUT_DIR = os.path.join(EVAL_ROOT, "predictions_comptage")
# Dossier commun pour les rapports de résultats
RES_DIR = os.path.join(EVAL_ROOT, "resultats")

# Création des dossiers
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# Fichier de résultat final
RESULT_FILE = os.path.join(RES_DIR, "resultats_comptage.txt")


# ----------------------------------------------------
# Boîte englobante YOLO-format -> xyxy
# ----------------------------------------------------
def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return x1, y1, x2, y2

# ----------------------------------------------------
# IoU (Intersection over Union)
# ----------------------------------------------------
def iou(boxA, boxB):
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB

    # 1) Inclusion totale
    if (x1B >= x1A and y1B >= y1A and 
        x2B <= x2A and y2B <= y2A):
        return 1.0

    # 2) Intersection classique
    inter_x1 = max(x1A, x1B)
    inter_y1 = max(y1A, y1B)
    inter_x2 = min(x2A, x2B)
    inter_y2 = min(y2A, y2B)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0: return 0.0

    areaA = (x2A - x1A) * (y2A - y1A)
    areaB = (x2B - x1B) * (y2B - y1B)
    union = areaA + areaB - inter_area
    
    if union == 0: return 0.0

    return inter_area / union

# ----------------------------------------------------
# Compare une image
# ----------------------------------------------------
def comparer_image(image_path, label_path, compteur):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Impossible de charger l'image {image_path}.")
        return 0, 0, 0
        
    h, w, _ = img.shape

    # --- 1. Charger les Labels (Ground Truth) ---
    gt_boxes = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5: continue
                
                cls, xc, yc, ww, hh = map(float, parts)
                # FILTRE : On ne garde que la classe 2 (Têtes)
                if int(cls) == 2: 
                    gt_boxes.append(yolo_to_xyxy(xc, yc, ww, hh, w, h))
    except Exception as e:
        print(f"[ERROR] Lecture label {label_path}: {e}")
        return 0, 0, 0

    # --- 2. Générer la prédiction ---
    base = os.path.splitext(os.path.basename(image_path))[0]
    pred_file = os.path.join(OUT_DIR, base + ".txt")

    compteur.charger_image(img)
    compteur.compter()
    
    # Exportation dans le dossier predictions_comptage
    compteur.exporter_predictions(pred_file) 

    # --- 3. Relire la prédiction pour comparer ---
    pred_boxes = []
    if os.path.exists(pred_file):
        try:
            with open(pred_file, "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5: continue
                    cls, xc, yc, ww, hh = map(float, parts)
                    pred_boxes.append(yolo_to_xyxy(xc, yc, ww, hh, w, h))
        except Exception as e:
            print(f"[ERROR] Lecture prédiction {pred_file}: {e}")

    # --- 4. Calcul des métriques (TP, FP, FN) ---
    TP = 0
    used_pred = set()

    for gt in gt_boxes:
        best_iou = 0
        best_idx = -1
        
        for i, pr in enumerate(pred_boxes):
            if i in used_pred: continue
            
            score = iou(gt, pr)
            if score > best_iou:
                best_iou = score
                best_idx = i

        if best_iou >= 0.5:
            TP += 1
            used_pred.add(best_idx)

    FP = len(pred_boxes) - len(used_pred)
    FN = len(gt_boxes) - TP

    return TP, FP, FN
    
# ----------------------------------------------------
# Programme global
# ----------------------------------------------------
def comparer_dataset():
    compteur = CompteurAmphi(model_path='yolo_head_test.pt')

    total_TP = 0
    total_FP = 0
    total_FN = 0

    images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Début de l'analyse COMPTAGE sur {len(images)} images...")

    for img_name in images:
        base = os.path.splitext(img_name)[0]
        image_path = os.path.join(IMG_DIR, img_name)
        label_path = os.path.join(LBL_DIR, base + ".txt")

        if not os.path.exists(label_path):
            print(f"[WARN] Pas de label pour {img_name} -> ignoré")
            continue

        TP, FP, FN = comparer_image(image_path, label_path, compteur)

        total_TP += TP
        total_FP += FP
        total_FN += FN

        print(f"Comptage: {img_name:<20} | TP={TP:2} FP={FP:2} FN={FN:2}")

    # --- Calcul final ---
    denom_prec = total_TP + total_FP
    denom_rapp = total_TP + total_FN

    precision = total_TP / denom_prec if denom_prec > 0 else 0
    rappel = total_TP / denom_rapp if denom_rapp > 0 else 0
    
    f1 = 0
    if (precision + rappel) > 0:
        f1 = 2 * precision * rappel / (precision + rappel)

    # --- Sauvegarde ---
    with open(RESULT_FILE, "w") as f:
        f.write("=== Résultats COMPTAGE (Têtes/Classe 2) ===\n\n")
        f.write(f"Images traitées : {len(images)}\n")
        f.write(f"TP (Vrais Positifs)  = {total_TP}\n")
        f.write(f"FP (Faux Positifs)   = {total_FP}\n")
        f.write(f"FN (Faux Négatifs)   = {total_FN}\n\n")
        f.write(f"Precision = {precision:.4f}\n")
        f.write(f"Rappel    = {rappel:.4f}\n")
        f.write(f"F1-score  = {f1:.4f}\n")

    print("\n" + "="*40)
    print(f"Terminé. Résultats sauvegardés dans : {RESULT_FILE}")
    print(f"Fichiers détaillés dans : {OUT_DIR}")
    print("="*40)

if __name__ == "__main__":
    comparer_dataset()