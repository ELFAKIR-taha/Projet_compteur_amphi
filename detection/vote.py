import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os

# --- PARAMÈTRES ---
DEFAULT_CONFIG = {
    'BASE_HAND_CONF_THRESH': 0.60,
    'ELBOW_CONF_THRESH': 0.50,
    'SHOULDER_CONF_THRESH': 0.30,
    'BASE_DEDUP_DIST': 55,       
    'SHOULDER_DEDUP_DIST': 40,   
    'SUPER_STRICT_DIST': 10,     
    'VERTICAL_ALIGN_TOL': 20,    
    'VERTICAL_SEARCH_FACTOR': 1.4, 
    'PAIR_WRIST_ELBOW_THRESH': 1.9,
    'PAIR_WRIST_SHOULDER_THRESH': 1.9,
    'PAIR_ELBOW_SHOULDER_THRESH': 1.9,
    'TOTAL_CONF_THRESH': 1.70,
    'POSE_MODEL_CONF': 0.01,
    'BASE_ANGLE_THRESH_DEG': 75,
    'MIN_WRIST_ELBOW_RATIO': 0.7,
    'HEAD_HEIGHT_MULTIPLIER': 6.0,
    'HEAD_CLASS_ID': 0
}

KEYPOINT_IDX = {
    'NOSE': 0,
    'LEFT_SHOULDER': 5, 'RIGHT_SHOULDER': 6,
    'LEFT_ELBOW': 7, 'RIGHT_ELBOW': 8,
    'LEFT_WRIST': 9, 'RIGHT_WRIST': 10
}

# --- COULEURS SQUELETTE ---
COLOR_LEFT = (255, 0, 0)    # BLEU
COLOR_RIGHT = (0, 255, 0)   # VERT

def clamp(v, a, b): return max(a, min(v, b))

def angle_between(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 180.0
    return np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))

def get_keypoint_coords(kpt, index, crop_x_offset, crop_y_offset):
    p = kpt[index]
    return (int(p[0]) + crop_x_offset, int(p[1]) + crop_y_offset, float(p[2]))

class HandDetector:
    def __init__(self, head_model_path="yolo_head_test.pt", pose_model_path="yolov8x-pose-p6.pt", config=None):
        self.cfg = config if config else DEFAULT_CONFIG
        print(f"Config chargée. Superposition stricte à {self.cfg['SUPER_STRICT_DIST']}px")
        self.head_model = YOLO(head_model_path)
        self.pose_model = YOLO(pose_model_path)

    @staticmethod
    def sauvegarder_vote_txt(results, fichier="historique.txt"):
        #Enregistre les résultats du VOTE.
        if not os.path.exists(fichier):
            with open(fichier, "w", encoding="utf-8") as f:
                f.write("")

        total_heads = len(results)
        nb_gauche = sum(1 for r in results if r['vote'] == 'G')
        nb_droite = sum(1 for r in results if r['vote'] == 'D')
        # On compte comme Abstention tout ce qui n'est pas G ou D (donc 'N', '2M', etc.)
        nb_abst = sum(1 for r in results if r['vote'] not in ['G', 'D'])

        if total_heads > 0:
            p_g = (nb_gauche / total_heads) * 100
            p_d = (nb_droite / total_heads) * 100
            p_a = (nb_abst / total_heads) * 100
        else:
            p_g = p_d = p_a = 0

        date_heure = datetime.now().strftime("%d/%m/%y à %Hh%M")
        ligne = f" [V]: {date_heure}, {p_g:.1f}% option A, {p_d:.1f}% option B et {p_a:.1f}% abstention ({nb_gauche + nb_abst + nb_droite} votes au total)\n"

        with open(fichier, "a", encoding="utf-8") as f:
            f.write(ligne)
        
        print(f"Sauvegarde effectuée dans {fichier}")

    def check_hand_smart(self, wrist, elbow, shoulder, angle_thresh_deg, min_dist):
        wx, wy, wc = wrist
        ex, ey, ec = elbow
        sx, sy, sc = shoulder

        if wy > sy - 10: return False, "POS"
        if abs(wy - ey) < min_dist: return False, "LEN"
        
        v = (wx - ex, wy - ey)
        if angle_between(v, (0, -1)) > angle_thresh_deg: return False, "ANG"

        is_strict_ok = (wc >= self.cfg['BASE_HAND_CONF_THRESH']) and \
                       (ec >= self.cfg['ELBOW_CONF_THRESH']) and \
                       (sc >= self.cfg['SHOULDER_CONF_THRESH'])
        if is_strict_ok: return True, "OK"

        rescued = False
        if (wc + ec) >= self.cfg['PAIR_WRIST_ELBOW_THRESH']: rescued = True
        elif (wc + sc) >= self.cfg['PAIR_WRIST_SHOULDER_THRESH']: rescued = True
        elif (ec + sc) >= self.cfg['PAIR_ELBOW_SHOULDER_THRESH']: rescued = True
        
        if (wc + ec + sc) < self.cfg['TOTAL_CONF_THRESH']: rescued = False
        return rescued, "RATT" if rescued else "FAIL"

    def _is_anatomical_duplicate_strict(self, candidate, accepted_hands):
        cx, cy = candidate['x'], candidate['y']     
        sx, sy = candidate['sx'], candidate['sy']   
        c_dist_base = candidate['dedup_dist'] 
        super_strict_limit = self.cfg['SUPER_STRICT_DIST'] 
        v_align_tol = self.cfg['VERTICAL_ALIGN_TOL']
        v_search_factor = self.cfg['VERTICAL_SEARCH_FACTOR']

        for hand in accepted_hands:
            hx, hy = hand['x'], hand['y']
            hsx, hsy = hand['sx'], hand['sy']

            dx = abs(cx - hx)
            dy = abs(cy - hy)
            dist_wrist = np.hypot(dx, dy)

            if dist_wrist < super_strict_limit: return True 
            if np.hypot(sx - hsx, sy - hsy) > self.cfg['SHOULDER_DEDUP_DIST']: continue 
            if dist_wrist < max(c_dist_base, hand['dedup_dist']): return True 
            if dx < v_align_tol and dy < (max(c_dist_base, hand['dedup_dist']) * v_search_factor): return True 

        return False

    def detect(self, img):
        if img is None: return []
        
        head_results = self.head_model(img, verbose=False)[0]
        
        heads_list = [] 
        all_candidates = [] 

        h_img, w_img = img.shape[:2]

        for i, box in enumerate(head_results.boxes):
            if int(box.cls[0]) != self.cfg['HEAD_CLASS_ID']: continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            h = y2 - y1
            
            heads_list.append({'id': i, 'box': (x1, y1, x2, y2), 'h': h})

            dedup_dist = max(20, int(self.cfg['BASE_DEDUP_DIST'] * (h / 100)))
            angle_thresh_deg = min(90, max(50, self.cfg['BASE_ANGLE_THRESH_DEG'] * (h / 150)))
            min_dist = max(5, h * self.cfg['MIN_WRIST_ELBOW_RATIO'])
            
            crop_size = int(h * self.cfg['HEAD_HEIGHT_MULTIPLIER'])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 - int(h * 1.2)
            half = crop_size // 2
            
            X1, Y1 = clamp(cx - half, 0, w_img), clamp(cy - half, 0, h_img)
            X2, Y2 = clamp(cx + half, 0, w_img), clamp(cy + half, 0, h_img)
            if (X2-X1) < 10: continue
            
            pose_results = self.pose_model(img[Y1:Y2, X1:X2], imgsz=448, conf=self.cfg['POSE_MODEL_CONF'], verbose=False)[0]
            if not hasattr(pose_results.keypoints, "data") or pose_results.keypoints.data is None: continue
            
            for k in pose_results.keypoints.data:
                nose = get_keypoint_coords(k, KEYPOINT_IDX['NOSE'], X1, Y1)
                nx, ny = nose[0], nose[1]

                if not (x1 <= nx <= x2 and y1 <= ny <= y2):
                    continue

                for side_name in ['LEFT', 'RIGHT']:
                    w = get_keypoint_coords(k, KEYPOINT_IDX[f'{side_name}_WRIST'], X1, Y1)
                    e = get_keypoint_coords(k, KEYPOINT_IDX[f'{side_name}_ELBOW'], X1, Y1)
                    s = get_keypoint_coords(k, KEYPOINT_IDX[f'{side_name}_SHOULDER'], X1, Y1)

                    is_valid, reason = self.check_hand_smart(w, e, s, angle_thresh_deg, min_dist)
                    if is_valid:
                        all_candidates.append({
                            'x': w[0], 'y': w[1],
                            'ex': e[0], 'ey': e[1],
                            'sx': s[0], 'sy': s[1],
                            'nx': nx,   'ny': ny,
                            'conf': w[2], 
                            'reason': reason, 
                            'dedup_dist': dedup_dist,
                            'head_h': h,
                            'side': 'G' if side_name == 'LEFT' else 'D',
                            'color': COLOR_LEFT if side_name == 'LEFT' else COLOR_RIGHT,
                            'head_id': i
                        })

        all_candidates.sort(key=lambda x: x['conf'], reverse=True)
        valid_hands_flat = []
        for cand in all_candidates:
            if not self._is_anatomical_duplicate_strict(cand, valid_hands_flat):
                valid_hands_flat.append(cand)
        
        final_results = []
        
        for head in heads_list:
            h_id = head['id']
            my_hands = [h for h in valid_hands_flat if h['head_id'] == h_id]
            
            has_left = any(h['side'] == 'G' for h in my_hands)
            has_right = any(h['side'] == 'D' for h in my_hands)
            
            vote_status = 'N' 
            
            if has_left and has_right:
                vote_status = 'N' 
            elif has_left:
                vote_status = 'G'
            elif has_right:
                vote_status = 'D'
            
            final_results.append({
                'head_id': h_id,
                'head_box': head['box'],
                'vote': vote_status,
                'hands': my_hands
            })

        return final_results