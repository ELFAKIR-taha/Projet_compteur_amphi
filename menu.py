import sys
import os
import tkinter as tk
from tkinter import messagebox
import cv2

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- IMPORTS ---
sys.path.append(resource_path("detection"))
try:
    from detection.interface import choisir_source, obtenir_image
    from detection.comptage import CompteurAmphi
    from detection.sondage import SondageDetector
    from detection.vote import HandDetector as VoteDetector
except ImportError as e:
    try:
        from interface import choisir_source, obtenir_image
        from comptage import CompteurAmphi
        from sondage import HandDetector as SondageDetector
        from vote import HandDetector as VoteDetector
    except ImportError as e2:
        print(f"Erreur critique : {e}")
        sys.exit(1)

# =============================================================================
# FONCTIONS LOGIQUES
# =============================================================================
PATH_HEAD_MODEL = resource_path("yolo_head_test.pt")
PATH_POSE_MODEL = resource_path("yolov8x-pose-p6.pt")

def lancer_analyse(nom_mode, callback_logique):
    root.withdraw()
    try:
        source = choisir_source(root)
        if source:
            img = obtenir_image(source, root)
            if img is not None:
                print(f"--- Démarrage mode : {nom_mode} ---")
                callback_logique(img)
    except Exception as e:
        print(f"Erreur : {e}")
        messagebox.showerror("Erreur", str(e))
    finally:
        root.deiconify()

def run_comptage(img):
    compteur = CompteurAmphi(model_path=PATH_HEAD_MODEL)
    compteur.charger_image(img)
    compteur.compter()
    compteur.annoter()
    if compteur.image_annotee is not None:
        cv2.imshow("Resultat Comptage", compteur.image_annotee)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    compteur.sauvegarder_nombre_etudiants()

def run_sondage(img):

    if SondageDetector is None:
        raise ImportError("Module sondage manquant.")

    print("Chargement du modèle de Sondage...")
    detector = SondageDetector(head_model_path=PATH_HEAD_MODEL, pose_model_path=PATH_POSE_MODEL)

    results = detector.detect_sondage(img)

    count_pour = 0
    count_contre = 0

    COLOR_POUR = (0, 255, 0)     # vert
    COLOR_CONTRE = (0, 0, 255)   # rouge

    # — Parcours des résultats —
    for r in results:
        status = r['sondage']
        x1, y1, x2, y2 = r['head_box']

        if status == "POUR":
            color = COLOR_POUR
            count_pour += 1
        elif status == "CONTRE":
            color = COLOR_CONTRE
            count_contre += 1
        else:
            continue

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    total = count_pour + count_contre

    if total > 0:
        perc_pour = (count_pour / total) * 100
        perc_contre = (count_contre / total) * 100
    else:
        perc_pour = perc_contre = 0

    # Bandeau d'affichage
    cv2.rectangle(img, (0, 0), (350, 90), (0, 0, 0), -1)

    cv2.putText(img, f"POUR : {perc_pour:.1f}% ({count_pour})", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_POUR, 2)

    cv2.putText(img, f"CONTRE: {perc_contre:.1f}% ({count_contre})", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_CONTRE, 2)

    print("---- SONDAGE ----")
    print(f"Participants actifs : {total}")
    print(f"POUR = {count_pour} ({perc_pour:.1f}%)")
    print(f"CONTRE = {count_contre} ({perc_contre:.1f}%)")

    cv2.imshow("Sondage", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    SondageDetector.sauvegarder_resultats_sondage(count_pour, count_contre)


def run_vote(img):
    if VoteDetector is None: raise ImportError("Module vote manquant.")

    print("Chargement du modèle de Vote...")
    detector = VoteDetector(head_model_path=PATH_HEAD_MODEL, pose_model_path=PATH_POSE_MODEL)
    
    # Récupère la liste des résultats PAR TÊTE
    results = detector.detect(img)
    
    count_gauche = 0
    count_droite = 0
    count_abst = 0
    
    # --- COULEURS (BGR) ---
    COLOR_GAUCHE = (255, 0, 0)   # Bleu
    COLOR_DROITE = (0, 255, 0)   # Vert
    COLOR_ABST = (0, 0, 0)       # Noir
    
    for res in results:
        vote = res['vote']
        hx1, hy1, hx2, hy2 = res['head_box']
        
        # 1. Déterminer la couleur de la Tête
        head_color = COLOR_ABST
        if vote == 'G':
            head_color = COLOR_GAUCHE
            count_gauche += 1
        elif vote == 'D':
            head_color = COLOR_DROITE
            count_droite += 1
        else:
            # Neutre ou 2 mains
            count_abst += 1

        # 2. Dessiner UNIQUEMENT la boite de la tête
        cv2.rectangle(img, (hx1, hy1), (hx2, hy2), head_color, 2)

    # 3. Affichage des Totaux
    total_heads = len(results)
    
    if total_heads > 0:
        perc_left = (count_gauche / total_heads) * 100
        perc_right = (count_droite / total_heads) * 100
        perc_abst = (count_abst / total_heads) * 100
    else:
        perc_left = perc_right = perc_abst = 0

    # Fond noir pour contenir les 3 lignes (hauteur 130px)
    cv2.rectangle(img, (0, 0), (400, 130), (0, 0, 0), -1)

    # Affichage des pourcentages
    cv2.putText(img, f"G: {perc_left:.1f}% ({count_gauche})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GAUCHE, 2)
    
    cv2.putText(img, f"D: {perc_right:.1f}% ({count_droite})", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_DROITE, 2)
    
    # Utilisation de gris clair pour l'abstention car COLOR_ABST est noir (invisible sur fond noir)
    cv2.putText(img, f"Abstention: {perc_abst:.1f}% ({count_abst})", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Affichage final console
    print(f"Têtes YOLO détectées: {total_heads}")
    print(f"Mains Gauche: {count_gauche}")
    print(f"Mains Droite: {count_droite}")
    print(f"Abstention: {count_abst}")

    cv2.imshow("Resultat Vote", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    VoteDetector.sauvegarder_vote_txt(results)
# =============================================================================
# INTERFACE & HISTORIQUE
# =============================================================================
def afficher_historique():
    root.withdraw()
    hist_window = tk.Toplevel(root)
    hist_window.title("Historique")
    hist_window.geometry("500x400")
    
    def fermer():
        hist_window.destroy()
        root.deiconify()
    hist_window.protocol("WM_DELETE_WINDOW", fermer)

    tk.Label(hist_window, text="Historique des relevés", font=("Helvetica", 16, "bold")).pack(pady=15)
    
    frame_list = tk.Frame(hist_window)
    frame_list.pack(fill="both", expand=True, padx=20, pady=5)
    
    scrollbar = tk.Scrollbar(frame_list)
    scrollbar.pack(side="right", fill="y")
    
    listbox = tk.Listbox(frame_list, yscrollcommand=scrollbar.set, font=("Arial", 11))
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)

    if getattr(sys, 'frozen', False): app_path = os.path.dirname(sys.executable)
    else: app_path = os.path.dirname(os.path.abspath(__file__))
    fichier = os.path.join(app_path, "historique.txt")
    
    if os.path.exists(fichier):
        with open(fichier, "r", encoding="utf-8") as f:
            for line in f.readlines()[::-1]:
                if line.strip(): listbox.insert("end", "  " + line.strip())
    else:
        listbox.insert("end", "Aucun historique.")

    tk.Button(hist_window, text="Retour", command=fermer, bg="#c0392b", fg="white").pack(pady=15)

def btn_cmd_comptage(): lancer_analyse("Comptage", run_comptage)
def btn_cmd_sondage(): lancer_analyse("Sondage", run_sondage)
def btn_cmd_vote(): lancer_analyse("Vote", run_vote)

root = tk.Tk()
root.title("Compteur d'Amphi")
root.geometry("400x480")
root.configure(bg="#f0f0f0")

tk.Label(root, text="Sélection du type de comptage", font=("Helvetica", 16, "bold"), bg="#f0f0f0").pack(pady=20)
btn_style = {"width": 25, "height": 2, "bg": "#000000", "fg": "white", "font": ("Arial", 12, "bold")}

tk.Button(root, text="Comptage", command=btn_cmd_comptage, **btn_style).pack(pady=10)
tk.Button(root, text="Sondage", command=btn_cmd_sondage, **btn_style).pack(pady=10)
tk.Button(root, text="Vote", command=btn_cmd_vote, **btn_style).pack(pady=10)
tk.Button(root, text="Historique", command=afficher_historique, **btn_style).pack(pady=10)
tk.Button(root, text="Quitter", command=root.quit, width=15, bg="#c0392b", fg="white").pack(pady=20)

if __name__ == "__main__":
    root.mainloop()