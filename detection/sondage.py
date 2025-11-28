from detection.vote import HandDetector
from datetime import datetime
import os

class SondageDetector(HandDetector):
    """
    Détecteur spécialisé sondage :
    - 'D'  = POUR  (main droite)
    - 'G'  = POUR  (main gauche)
    - 'None' = CONTRE (aucune main levée)
    """
    def detect_sondage(self, img):
        raw = super().detect(img)
        
        results = []
        for r in raw:
            vote = r['vote']

            if vote == 'D':
                r['sondage'] = 'POUR'
            elif vote == 'G':
                r['sondage'] = 'POUR'
            else:
                r['sondage'] = 'CONTRE'
            
            results.append(r)
        
        return results

    @staticmethod
    def sauvegarder_resultats_sondage(pour, contre, fichier="historique.txt"):
        #Enregistre les résultats du SONDAGE.
        if not os.path.exists(fichier):
            with open(fichier, "w", encoding="utf-8") as f:
                f.write("")

        total = pour + contre
        if total > 0:
            p_p = (pour / total) * 100
            p_c = (contre / total) * 100
        else:
            p_p = p_c = 0

        date_heure = datetime.now().strftime("%d/%m/%y à %Hh%M")
        ligne = f"[S]: {date_heure}, {p_p:.1f}% des étudiants sont pour ({pour} pour et {contre} contre)\n"

        with open(fichier, "a", encoding="utf-8") as f:
            f.write(ligne)
        
        print(f"Sauvegarde effectuée dans {fichier}")