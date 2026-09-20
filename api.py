"""
API - Prédiction du Risque de Maladie Cardiaque (version simplifiée pour intégration)
========================================================================================
IUT de Douala - GI1 - Projet cardio_ml

Pour lancer localement :
    uvicorn api:app --reload

Pour déployer (Render, PythonAnywhere...), voir requirements.txt fourni à côté.
"""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. CHARGEMENT DU MODÈLE ET DES FICHIERS ASSOCIÉS (une seule fois, au démarrage)
# ---------------------------------------------------------------------------

try:
    modele = joblib.load("modele_risque_cardiaque.pkl")
    scaler = joblib.load("scaler.pkl")
    moyennes = joblib.load("moyennes_dataset.pkl")
    colonnes = joblib.load("colonnes_modele.pkl")
except FileNotFoundError as erreur:
    raise RuntimeError(
        "Fichiers du modèle introuvables. entrainementmodele.py doit avoir "
        "été lancé une fois, et les 4 fichiers .pkl doivent être dans ce dossier."
    ) from erreur

COLONNES_NUMERIQUES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

NOMS_LISIBLES = {
    "age": "âge", "sex": "sexe", "cp": "type de douleur thoracique",
    "trestbps": "tension artérielle au repos", "chol": "cholestérol",
    "fbs": "glycémie à jeun élevée", "restecg": "résultats ECG au repos",
    "thalach": "fréquence cardiaque maximale", "exang": "angine induite par l'effort",
    "oldpeak": "dépression ST à l'effort", "slope": "pente du segment ST",
    "ca": "nombre de vaisseaux colorés", "thal": "thalassémie",
}

RAPPEL_MEDICAL = (
    "Cette estimation est basée sur un modèle statistique et ne constitue "
    "en aucun cas un diagnostic médical. Consultez un professionnel de "
    "santé pour un avis médical fiable."
)

# ---------------------------------------------------------------------------
# 2. CRÉATION DE L'APPLICATION FASTAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="API - Prédiction du Risque de Maladie Cardiaque",
    description="Projet académique IUT de Douala - GI1",
    version="1.0.0",
)

# Autorise le site React (Lovable) à appeler cette API depuis le navigateur.
# "*" = tout le monde peut appeler l'API. Pour un rendu académique demain,
# c'est le plus simple ; à restreindre plus tard à l'URL exacte du site
# (ex: https://vital-beat-guide.lovable.app) si vous voulez sécuriser après.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 3. DONNÉES ATTENDUES EN ENTRÉE
# ---------------------------------------------------------------------------

class DonneesPatient(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1, description="0 = femme, 1 = homme")
    cp: int = Field(..., ge=0, le=3)
    trestbps: float
    chol: float
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 55, "sex": 1, "cp": 2, "trestbps": 140, "chol": 230,
                "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
                "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2,
            }
        }


# ---------------------------------------------------------------------------
# 4. LOGIQUE DE PRÉDICTION (identique à ce qu'on a déjà testé ensemble)
# ---------------------------------------------------------------------------

def interpreter_score(score_risque: float) -> dict:
    if score_risque < 0.33:
        return {
            "niveau_risque": "faible",
            "recommandation": "Votre profil ne présente pas de signe de risque majeur. Continuez à maintenir de bonnes habitudes de vie.",
        }
    elif score_risque < 0.66:
        return {
            "niveau_risque": "modéré",
            "recommandation": "Certains facteurs méritent votre attention. Nous vous conseillons d'en parler à votre médecin lors d'une prochaine visite.",
        }
    else:
        return {
            "niveau_risque": "élevé",
            "recommandation": "Plusieurs facteurs de risque importants ont été détectés. Nous vous recommandons de consulter un cardiologue rapidement.",
        }


def facteurs_dominants_par_patient(donnees_patient: list, top_n: int = 13) -> list:
    if hasattr(modele, "feature_importances_"):
        importances = modele.feature_importances_
    elif hasattr(modele, "coef_"):
        importances = abs(modele.coef_[0])
    else:
        return []

    scores = []
    for i, col in enumerate(colonnes):
        moyenne_col = moyennes[col]
        ecart = (donnees_patient[i] - moyenne_col) / (moyenne_col + 1e-6)
        score_influence = abs(ecart) * importances[i]
        if donnees_patient[i] > moyenne_col:
            scores.append((col, score_influence))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [NOMS_LISIBLES.get(col, col) for col, _ in scores[:top_n]]


# ---------------------------------------------------------------------------
# 5. ROUTES
# ---------------------------------------------------------------------------

@app.get("/")
def accueil():
    """Route de vérification : permet de savoir si l'API tourne."""
    return {"message": "API de prédiction du risque cardiaque en ligne."}


@app.post("/predict")
def predict(patient: DonneesPatient):
    """
    Reçoit les données de santé d'un patient et renvoie le score de risque,
    le niveau, une recommandation et les facteurs dominants.

    Exemple d'appel depuis le frontend React (fetch) :

        const reponse = await fetch("https://VOTRE-URL-RENDER/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                age: 55, sex: 1, cp: 2, trestbps: 140, chol: 230,
                fbs: 0, restecg: 1, thalach: 150, exang: 0,
                oldpeak: 1.5, slope: 1, ca: 0, thal: 2
            })
        });
        const resultat = await reponse.json();
    """
    donnees_dict = patient.model_dump()
    donnees_liste = [donnees_dict[col] for col in colonnes]

    df_patient = pd.DataFrame([donnees_liste], columns=colonnes)
    df_patient[COLONNES_NUMERIQUES] = scaler.transform(df_patient[COLONNES_NUMERIQUES])

    try:
        proba = modele.predict_proba(df_patient)[0][1]
    except Exception as erreur:
        raise HTTPException(status_code=500, detail=f"Erreur du modèle : {erreur}")

    interpretation = interpreter_score(proba)
    facteurs = facteurs_dominants_par_patient(donnees_liste, top_n=13)

    return {
        "score_risque": round(float(proba), 2),
        "niveau_risque": interpretation["niveau_risque"],
        "recommandation": interpretation["recommandation"],
        "facteurs_dominants": facteurs,
        "avertissement": RAPPEL_MEDICAL,
    }