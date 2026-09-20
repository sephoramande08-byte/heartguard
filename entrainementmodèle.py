import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DATA_PATH = "heart.csv"          # Chemin vers le dataset téléchargé
MODEL_OUTPUT = "modele_risque_cardiaque.pkl"
SCALER_OUTPUT = "scaler.pkl"

# Colonnes numériques à normaliser (les autres sont déjà catégorielles/binaires)
COLONNES_NUMERIQUES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Noms lisibles des variables, pour affichage à l'utilisateur final
NOMS_LISIBLES = {
    "age": "âge",
    "sex": "sexe",
    "cp": "type de douleur thoracique",
    "trestbps": "tension artérielle au repos",
    "chol": "cholestérol",
    "fbs": "glycémie à jeun élevée",
    "restecg": "résultats ECG au repos",
    "thalach": "fréquence cardiaque maximale",
    "exang": "angine induite par l'effort",
    "oldpeak": "dépression ST à l'effort",
    "slope": "pente du segment ST",
    "ca": "nombre de vaisseaux colorés",
    "thal": "thalassémie",
}


# ---------------------------------------------------------------------------
# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES
# ---------------------------------------------------------------------------

def charger_et_nettoyer_donnees(path):
    """Charge le CSV et traite les valeurs manquantes."""
    df = pd.read_csv(path)
    print(f"Dataset chargé : {df.shape[0]} patients, {df.shape[1]} colonnes")

    valeurs_manquantes = df.isnull().sum()
    if valeurs_manquantes.sum() > 0:
        print("\nValeurs manquantes détectées :")
        print(valeurs_manquantes[valeurs_manquantes > 0])
        # Remplacement par la médiane (plus robuste que la moyenne aux valeurs extrêmes)
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        print("→ Valeurs manquantes remplacées par la médiane de chaque colonne.")
    else:
        print("Aucune valeur manquante détectée.")

    return df


# ---------------------------------------------------------------------------
# 2. ENTRAÎNEMENT ET COMPARAISON DES MODÈLES
# ---------------------------------------------------------------------------

def comparer_modeles(X_train, X_test, y_train, y_test):
    """Entraîne plusieurs modèles et affiche leurs performances comparées."""
    modeles = {
        "Régression Logistique": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    resultats = {}
    print("\n" + "=" * 60)
    print("COMPARAISON DES MODÈLES")
    print("=" * 60)

    for nom, modele in modeles.items():
        modele.fit(X_train, y_train)
        y_pred = modele.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        resultats[nom] = {
            "modele": modele,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        print(f"\n--- {nom} ---")
        print(f"Accuracy  : {accuracy:.3f}")
        print(f"Précision : {precision:.3f}")
        print(f"Recall    : {recall:.3f}  (important : peu de malades non détectés)")
        print(f"F1-score  : {f1:.3f}")
        print("Matrice de confusion :")
        print(confusion_matrix(y_test, y_pred))

    return resultats


def choisir_meilleur_modele(resultats):
    """Sélectionne le modèle avec le meilleur recall (priorité médicale : ne pas rater un malade)."""
    meilleur_nom = max(resultats, key=lambda k: resultats[k]["recall"])
    print(f"\n>>> Modèle retenu : {meilleur_nom} (meilleur recall)")
    return meilleur_nom, resultats[meilleur_nom]["modele"]


# ---------------------------------------------------------------------------
# 3. FACTEURS DE RISQUE DOMINANTS (Option 1 retenue)
# ---------------------------------------------------------------------------

def obtenir_facteurs_dominants(modele, colonnes, top_n=3):
    """
    Retourne les variables qui pèsent le plus dans les décisions du modèle,
    globalement (utile pour le rapport). Fonctionne avec Random Forest
    (feature_importances_) ou Régression Logistique (coef_).
    """
    if hasattr(modele, "feature_importances_"):
        importances = modele.feature_importances_
    elif hasattr(modele, "coef_"):
        importances = np.abs(modele.coef_[0])
    else:
        return []

    classement = sorted(zip(colonnes, importances), key=lambda x: x[1], reverse=True)
    return classement[:top_n]


def facteurs_dominants_par_patient(modele, colonnes, donnees_patient, moyennes, top_n=3):
    """
    Identifie, POUR UN PATIENT DONNÉ, quelles variables s'écartent le plus
    de la moyenne du dataset dans le sens "à risque", pondérées par
    l'importance globale du modèle. Sert à personnaliser le message renvoyé
    par l'API (champ "facteurs_dominants" du JSON de prédiction).
    """
    if hasattr(modele, "feature_importances_"):
        importances = modele.feature_importances_
    elif hasattr(modele, "coef_"):
        importances = np.abs(modele.coef_[0])
    else:
        return []

    scores = []
    for i, col in enumerate(colonnes):
        ecart = (donnees_patient[i] - moyennes[col]) / (moyennes[col] + 1e-6)
        score_influence = abs(ecart) * importances[i]
        if donnees_patient[i] > moyennes[col]:  # variable au-dessus de la moyenne
            scores.append((col, score_influence))

    scores.sort(key=lambda x: x[1], reverse=True)
    noms = [NOMS_LISIBLES.get(col, col) for col, _ in scores[:top_n]]
    return noms


# ---------------------------------------------------------------------------
# 4. CLASSIFICATION DU NIVEAU DE RISQUE + RECOMMANDATION
# ---------------------------------------------------------------------------

RAPPEL_MEDICAL = (
    "Cette estimation est basée sur un modèle statistique et ne constitue "
    "en aucun cas un diagnostic médical. Consultez un professionnel de "
    "santé pour un avis médical fiable."
)


def interpreter_score(score_risque):
    """Transforme une probabilité (0-1) en niveau de risque + recommandation."""
    if score_risque < 0.33:
        niveau = "faible"
        recommandation = (
            "Votre profil ne présente pas de signe de risque majeur. "
            "Continuez à maintenir de bonnes habitudes de vie."
        )
    elif score_risque < 0.66:
        niveau = "modéré"
        recommandation = (
            "Certains facteurs méritent votre attention. Nous vous "
            "conseillons d'en parler à votre médecin lors d'une prochaine visite."
        )
    else:
        niveau = "élevé"
        recommandation = (
            "Plusieurs facteurs de risque importants ont été détectés. "
            "Nous vous recommandons de consulter un cardiologue rapidement."
        )

    return {
        "niveau_risque": niveau,
        "recommandation": recommandation,
        "avertissement": RAPPEL_MEDICAL,
    }


def construire_reponse_prediction(modele, colonnes, donnees_patient, moyennes):
    """
    Construit le JSON complet tel que renvoyé par l'API à l'application mobile.
    `donnees_patient` : liste de valeurs dans le même ordre que `colonnes`.
    """
    proba = modele.predict_proba(pd.DataFrame([donnees_patient], columns=colonnes))[0][1]  # proba classe "malade"
    interpretation = interpreter_score(proba)
    facteurs = facteurs_dominants_par_patient(modele, colonnes, donnees_patient, moyennes, top_n=13)

    return {
        "score_risque": round(float(proba), 2),
        "niveau_risque": interpretation["niveau_risque"],
        "recommandation": interpretation["recommandation"],
        "facteurs_dominants": facteurs,
        "avertissement": interpretation["avertissement"],
    }


# ---------------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    # 1. Chargement et nettoyage
    df = charger_et_nettoyer_donnees(DATA_PATH)

    X = df.drop(columns=["target"])
    y = df["target"]
    colonnes = list(X.columns)

    # 2. Normalisation des variables numériques (utile surtout pour la régression logistique)
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[COLONNES_NUMERIQUES] = scaler.fit_transform(X[COLONNES_NUMERIQUES])

    # 3. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Comparaison des modèles
    resultats = comparer_modeles(X_train, X_test, y_train, y_test)
    nom_meilleur, meilleur_modele = choisir_meilleur_modele(resultats)

    # 5. Facteurs de risque dominants (vue globale, pour le rapport)
    print("\n" + "=" * 60)
    print(f"FACTEURS DE RISQUE LES PLUS INFLUENTS ({nom_meilleur})")
    print("=" * 60)
    for col, importance in obtenir_facteurs_dominants(meilleur_modele, colonnes, top_n=5):
        print(f"  {NOMS_LISIBLES.get(col, col):<35} {importance:.3f}")

    # 6. Ré-entraînement final sur 100% des données (pour maximiser l'apprentissage)
    meilleur_modele.fit(X_scaled, y)

    # 7. Sauvegarde du modèle + du scaler + des moyennes (nécessaires pour l'API)
    joblib.dump(meilleur_modele, MODEL_OUTPUT)
    joblib.dump(scaler, SCALER_OUTPUT)
    joblib.dump(X.mean().to_dict(), "moyennes_dataset.pkl")
    joblib.dump(colonnes, "colonnes_modele.pkl")

    print(f"\nModèle sauvegardé : {MODEL_OUTPUT}")
    print(f"Scaler sauvegardé : {SCALER_OUTPUT}")
    print("Fichiers prêts à être utilisés par l'API FastAPI.")

    # 8. Exemple de prédiction complète (comme le ferait l'API)
    print("\n" + "=" * 60)
    print("EXEMPLE DE RÉPONSE API POUR UN PATIENT")
    print("=" * 60)
    exemple_patient = X_scaled.iloc[0].tolist()
    moyennes = X.mean().to_dict()
    reponse = construire_reponse_prediction(meilleur_modele, colonnes, exemple_patient, moyennes)
    import json
    print(json.dumps(reponse, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()