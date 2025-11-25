from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Simulation de chargement de modèles (dictionnaire simple pour l'exemple)
# Dans la vraie vie, on chargerait des fichiers .pkl ici
ml_models = {
    "logistic_model": "loaded_logistic_model_object",
    "rf_model": "loaded_rf_model_object"
}

app = FastAPI()

# Définition du schéma de données d'entrée (Iris dataset simulation)
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/models")
def list_models():
    return {"available_models": list(ml_models.keys())}

@app.post("/predict/{model_name}")
def predict(model_name: str, input_data: IrisInput):
    if model_name not in ml_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Simulation de logique de prédiction
    # Note: Dans le test, nous allons "mocker" (simuler) cette partie
    # pour ne pas avoir besoin de vrais modèles Scikit-Learn chargés.
    model = ml_models[model_name]
    
    # Ici, on ferait normalement: prediction = model.predict(...)
    # Pour l'instant, on retourne une fausse prédiction
    prediction = 0 
    
    return {"model": model_name, "prediction": prediction}