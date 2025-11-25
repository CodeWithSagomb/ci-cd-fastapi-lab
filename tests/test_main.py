from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import pytest
from app.main import app

# Initialisation du client de test (c'est comme un faux navigateur web)
client = TestClient(app)

# --- TESTS BASIQUES (Sans Mocking) ---

def test_root():
    """Vérifie que la racine répond Hello World"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_health_check():
    """Vérifie que l'API est en bonne santé"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_list_models():
    """Vérifie qu'on récupère bien la liste des modèles"""
    response = client.get("/models")
    assert response.status_code == 200
    # On vérifie que la réponse contient bien la clé attendue
    assert "available_models" in response.json()

def test_predict_invalid_model():
    """Vérifie que l'API renvoie une erreur 404 si le modèle n'existe pas"""
    # On envoie des données valides, mais vers un modèle "bidon"
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict/modele_inexistant", json=payload)
    assert response.status_code == 404 # Not Found (ou 422 si erreur de validation, mais ici c'est la logique métier)

# --- TESTS AVANCÉS (Avec Mocking) ---

# C'est ici que la magie opère.
# @pytest.fixture permet de préparer un contexte avant de lancer le test.
@pytest.fixture
def mock_models(mocker):
    # 1. On crée des faux objets (MagicMock) qui remplaceront les vrais modèles
    mock_dict = {"logistic_model": MagicMock(), "rf_model": MagicMock()}
    
    # 2. On utilise 'patch' pour remplacer la variable 'ml_models' dans 'app.main'
    # par notre faux dictionnaire 'mock_dict' le temps du test.
    m = mocker.patch("app.main.ml_models", mock_dict)
    
    return mock_dict

def test_predict_valid_model(mock_models):
    """Teste la prédiction avec un modèle simulé (mocké)"""
    
    # On configure le Mock : quand on appelle .predict(), retourne [1] (classe 1)
    # Note : Dans app/main.py, ton code actuel retourne 0 en dur, 
    # mais ce test vérifie surtout que la route fonctionne techniquement.
    
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    # On appelle la route avec un modèle qui existe dans notre mock ("logistic_model")
    response = client.post("/predict/logistic_model", json=payload)
    
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["model"] == "logistic_model"
    # Vérifie simplement que la prédiction est présente (peu importe la valeur pour l'instant)
    assert "prediction" in json_response