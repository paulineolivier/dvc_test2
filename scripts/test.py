from joblib import load
import json
from sklearn.metrics import accuracy_score
import pandas as ps
from pathlib import Path

if __name__ == "__main__":
	
	# on récupère les données
	test = ps.read_csv("data/prepared/test.csv")

	# split features/traget
	y_test = test.Survived
	X_test = test.drop('Survived', axis=1)
	
	# on charge le modèle
	model = load("model/model.joblib")

	# on fait les prédictions sur le test
	predictions = model.predict(X_test)

	# on calcul l'accuracy et on la stocke dans un fichier json
	accuracy = accuracy_score(y_test, predictions)
	print('Accuracy :', accuracy)
	metrics = {"accuracy": accuracy}
	accuracy_path = Path("metrics/accuracy.json")
	accuracy_path.write_text(json.dumps(metrics))