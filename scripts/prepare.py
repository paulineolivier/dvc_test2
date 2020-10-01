from pipelines import pipe_preprocessing
from pipelines import pipe_fe
import pandas as ps
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

	# lecture des données
	data = ps.read_csv('data/raw/data.csv')

	# préprocessing du train
	train_preprocessed = pipe_preprocessing.fit_transform(data)

	# splitting de données
	X_train, X_test = train_test_split(train_preprocessed, test_size=0.33, random_state=42)

	# feature engineering du train et du test
	X_train_fe = pipe_fe.fit_transform(X_train)
	X_test_fe = pipe_fe.transform(X_test)

	# on sauvegarde les fichiers obtenus
	print("Création du fichier de train...")
	X_train_fe.to_csv("data/prepared/train.csv", index=None)
	print("Création du fichier de test...")
	X_test_fe.to_csv("data/prepared/test.csv", index=None)
