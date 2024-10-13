import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Charger les données depuis un fichier CSV
df = pd.read_csv('C:\\Users\\Administrator\\PycharmProjects\\SSSSSSSSSSSSSSSS\\SML.csv')

# Prétraitement des données textuelles
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Commentaire'])
sequences = tokenizer.texts_to_sequences(df['Commentaire'])
padded_sequences = pad_sequences(sequences)


# Préparation des étiquettes de sortie
encoder = OneHotEncoder(sparse=False)
sentiment_labels_onehot = encoder.fit_transform(df['Sentiment'].values.reshape(-1, 1))

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, sentiment_labels_onehot, test_size=0.5, random_state=42)

# Création et compilation du modèle
model = Sequential([
    Embedding(len(tokenizer.word_index) + 1, 128, input_length=padded_sequences.shape[1]),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(sentiment_labels_onehot.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))

# Évaluation du modèle sur les données de test
loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

# Dictionnaire de correspondance entre les indices et les étiquettes
index_to_sentiment = {i: label.split('_')[1] for i, label in enumerate(encoder.get_feature_names_out())}

# Boucle pour permettre à l'utilisateur de créer des commentaires
while True:
    user_comment = input("Ajoutez votre commentaire (ou tapez 'exit' pour quitter) : ")

    if user_comment.lower() == 'exit':
        break

    user_sequence = tokenizer.texts_to_sequences([user_comment])
    user_padded_sequence = pad_sequences(user_sequence, maxlen=padded_sequences.shape[1])

    # Prédire le sentiment du commentaire de l'utilisateur
    user_prediction = model.predict(user_padded_sequence)
    predicted_sentiment_index = np.argmax(user_prediction)

    # Afficher le résultat
    print(f"Votre commentaire : {user_comment}")
    print(f"Sentiment prédit : {index_to_sentiment[predicted_sentiment_index]}\n")
