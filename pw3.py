import streamlit as st
import visualisation

# Charger les données
data = visualisation.data

# Afficher les 5 premières lignes de données
st.write(data.head(5))

# Construire le corpus
corpus = visualisation.build_corpus(data)

# Afficher les 2 premiers éléments du corpus
st.write(corpus[0:2])

# Construire le modèle word2vec
model = visualisation.model

# Afficher le vecteur pour 'trump'
st.write(model.wv['trump'])

# Construire le vocabulaire
vocab = visualisation.vocab

# Afficher le plot TSNE
fig = visualisation.tsne_plot(model)
st.pyplot(fig)
