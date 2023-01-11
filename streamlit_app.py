# librería de la webapp
import streamlit as st

# librerías de python
import os
import re

# librerías de procesamiento de datos
import keras
import matplotlib.pyplot as plt
import nltk
import numpy as np
import tensorflow_hub as hub

from keras import backend as K
from nltk import WordPunctTokenizer
from nltk.corpus import stopwords
from unidecode import unidecode
from wordcloud import WordCloud

# datos persistentes entre sesiones de streamlit
if 'embedd128' not in st.session_state:
    # tokens entrenados a partir de Google News en Español
    st.session_state['embedd128'] = hub.load(
        "https://tfhub.dev/google/tf2-preview/nnlm-es-dim128-with-normalization/1"
    )

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

WPT = WordPunctTokenizer()
wc = WordCloud()

# Diccionarios para evaluar las predicciones de los modelos
polaridad = {0: 'Positivos', 1: 'Negativos'}
emocion = {0: 'Alegria', 1: 'Ira', 2: 'Miedo', 3: 'Tristeza'}


def text_to_matrix(input):
    """Genera las entradas necesarias para los modelos entrenados"""
    string = str(input)

    # sanitizar entrada
    sanitized = re.sub(r'https?://\S+|www\.\S+', '', string)  # quitar URLs
    sanitized = re.sub(r'@\S+', '', sanitized)  # quitar menciones @
    sanitized = re.sub(r'#\S+', '', sanitized)  # quitar texto de hashtag

    # convertir a ASCII (eliminar diacríticos y otros caracteres especiales)
    sanitized = unidecode(string.lower())

    # tokenizar
    tokens = WPT.tokenize(sanitized)

    # eliminar datos redundates de los tokens (stopwords y símbolos):
    limpio = tokens[:]
    for tokens in tokens:
        # eliminar stopwords
        if tokens in stopwords.words('spanish'):
            limpio.remove(tokens)

        # eliminar simbolos
        if tokens in "::\/.,';]:#[\=-¿?><"":}{+_)(*&^%$@°|¬¡!~¨":
            limpio.remove(tokens)  # Lista con Tokens

    z = np.zeros((len(limpio), 128))  # Pre padding

    # obtener datos previamente guardados
    embedd128 = st.session_state.embedd128

    for i, token in enumerate(limpio):
        temp = embedd128([token]).numpy()
        z[i][:] = temp[0][:]  # embed() funcion de vectorizacion

    z = np.dot(np.transpose(z), z)
    z = np.expand_dims(z, axis=(0, 1))

    return z


def text_to_wordcloud(inp):
    """
    Prepara el texto para wordcloud
    """

    # Tokenizar
    toks = WPT.tokenize(inp)
    limpio = toks[:]
    # Limpiar los tokens:
    for tokens in toks:
        if tokens in stopwords.words('spanish'):  # Eliminar stopwords
            limpio.remove(tokens)
        # eliminar simbolos
        if tokens in "::\/.,';]:#[\=-¿?><"":}{+_)(*&^%$@°|¬¡!~¨":
            limpio.remove(tokens)  # Lista con Tokens
    union = ' '.join(limpio)
    return union


@st.cache(allow_output_mutation=True)
def get_model(path):
    model = keras.models.load_model(path, compile=False)
    return model


def determine_polarity(input_text, model):
    # Cargamos los modelos entrenados:
    mt = text_to_matrix(input_text)  # Respresentación numérica del texto
    results = model.predict(mt)
    a1 = np.argmax(results)
    re1 = polaridad[a1]
    return re1


def main():
    st.title("Hola soy Psibot")

    text_input = st.text_area('Escribe lo que me quieras platicar:',
                              on_change=None, placeholder='Exprésate aquí')

    model_pol = get_model('Modelos/CNNpol128in2.h5')
    model_4e = get_model('Modelos/Modelo_E4t128_in4.h5')

    if st.button('¡Platícame!'):
        if text_input == '':
            st.write('Escribe en el espacio de arriba lo que quieras platicarme.')

        else:
            # Sentimientos
            st.write('Polaridad inferida:')
            output_matrix = text_to_matrix(text_input)

            # NOTA: para que el modelo pueda ejecutarse bajo un CPU,
            # se necesita instalar una versión de tensorflow y keras compatible
            # (ej. intel-tensorflow)
            predict = model_pol.predict(output_matrix)
            pclass = np.argmax(predict, axis=1)[0]

            confidence = np.floor(10000 * predict[0][pclass])/100

            st.write(
                f'{polaridad[pclass]} (confidencia del {confidence}%)'
            )

            # Emociones
            st.write('Emoción inferida:')
            output_matrix = text_to_matrix(text_input)

            # NOTA: para que el modelo pueda ejecutarse bajo un CPU,
            # se necesita instalar una versión de tensorflow y keras compatible
            # (ej. intel-tensorflow)
            predict = model_4e.predict(output_matrix)
            eclass = np.argmax(predict, axis=1)[0]

            confidence = np.floor(10000 * predict[0][eclass])/100

            st.write(
                f'{emocion[eclass]} (confidencia del {confidence}%)'
            )

            # st.write(output_matrix.shape, type(output_matrix), type(model))


if __name__ == '__main__':
    main()
