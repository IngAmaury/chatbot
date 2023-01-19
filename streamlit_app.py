# librerías de python
import os
import re

# librerías de procesamiento de datos
import keras
import matplotlib.pyplot as plt
import nltk
import numpy as np
import streamlit as st  # librería de la webapp
from nltk import WordPunctTokenizer
from nltk.corpus import stopwords
from streamlit_server_state import server_state, server_state_lock
from unidecode import unidecode


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
    tokens = WordPunctTokenizer().tokenize(sanitized)

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
    with server_state_lock.embedd128:
        embedd128 = server_state.embedd128

    for i, token in enumerate(limpio):
        temp = embedd128([token]).numpy()
        z[i][:] = temp[0][:]  # embed() funcion de vectorizacion

    z = np.dot(np.transpose(z), z)
    z = np.expand_dims(z, axis=(0, 1))

    return z


@st.cache
def text_to_wordcloud(inp):
    """
    Prepara el texto para wordcloud
    """

    # Tokenizar
    toks = WordPunctTokenizer().tokenize(inp)
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
            # Diccionarios para evaluar las predicciones de los modelos
            polaridad = {0: 'Positivos', 1: 'Negativos'}
            emocion = {0: 'Alegria', 1: 'Ira', 2: 'Miedo', 3: 'Tristeza'}

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

            # Nube de palabras
            from wordcloud import WordCloud

            word_cloud = WordCloud().generate(text_to_wordcloud(text_input))
            plt.style.use('dark_background')
            plt.axis('off')
            plt.imshow(word_cloud, interpolation='bicubic')

            # guardar imagen, mostrarla y borrarla despues de su uso.
            plt.savefig('x', dpi=800)
            st.image('x.png')
            os.remove('x.png')


if __name__ == '__main__':
    # tokens entrenados a partir de Google News en Español
    tmurl = "https://tfhub.dev/google/tf2-preview/nnlm-es-dim128-with-normalization/1"

    # datos persistentes entre sesiones de streamlit
    with server_state_lock['embedd128']:
        if 'embedd128' not in server_state:
            import tensorflow_hub as hub
            server_state.embedd128 = hub.load(tmurl)

    try:
        nltk.data.find('stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    main()  # inicializar aplicación
