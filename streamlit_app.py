# librería de la webapp
import streamlit as st

# libreriasde procesamiento de datos
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import re
import tensorflow_hub as hub

from keras.models import load_model
from nltk import WordPunctTokenizer
from nltk.corpus import stopwords
from unidecode import unidecode
from wordcloud import WordCloud

# inicializar datos
nltk.download('stopwords')
WPT = WordPunctTokenizer()

# tokens entrenados a partir de Google News en Español
embedd128 = hub.load(
    "https://tfhub.dev/google/tf2-preview/nnlm-es-dim128-with-normalization/1"
)

wc = WordCloud()

# Diccionarios para evaluar las predicciones de los modelos
polaridad = {0: 'Positivas', 1: 'Negativas'}
emocion = {0: 'Alegria', 1: 'Sorpresa',
           2: 'Tristeza', 3: 'Miedo', 4: 'Ira', 5: 'Disguto'}

# ubicación del modelo
model_path = os.path.join('Modelos/CNNpol128in2.h5')


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


def get_model():
    return load_model(model_path)


def determine_polarity(input_text, model):
    # Cargamos los modelos entrenados:
    mt = text_to_matrix(input_text)  # Respresentación numérica del texto
    results = model.predict(mt)
    # result2=modelAS6.predict(mt)
    a1 = np.argmax(results)
    # a2=sum(result2)
    re1 = polaridad[a1]
    #re2=emocion[np.where(a2 == np.amax(a2))[0][0]]
    # v=txtWC(input_text) ###Texto original procesado para la nube de palabras
    # wc_result=wc.generate(v) ## Variable para almacenar la nube de palabras
    # plt.axis("off")
    #plt.imshow(wc_result, interpolation='bilinear')
    # S=wc_result
    # plt.show()
    return re1


def main():
    st.title("Hola soy Psibot")

    text_input = st.text_area('Escribe lo que me quieras contar',
                              on_change=None, placeholder='Exprésate aquí')
    model = get_model()

    if st.button('Contar'):
        if text_input == '':
            st.write('Escribe en el espacio de arriba para contarme algo')

        else:
            st.write('Sentimentos:')
            output_matrix = text_to_matrix(text_input)

            # NOTA: para que el modelo pueda ejecutarse bajo un CPU,
            # se necesita instalar una versión de tensorflow y keras compatible
            # (ej. intel-tensorflow)
            predict = model.predict(output_matrix)
            pclass = np.argmax(predict, axis=1)[0]

            confidence = np.floor(10000 * predict[0][pclass])/100

            st.write(
                f'{polaridad[pclass]} (confidencia del {confidence}%)'
            )

            # st.write(output_matrix.shape, type(output_matrix), type(model))


if __name__ == '__main__':
    main()
