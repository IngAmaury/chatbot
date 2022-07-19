#lIBRERIA DE LAWEB APP:
import streamlit as st
##Cargamos las librerias necesarias para trabajar los datos
import re
import nltk
nltk.download('punkt')
from nltk import WordPunctTokenizer
WPT=WordPunctTokenizer()
from nltk.stem import SnowballStemmer
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import os
from keras.models import load_model
#import io
##Embedding tf2-preview-nnlm https://tfhub.dev/google/collections/tf2-preview-nnlm/1
#https://tfhub.dev/google/collections/nnlm/1
embedd128 = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-es-dim128-with-normalization/1")
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
wc = WordCloud()
### Genera las entradas necesarias para los modelos entrenados
def tx2m(inpt):
  string=str(inpt)
  string = re.sub(r'https?://\S+|www\.\S+', '', string) #Quitar URLS
  string = re.sub(r'@\S+\S+','',string) #Quitar menciones @
  inp = re.sub(r'#\S+\S+','',string) #Quitar texto de hashtag
  s=inp.lower()##a minusculas
  ##Quitar caracteres especiales
  a,b = 'áäéëíïóöúü','aaeeiioouu'
  trans = str.maketrans(a,b)
  frs=s.translate(trans)
  ##Tokenizar
  toks=WPT.tokenize(frs)
  limpio=toks[:]
  ##Limpiar los tokens:
  for tokens in toks:
    if tokens in stopwords.words('spanish'):#Eliminar stopwords
      limpio.remove(tokens)
    if tokens in "::\/.,';]:#[\=-¿?><"":}{+_)(*&^%$@°|¬¡!~¨":#eliminar simbolos
      limpio.remove(tokens)##Lista con Tokens
  z=np.zeros((len(limpio),128))##Pre padding
  for k in range(len(limpio)):
    temp=embedd128([limpio[k]]).numpy()
    z[k][:]=temp[0][:]##embed() funcion de vectorizacion
  z=np.dot(np.transpose(z),z)
  res = np.expand_dims(z, axis=(0,1))
  return res
# Preaara el texto para la funcion de nubes de palabras
def txt2WC(inp):
  ##Tokenizar
  toks=WPT.tokenize(inp)
  limpio=toks[:]
  ##Limpiar los tokens:
  for tokens in toks:
    if tokens in stopwords.words('spanish'):#Eliminar stopwords
      limpio.remove(tokens)
    if tokens in "::\/.,';]:#[\=-¿?><"":}{+_)(*&^%$@°|¬¡!~¨":#eliminar simbolos
      limpio.remove(tokens)##Lista con Tokens
  union=' '.join(limpio)
  return union
##Cargamos los modelos
#modelBin = tf.keras.models.load_model('')
#modelAS6 = tf.keras.models.load_model('protomodeloAS6p1.h5')
model_path = os.path.join('Modelos/CNNpol128in2.h5')
def model_load():
    model = load_model(model_path)
    return model
##Diccionarios para evaluar las predicciones de los modelos
polaridad={0:'Positivas',1:'Negativas'}
emocion={0:'Alegria',1:'Sorpresa',2:'Tristeza',3:'Miedo',4:'Ira',5:'Disguto'}
def POL(input_text):
  ####Cargamos los modelos entrenados:
  mt=tx2m(input_text) ###Respresentación numérica del texto
  results=model.predict(mt)
  #result2=modelAS6.predict(mt)
  a1=argmax(results)
  #a2=sum(result2)
  re1=polaridad[a1]
  #re2=emocion[np.where(a2 == np.amax(a2))[0][0]]
  #v=txtWC(input_text) ###Texto original procesado para la nube de palabras
  #wc_result=wc.generate(v) ## Variable para almacenar la nube de palabras
  #plt.axis("off")
  #plt.imshow(wc_result, interpolation='bilinear')
  #S=wc_result
  #plt.show()
  return re1
############__________WEBAPP_______###################
st.title("Hola soy Psibot")
txtInput = st.text_area('Escribe lo que me quieras contar',on_change=None, placeholder='Expresate aquí')
model = model_load()
if st.button('Contar'):
  if txtInput=='':
    st.write('Escribe en el espacio de arriba para contarme algo')
  else:
     st.write('Sentimentos:')
     outTXT = np.argmax(model.predict(txtInput))
     st.write(outTxt)
