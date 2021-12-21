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
from tensorflow import keras
import tensorflow_hub as hub
#import os
#import pandas as pd
#import io
##Embedding tf2-preview-nnlm https://tfhub.dev/google/collections/tf2-preview-nnlm/1
#https://tfhub.dev/google/collections/nnlm/1
embedd = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-de-dim128-with-normalization/1")
##Embedding de 50
#embedd = hub.load("https://tfhub.dev/google/nnlm-es-dim50-with-normalization/2")
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
wc = WordCloud()
### Genera las entradas necesarias para los modelos entrenados
def tx2m(inp):
  fr=inp.lower()##a minusculas
  ##Quitar caracteres especiales
  a,b = 'áäéëíïóöúü','aaeeiioouu'
  trans = str.maketrans(a,b)
  frs=fr.translate(trans)
  ##Tokenizar
  toks=WPT.tokenize(frs)
  limpio=toks[:]
  ##Limpiar los tokens:
  for tokens in toks:
    if tokens in stopwords.words('spanish'):#Eliminar stopwords
      limpio.remove(tokens)
    if tokens in "::\/.,';]:#[\=-¿?><"":}{+_)(*&^%$@°|¬¡!~¨":#eliminar simbolos
      limpio.remove(tokens)##Lista con Tokens
  ##Cogido extraido del blog 
  ##http://josearcosaneas.github.io/python/r/procesamiento/lenguaje/2017/01/02/procesamiento-lenguaje-natural-0.html
  Snowball_stemmer = SnowballStemmer('spanish')
  stemmers = [Snowball_stemmer.stem(lim) for lim in limpio]
  final = [stem for stem in stemmers if stem.isalpha() and len(stem) > 1] ##Resultado del stem
  ##Conversion a matriz
  #Analisis de la lonitud del texto
  long=len(final)
  n=1+(long//48)
  z=np.zeros((n,1,48,128))##Pre padding
  count,av=0,0
  for k in final:
    temp=embedd([k]).numpy()
    z[av,0,count,:]=temp[0][:]##embed() funcion de vectorizacion
    count+=1
    if count==47:
      count=0
      av+=1
  return z
# Preaara el texto para la funcion de nubes de palabras
def txtWC(inp):
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
modelBin = tf.keras.models.load_model('protomodelo.h5')
modelAS6 = tf.keras.models.load_model('protomodeloAS6p1.h5')
##Diccionarios para evaluar las predicciones de los modelos
polaridad={0:'Positivas',1:'Negativas'}
emocion={0:'Alegria',1:'Sorpresa',2:'Tristeza',3:'Miedo',4:'Ira',5:'Disguto'}
def AS(input_text):
  ####Cargamos los modelos entrenados:
  mt=tx2m(input_text) ###Respresentación numérica del texto
  result1=modelBin.predict(mt)
  result2=modelAS6.predict(mt)
  a1=sum(result1)
  a2=sum(result2)
  re1=polaridad[np.where(a1 == np.amax(a1))[0][0]]
  re2=emocion[np.where(a2 == np.amax(a2))[0][0]]
  v=txtWC(input_text) ###Texto original procesado para la nube de palabras
  wc_result=wc.generate(v) ## Variable ppara almacenar la nube de palabras
  #plt.axis("off")
  #plt.imshow(wc_result, interpolation='bilinear')
  #S=wc_result
  plt.show()
  return re1
############__________WEBAPP_______###################
st.title("Hola soy Psibot")
txt = st.text_area('Introduce lo que me quieres contar',on_change=None, placeholder='Expresate aquí')
if st.button('Contar'):
  if txt=='':
    st.write('Escribe en el espacio de arriba')
  else:
     st.write('Sentimentos:')
     st.write(AS(txt))
