#lIBRERIA DE LAWEB APP:
import streamlit as st
##Cargamos las librerias necesarias para trabajar los datos
#import re
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
#embedd = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-de-dim128-with-normalization/1")
##Embedding de 50
embedd = hub.load("https://tfhub.dev/google/nnlm-es-dim50-with-normalization/2")
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
  n=1+(long//20)
  z=np.zeros((n,1,20,50))##Pre padding
  count,av=0,0
  for k in final:
    temp=embedd([k]).numpy()
    z[av,0,count,:]=temp[0][:]##embed() funcion de vectorizacion
    count+=1
    if count==19:
      count=0
      av+=1
  return z
# Preaara el texto para la funcion de nubes de palabras
def txtWC(inp):
  #quitar emojis como funcion:
  s=inp
  def deEmojify(texto):
      return emoji.get_emoji_regexp().sub("", texto)
  ##Quitar emojis
  fr=deEmojify(s)##Listo
  ##Tokenizar
  toks=WPT.tokenize(fr)
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
#############
st.title("Hola soy Psibot")
