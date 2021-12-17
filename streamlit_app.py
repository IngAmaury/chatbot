import streamlit as st
with st.spinner('Espere, por favor...'):
#time.sleep(5)
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
!pip install emoji
import emoji
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import io
##Embedding tf2-preview-nnlm https://tfhub.dev/google/collections/tf2-preview-nnlm/1
#https://tfhub.dev/google/collections/nnlm/1
#embedd = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-de-dim128-with-normalization/1")
##Embedding de 50
embedd = hub.load("https://tfhub.dev/google/nnlm-es-dim50-with-normalization/2")
st.success('Done!')
st.title("Hola soy tu chatbot")
