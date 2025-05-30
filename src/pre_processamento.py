
import re
import string
import nltk
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words_en = set(stopwords.words('english'))
stop_words_pt = set(stopwords.words('portuguese'))
stop_words = stop_words_en.union(stop_words_pt)
stemmer = PorterStemmer()

def limpar_texto(texto):
    texto = re.sub(r'<.*?>', '', texto)
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto)
    palavras = texto.split()
    palavras = [stemmer.stem(p) for p in palavras if p not in stop_words]
    return ' '.join(palavras)

def preprocessar_dataset(df):
    if 'conteudo' not in df.columns:
        raise KeyError("Coluna 'conteudo' não encontrada no DataFrame!")
    df['conteudo'] = df['conteudo'].fillna('').apply(limpar_texto)    
    return df

def vetorizar_textos(textos, **kwargs):
    vectorizer = TfidfVectorizer(**kwargs)
    X = vectorizer.fit_transform(textos)
    return X, vectorizer

