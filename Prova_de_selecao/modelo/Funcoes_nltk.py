import re
import nltk

for pacote in ["stopwords", "punkt", "punkt_tab", "rslp"]:
    nltk.download(pacote, quiet=True)

from Calculo_canonicidade import e_canonica
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

STOPWORDS = set(stopwords.words("portuguese"))
stemmer = RSLPStemmer()

PALAVRA_RE = re.compile(r"^[^\W\d_]+(?:-[^\W\d_]+)*$")
ASPAS = "“”‘’«»\"'"

COLUNAS_NLTK = [
    "% Stopwords",
    "Tam. Médio Palavra",
    "Tam. Médio Frase",
    "Diversidade Lexical",
]


# ---------------------------------------------------------------------------
# Tokenização
# ---------------------------------------------------------------------------
def extrair_palavras(texto: str) -> list[str]:
    tokens = word_tokenize(texto.lower(), language="portuguese")
    palavras = []
    for t in tokens:
        t = t.strip(ASPAS)
        if t and PALAVRA_RE.match(t):
            palavras.append(t)
    return palavras


def extrair_frases(texto: str) -> list[str]:
    return [s for s in sent_tokenize(texto, language="portuguese") if s.strip()]


def radical(palavra: str) -> str:
    return stemmer.stem(palavra)

# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def calcular_metricas(texto: str):
    texto = texto.strip()
    palavras = extrair_palavras(texto) if texto else []
    qtd_palavras = len(palavras)

    if qtd_palavras == 0:
        return {
            "Qtd. Palavras": 0,
            "Palavras Canônicas": 0,
            "% Canônicas": 0.0,
            "Qtd. Frases": 0,
            "% Stopwords": 0.0,
            "Tam. Médio Palavra": 0.0,
            "Tam. Médio Frase": 0.0,
            "Diversidade Lexical": 0.0,
        }, []

    frases = extrair_frases(texto)
    qtd_frases = max(len(frases), 1)

    qtd_canonicas = sum(1 for p in palavras if e_canonica(p))
    qtd_stop = sum(1 for p in palavras if p in STOPWORDS)

    tam_medio_palavra = sum(len(p.replace("-", "")) for p in palavras) / qtd_palavras

    return {
        "Qtd. Palavras": qtd_palavras,
        "Palavras Canônicas": qtd_canonicas,
        "% Canônicas": round(qtd_canonicas / qtd_palavras * 100, 2),
        "Qtd. Frases": qtd_frases,
        "% Stopwords": round(qtd_stop / qtd_palavras * 100, 2),
        "Tam. Médio Palavra": round(tam_medio_palavra, 3),
        "Tam. Médio Frase": round(qtd_palavras / qtd_frases, 3),
        "Diversidade Lexical": round(len(set(palavras)) / qtd_palavras, 4),
    }, palavras

def tokenizador_nltk(texto):
    return [radical(p) for p in extrair_palavras(texto) if p not in STOPWORDS]