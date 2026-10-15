import sys
from pathlib import Path

DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_MODELO = DIRETORIO_ATUAL.parent / "modelo"
sys.path.append(str(DIRETORIO_MODELO))

import joblib
import pandas as pd
import streamlit as st
from spellchecker import SpellChecker
from Funcoes_nltk import (
    COLUNAS_NLTK,
    STOPWORDS,
    calcular_metricas,
    extrair_palavras,
    radical,
    tokenizador_nltk,
)

from Trocas import trocar_genero_texto

spell = SpellChecker(language="pt")

st.set_page_config(
    page_title="Classificador & Troca de Gênero",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def carregar_modelo():
    caminho_modelo = DIRETORIO_MODELO / "modelo_svm.joblib"
    return joblib.load(caminho_modelo)

try:
    modelo_svm = carregar_modelo()
except Exception as e:
    st.error(f"Erro ao carregar o modelo treinado: {e}")
    st.stop()


def extrair_features(texto: str) -> pd.DataFrame:
    metricas, palavras = calcular_metricas(texto)
    qtd_erros = len(spell.unknown(palavras)) if spell and palavras else 0

    dados = {
        "Texto_Cru": texto,
        "Qtd. Palavras": metricas["Qtd. Palavras"],
        "% Canônicas": metricas["% Canônicas"],
        "Qtd. Frases": metricas["Qtd. Frases"],
        "Erros Ortog.": qtd_erros,
        "% Stopwords": metricas["% Stopwords"],
        "Tam. Médio Palavra": metricas["Tam. Médio Palavra"],
        "Tam. Médio Frase": metricas["Tam. Médio Frase"],
        "Diversidade Lexical": metricas["Diversidade Lexical"],
    }
    return pd.DataFrame([dados])


def analisar(texto: str):
    df = extrair_features(texto)
    nivel = modelo_svm.predict(df)[0]
    return nivel, df.iloc[0]


METRICAS_EXIBIDAS = [
    ("Qtd. Palavras", "Qtd. Palavras", "{:.0f}"),
    ("Qtd. Frases", "Qtd. Frases", "{:.0f}"),
    ("Erros Ortográficos", "Erros Ortog.", "{:.0f}"),
    ("Div. Lexical", "Diversidade Lexical", "{:.2f}"),
    ("% Canônicas", "% Canônicas", "{:.1f}%"),
    ("% Stopwords", "% Stopwords", "{:.1f}%"),
    ("Tam. Médio Palavra", "Tam. Médio Palavra", "{:.1f}"),
    ("Tam. Médio Frase", "Tam. Médio Frase", "{:.1f}"),
]


def mostrar_metricas(m, m_ref=None):
    for i in range(0, len(METRICAS_EXIBIDAS), 4):
        cols = st.columns(4)
        for col, (rotulo, chave, fmt) in zip(cols, METRICAS_EXIBIDAS[i:i + 4]):
            valor = float(m[chave])
            delta = None
            if m_ref is not None:
                diff = valor - float(m_ref[chave])
                delta = f"{diff:+.2f}" if abs(diff) > 1e-9 else "0"
            col.metric(rotulo, fmt.format(valor), delta=delta, delta_color="off")


def mostrar_nivel(rotulo: str, nivel) -> None:
    st.markdown(
        f"""
        <div style="line-height:1.3;">
            <div style="font-size:0.85rem; opacity:0.7;">{rotulo}</div>
            <div style="font-size:1.4rem; font-weight:600;">{nivel}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Interface Streamlit
st.title("Classificador de Nível Escolar & Troca de Gênero")
st.markdown(
    "Analise o nível escolar do texto ou troque o gênero de todas as palavras "
    "possíveis e veja o quanto a mutação altera a complexidade do texto."
)

col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("1. Texto de Entrada")
    texto_usuario = st.text_area(
        "Texto para Análise:",
        height=280,
        placeholder="Escreva ou cole aqui o texto..."
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        botao_classificar = st.button("Classificar Texto", type="primary", use_container_width=True)
    with col_btn2:
        botao_trocar_genero = st.button("Trocar Gênero e Comparar", use_container_width=True)

with col_output:
    st.subheader("2. Resultados")

    if botao_trocar_genero:
        if not texto_usuario.strip():
            st.warning("Insira um texto antes de realizar a troca de gênero.")
        else:
            with st.spinner("Trocando gênero e classificando os dois textos..."):
                texto_modificado, alteracoes = trocar_genero_texto(texto_usuario)
                nivel_orig, m_orig = analisar(texto_usuario)
                nivel_novo, m_novo = analisar(texto_modificado)

            st.markdown("### Texto Reescrito")
            st.text_area(
                "Texto no outro gênero:",
                value=texto_modificado,
                height=200,
                disabled=True,
            )
            st.caption(f"{len(alteracoes)} palavras alteradas.")

            st.markdown("---")
            st.markdown("### Impacto da mutação na complexidade")

            col_n1, col_n2 = st.columns(2)
            with col_n1:
                mostrar_nivel("Nível do texto original", nivel_orig)
            with col_n2:
                mostrar_nivel("Nível do texto mutado", nivel_novo)

            if nivel_orig == nivel_novo:
                st.info("A mutação **não alterou** o nível estimado pelo modelo.")
            else:
                st.warning(f"A mutação **mudou** o nível estimado: {nivel_orig} → {nivel_novo}.")

            st.markdown("#### Texto original")
            mostrar_metricas(m_orig)

            st.markdown("#### Texto mutado (variação em relação ao original)")
            mostrar_metricas(m_novo, m_ref=m_orig)

    elif botao_classificar:
        if not texto_usuario.strip():
            st.warning("Insira um texto válido antes de classificar.")
        else:
            with st.spinner("Classificando com modelo SVM..."):
                predicao, m = analisar(texto_usuario)

            st.success(f"### Nível Estimado: **{predicao}**")
            st.markdown("---")
            st.markdown("### Métricas do Texto")
            mostrar_metricas(m)