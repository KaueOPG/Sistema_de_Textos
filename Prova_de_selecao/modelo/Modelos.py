import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from Funcoes_nltk import (
    COLUNAS_NLTK,
    STOPWORDS,
    calcular_metricas,
    extrair_palavras,
    radical,
    tokenizador_nltk,
)

COLUNAS_NUMERICAS = [
    "Qtd. Palavras", "% Canônicas", "Qtd. Frases", "Erros Ortog.",
] + COLUNAS_NLTK

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

df_metricas = pd.read_csv("tabela_instancias_detalhada.csv")
df_textos = pd.read_csv("tabela_textos_crus.csv")

df_limpo = df_metricas.merge(
    df_textos[["Classificação", "Arquivo", "Texto_Cru"]],
    on=["Classificação", "Arquivo"]
)

X = df_limpo[["Texto_Cru"] + COLUNAS_NUMERICAS]
y = df_limpo["Classificação"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Pré-processamento: 
def construir_preprocessador(scaler):
    return ColumnTransformer(
        transformers=[
            (
                "text_tfidf",
                TfidfVectorizer(
                    tokenizer=tokenizador_nltk,
                    lowercase=False,      
                    token_pattern=None,  
                    ngram_range=(1, 2),
                    max_features=5000,
                ),
                "Texto_Cru",
            ),
            ("num_scaler", scaler, COLUNAS_NUMERICAS),
        ]
    )


# Modelos (SVM e Naive Bayes)
modelos = {
    "SVM (LinearSVC)": Pipeline([
        ("features", construir_preprocessador(StandardScaler())),
        ("classifier", LinearSVC(class_weight="balanced", max_iter=5000, random_state=42)),
    ]),
    "Naive Bayes (MultinomialNB)": Pipeline([
        ("features", construir_preprocessador(MinMaxScaler())),
        ("classifier", MultinomialNB()),
    ]),
}

NOMES_ARQUIVOS_SALVAR = {
    "SVM (LinearSVC)": "modelo_svm.joblib",
    "Naive Bayes (MultinomialNB)": "modelo_naive_bayes.joblib",
}

# Treinamento e avaliação
resumo = []
for nome, pipeline in modelos.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("=" * 70)
    print(nome)
    print("=" * 70)
    print(classification_report(y_test, y_pred))
    rotulos = sorted(y.unique())
    print("Matriz de confusão (linhas = real, colunas = previsto):")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=rotulos),
        index=rotulos, columns=rotulos,
    ))
    print()
    resumo.append((nome, f1_score(y_test, y_pred, average="macro")))

    # Salva o modelo treinado
    arquivo_saida = NOMES_ARQUIVOS_SALVAR[nome]
    joblib.dump(pipeline, arquivo_saida)

print("=" * 70)
print("RESUMO (F1 macro)")
for nome, f1 in sorted(resumo, key=lambda r: r[1], reverse=True):
    print(f"  {nome:<30} {f1:.4f}")