# Sistema_de_Textos
# 📚 Sistema de Textos — Classificador de Nível Escolar e Troca de Gênero

> Aplicação web interativa para análise estatística de textos, predição de nível escolar e alteração automática de gênero gramatical em português.

🔗 **Acesse a aplicação online:** [Sistema de Textos no Streamlit](https://sistemadetextos-gzap64ay8jbfmywmhmnegc.streamlit.app/)

---

## Sobre o Projeto

O **Sistema de Textos** foi desenvolvido para analisar características linguísticas e estilísticas de textos em português, estimar a qual etapa da educação básica/superior o texto pertence e realizar transformações gramaticais de gênero com acompanhamento de variação de métricas.

### Principais Funcionalidades

1. **Classificação e Análise de Texto:**
   - Cálculo de métricas lexicais e estatísticas:
     - Quantidade total de palavras e frases.
     - Porcentagem de *stopwords*.
     - Diversidade lexical (*Type-Token Ratio*).
     - Tamanho médio de palavras e frases.
     - Contagem de possíveis erros ortográficos.
   - Predição automatizada da etapa escolar estimada do texto.

2. **Troca de Gênero e Comparação:**
   - Flexão gramatical automática de substantivos, adjetivos, artigos e pronomes (masculino ↔ feminino).
   - Preservação de caixa de letras (maiúsculas/minúsculas).
   - Reclassificação do texto alterado e exibição de painel comparativo de métricas antes e depois da transformação.

---

## Categorias do Modelo de Machine Learning

O modelo de classificação analisa o texto e o enquadra em uma das quatro etapas de ensino:

-  **Ensino Fundamental I**
-  **Ensino Fundamental II**
-  **Ensino Médio**
-  **Ensino Superior**

---

## Tecnologias Utilizadas

- **Linguagem:** Python 3 ou superior
- **Interface Web:** Streamlit
- **Processamento de Linguagem Natural (PLN):**
  - **spaCy** (`pt_core_news_sm`) para análise morfossintática e flexão de gênero.
  - **NLTK** para tokenização e remoção de stopwords.
  - **pyspellchecker** para verificação ortográfica.
- **Ciência de Dados:**
  - **Scikit-Learn** (`LinearSVC`, `TF-IDF Vectorizer`, `ColumnTransformer`, `StandardScaler`).
  - **Pandas** para estruturação e manipulação de dados.
  - **Joblib** para persistência do modelo treinado.

---

##  Estrutura do Repositório

```text
Sistema_de_Textos/
├── Prova_de_selecao/
│   ├── dataset_caed/      # Base de dados utilizada para treino e validação
│   ├── modelo/            # Scripts de treinamento, avaliação e artefato do modelo (.joblib)
│   └── web/               # Código da aplicação Streamlit e módulos de processamento
│       ├── app.py         # Interface do usuário (Streamlit)
│       └── Trocas.py      # Módulo de flexão e alteração sintática de gênero
├── README.md              # Documentação do projeto
├── requirements.txt       # Dependências e bibliotecas Python necessárias
└── relatorio.pdf          # Relatório com explicações avançadas e o desenvolvimento aplicado
```

## Como Executar o Projeto Localmente

### Passo a Passo

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/seu-usuario/Sistema_de_Textos.git](https://github.com/seu-usuario/Sistema_de_Textos.git)
   cd Sistema_de_Textos
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate

   # Windows (Command Prompt / PowerShell)
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute a aplicação Streamlit:**
   ```bash
   streamlit run Prova_de_selecao/web/app.py
   ```
