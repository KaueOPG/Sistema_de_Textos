from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import pandas as pd
from Funcoes_nltk import STOPWORDS, calcular_metricas
from spellchecker import SpellChecker
    
spell = SpellChecker(language="pt")

MAPEAMENTO_CLASSIFICACAO = {
    "1_Ensino_Fundamental_I": "Ensino Fundamental I",
    "2_Ensino_Fundamental_II": "Ensino Fundamental II",
    "3_Ensino_Medio": "Ensino Médio",
    "4_Ensino_Superior": "Ensino Superior",
}

EXCLUIR_STOPWORDS_NA_TABELA = True

def analisar_texto(texto):
    metricas, palavras = calcular_metricas(texto)

    qtd_erros = 0
    if spell is not None and palavras:
        qtd_erros = len(spell.unknown(palavras))

    return metricas, palavras, qtd_erros


def ler_arquivo(arquivo):
    try:
        with open(arquivo, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(arquivo, "r", encoding="latin-1") as f:
            return f.read()


def analisar_dataset(caminho_dataset):
    caminho_base = Path(caminho_dataset)

    if not caminho_base.exists():
        print(f"Erro: O diretório '{caminho_base}' não foi encontrado.")
        return

    categorias = sorted(
        d for d in caminho_base.iterdir() if d.is_dir() and not d.name.startswith(".")
    )

    # Leitura inicial de todos os arquivos
    todos_arquivos = []
    for categoria in categorias:
        nome_pasta = categoria.name
        classificacao = MAPEAMENTO_CLASSIFICACAO.get(nome_pasta, nome_pasta)
        for arquivo in sorted(categoria.glob("*.txt")):
            conteudo_limpo = ler_arquivo(arquivo).strip()
            todos_arquivos.append({
                "classificacao": classificacao,
                "nome_arquivo": arquivo.name,
                "caminho": arquivo,
                "conteudo": conteudo_limpo,
            })

    # Agrupamento para verificação de duplicidade
    hash_map = defaultdict(list)
    arquivos_vazios = []

    for item in todos_arquivos:
        texto = item["conteudo"]
        if not texto:
            arquivos_vazios.append(item)
            continue
        h = hashlib.md5(texto.encode("utf-8")).hexdigest()
        hash_map[h].append(item)

    arquivos_validos = []
    excluidos_cross_categoria = []
    excluidos_mesma_categoria = []

    for h, grupo in hash_map.items():
        classificacoes_distintas = sorted(list(set(x["classificacao"] for x in grupo)))
        if len(classificacoes_distintas) > 1:
            excluidos_cross_categoria.append({
                "grupo": grupo,
                "classificacoes": classificacoes_distintas,
            })
        else:
            arquivos_validos.append(grupo[0])
            if len(grupo) > 1:
                for copia in grupo[1:]:
                    excluidos_mesma_categoria.append({
                        "copia": copia,
                        "original": grupo[0],
                    })

    # Impressão detalhada das mudanças
    print("\n" + "=" * 130)
    print("                              RELATÓRIO DE ALTERAÇÕES E DE DUPLICAÇÕES")
    print("=" * 130)

    print("\n1. TEXTOS EXCLUÍDOS POR DUPLICIDADE ENTRE DUAS OU MAIS CLASSIFICAÇÕES:")
    if excluidos_cross_categoria:
        tot_cross_files = sum(len(g["grupo"]) for g in excluidos_cross_categoria)
        print(f"   [!] Total de grupos afetados: {len(excluidos_cross_categoria)} | Total de arquivos descartados: {tot_cross_files}")
        for idx, item in enumerate(excluidos_cross_categoria, 1):
            cats_str = " <-> ".join(item["classificacoes"])
            print(f"   Grupo {idx} (Categorias envolvidas: {cats_str}):")
            for arq in item["grupo"]:
                print(f"      - {arq['nome_arquivo']} [{arq['classificacao']}]")
    else:
        print("   Nenhum texto duplicado entre classificações distintas foi encontrado.")

    print("\n2. TEXTOS EXCLUÍDOS POR DUPLICIDADE NA MESMA CLASSIFICAÇÃO:")
    if excluidos_mesma_categoria:
        print(f"   [!] Total de cópias extras descartadas: {len(excluidos_mesma_categoria)}")
        for item in excluidos_mesma_categoria:
            copia = item["copia"]
            original = item["original"]
            print(f"   - Descartado: {copia['nome_arquivo']} [{copia['classificacao']}] (Cópia mantida: {original['nome_arquivo']})")
    else:
        print("   Nenhuma cópia extra na mesma classificação foi encontrada.")

    if arquivos_vazios:
        print("\n3. ARQUIVOS VAZIOS ENCONTRADOS E IGNORADOS:")
        print(f"   [!] Total de arquivos vazios: {len(arquivos_vazios)}")
        for arq in arquivos_vazios:
            print(f"   - {arq['nome_arquivo']} [{arq['classificacao']}]")

    print("\nRESUMO QUANTITATIVO:")
    print(f"  - Total de arquivos lidos: {len(todos_arquivos)}")
    print(f"  - Arquivos válidos mantidos: {len(arquivos_validos)}")
    tot_cross_descartados = sum(len(g["grupo"]) for g in excluidos_cross_categoria)
    print(f"  - Descartados por duplicidade entre classificações: {tot_cross_descartados}")
    print(f"  - Descartados por duplicidade na mesma classificação: {len(excluidos_mesma_categoria)}")
    if arquivos_vazios:
        print(f"  - Arquivos vazios descartados: {len(arquivos_vazios)}")
    print("=" * 130 + "\n")

    # Processamento dos arquivos válidos por categoria
    validos_por_cat = defaultdict(list)
    for arq in arquivos_validos:
        validos_por_cat[arq["classificacao"]].append(arq)

    todas_instancias = []
    textos_crus = []
    tabela_totais = []
    tabela_palavras_frequentes = []

    ordem_classificacoes = list(dict.fromkeys(MAPEAMENTO_CLASSIFICACAO.values()))

    for classificacao in ordem_classificacoes:
        arquivos_cat = validos_por_cat.get(classificacao, [])
        qtd_textos = len(arquivos_cat)

        cat_palavras = cat_canonicas = cat_frases = cat_erros = cat_stop = 0
        contador_palavras_cat = Counter()

        for item in arquivos_cat:
            arquivo_nome = item["nome_arquivo"]
            conteudo_limpo = item["conteudo"]

            metricas, palavras_lista, qtd_erros = analisar_texto(conteudo_limpo)
            contador_palavras_cat.update(palavras_lista)

            cat_palavras += metricas["Qtd. Palavras"]
            cat_canonicas += metricas["Palavras Canônicas"]
            cat_frases += metricas["Qtd. Frases"]
            cat_stop += round(
                metricas["% Stopwords"] / 100 * metricas["Qtd. Palavras"]
            )
            cat_erros += qtd_erros

            textos_crus.append({
                "Classificação": classificacao,
                "Arquivo": arquivo_nome,
                "Texto_Cru": conteudo_limpo,
            })

            # Instância individual para o CSV de métricas
            todas_instancias.append({
                "Classificação": classificacao,
                "Arquivo": arquivo_nome,
                "Qtd. Palavras": metricas["Qtd. Palavras"],
                "Palavras Canônicas": metricas["Palavras Canônicas"],
                "% Canônicas": metricas["% Canônicas"],
                "Qtd. Frases": metricas["Qtd. Frases"],
                "Erros Ortog.": qtd_erros,
                "% Stopwords": metricas["% Stopwords"],
                "Tam. Médio Palavra": metricas["Tam. Médio Palavra"],
                "Tam. Médio Frase": metricas["Tam. Médio Frase"],
                "Diversidade Lexical": metricas["Diversidade Lexical"],
            })

        pct_canonicas_cat = (cat_canonicas / cat_palavras * 100) if cat_palavras else 0.0
        pct_stop_cat = (cat_stop / cat_palavras * 100) if cat_palavras else 0.0
        media_frase_cat = (cat_palavras / cat_frases) if cat_frases else 0.0

        tabela_totais.append({
            "Classificação": classificacao,
            "Total Textos": qtd_textos,
            "Total Palavras": cat_palavras,
            "Total Frases": cat_frases,
            "Palavras/Frase": f"{media_frase_cat:.2f}",
            "Palavras Canônicas": cat_canonicas,
            "% Canônicas": f"{pct_canonicas_cat:.2f}%",
            "% Stopwords": f"{pct_stop_cat:.2f}%",
            "Erros Ortográficos": cat_erros,
        })

    diretorio_script = Path(__file__).resolve().parent

    # CSV com as métricas detalhadas
    df_instancias = pd.DataFrame(todas_instancias)
    caminho_csv_metricas = diretorio_script / "tabela_instancias_detalhada.csv"
    df_instancias.to_csv(caminho_csv_metricas, index=False, encoding="utf-8-sig")

    # CSV isolado apenas para o Texto Cru
    df_textos = pd.DataFrame(textos_crus)
    caminho_csv_textos = diretorio_script / "tabela_textos_crus.csv"
    df_textos.to_csv(caminho_csv_textos, index=False, encoding="utf-8-sig")

    # Tabela
    df_totais = pd.DataFrame(tabela_totais)
    print("                        TABELA: VALORES TOTAIS POR CLASSIFICAÇÃO")
    print("=" * 130)
    print(df_totais.to_string(index=False))



diretorio_atual = Path(__file__).resolve().parent
caminho_dataset = diretorio_atual.parent / "dataset_caed"
analisar_dataset(caminho_dataset)