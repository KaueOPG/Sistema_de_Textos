import random
import spacy

try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    from spacy.cli import download
    download("pt_core_news_sm")
    nlp = spacy.load("pt_core_news_sm")

# Mapeamento para artigos e pronomes
ARTIGOS_E_PRONOMES = {
    "o": "a", "a": "o", "os": "as", "as": "os",
    "um": "uma", "uma": "um", "uns": "umas", "umas": "uns",
    "do": "da", "da": "do", "dos": "das", "das": "dos",
    "no": "na", "na": "no", "nos": "nas", "nas": "nos",
    "pelo": "pela", "pela": "pelo", "pelos": "pelas", "pelas": "pelos",
    "ao": "à", "à": "ao", "aos": "às", "às": "aos",
    "ele": "ela", "ela": "ele", "eles": "elas", "elas": "eles",
    "dele": "dela", "dela": "dele", "deles": "delas", "delas": "deles",
    "meu": "minha", "minha": "meu", "meus": "minhas", "minhas": "meus",
    "seu": "sua", "sua": "seu", "seus": "suas", "suas": "seus",
    "este": "esta", "esta": "este", "estes": "estas", "estas": "estes",
    "esse": "essa", "essa": "esse", "esses": "essas", "essas": "esses",
    "aquele": "aquela", "aquela": "aquele", "aqueles": "aquelas", "aquelas": "aqueles",
}


def flexionar_genero(token) -> str | None:
    p = token.text.lower()

    if p in ARTIGOS_E_PRONOMES:
        return ARTIGOS_E_PRONOMES[p]

    if token.pos_ in {"NOUN", "ADJ", "DET", "PRON"}:
        if p.endswith("o") and len(p) > 2: return p[:-1] + "a"
        if p.endswith("a") and len(p) > 2: return p[:-1] + "o"
        if p.endswith("os") and len(p) > 3: return p[:-2] + "as"
        if p.endswith("as") and len(p) > 3: return p[:-2] + "os"
        if p.endswith("or") and len(p) > 2: return p + "a"
        if p.endswith("ora") and len(p) > 3: return p[:-1]

    return None


def preservar_caixa(original: str, nova: str) -> str:
    if original.isupper(): return nova.upper()
    if original.istitle(): return nova.capitalize()
    return nova.lower()


def trocar_genero_texto(texto: str, proporcao: float = 1.0) -> tuple[str, list[dict]]:
    doc = nlp(texto)
    
    candidatos = [t for t in doc if t.is_alpha and flexionar_genero(t) is not None]
    
    qtd = max(1, int(len(candidatos) * proporcao)) if candidatos else 0
    sorteados = set(random.sample(candidatos, min(qtd, len(candidatos)))) if candidatos else set()

    tokens_resultado = []
    alteracoes = []

    for token in doc:
        if token in sorteados:
            nova_palavra = flexionar_genero(token)
            nova_fmt = preservar_caixa(token.text, nova_palavra)
            tokens_resultado.append(nova_fmt + token.whitespace_)
            alteracoes.append({
                "Original": token.text,
                "Gênero Trocado": nova_fmt,
                "Classe Gramatical": token.pos_
            })
        else:
            tokens_resultado.append(token.text_with_ws)

    return "".join(tokens_resultado), alteracoes