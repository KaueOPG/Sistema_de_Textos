VOGAIS = set("aeiouáéíóúâêôãõàü")

def classificar_canonicidade(palavra: str) -> str:
    letras = [c for c in palavra.lower() if c.isalpha()]
    
    # Verifica se a palavra tem número ímpar de letras (não canônica)
    if len(letras) % 2 != 0:
        return "Não Canônica"
    
    # Gera o padrão de C e V para a palavra
    padrao = "".join("V" if c in VOGAIS else "C" for c in letras)
    
    # Verifica se o padrão é a repetição exata de 'CV'
    if padrao == "CV" * (len(letras) // 2):
        return "Canônica"
    
    return "Não Canônica"


def e_canonica(palavra: str) -> bool:
    return classificar_canonicidade(palavra) == "Canônica"