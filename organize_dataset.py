from pathlib import Path
import shutil
import re

origem = Path("./dataset")
destino = Path("./dataset_mvcnn")

extensoes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

#Padronizando nome dos arquivos
for arquivo in origem.rglob("*"):

    if arquivo.suffix.lower() not in extensoes:
        continue

    novo_nome = arquivo.name.replace("-", "_")

    if novo_nome != arquivo.name:

        novo_caminho = arquivo.with_name(novo_nome)

        arquivo.rename(novo_caminho)

        print(f"{arquivo.name} -> {novo_nome}")

#Padronizando pastas
for arquivo in origem.rglob("*"):

    if arquivo.suffix.lower() not in extensoes:
        continue

    nome_sem_ext = arquivo.stem

    # Remove apenas números no final
    individuo = re.sub(r'[\W\d_]+$', '', nome_sem_ext)
    
    pasta_individuo = destino / individuo

    nome_arquivo = nome_sem_ext +"_"+ arquivo.parent.name + arquivo.suffix
  
    pasta_individuo.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        arquivo,
        pasta_individuo / nome_arquivo
    )

print("Concluído.")