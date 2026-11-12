# As letras de diferentes países estão ficando parecidas?

Projeto de Computação Social: semântica das letras do top 40 anual de 6 países (EUA, Brasil, México, Espanha,
Austrália, Japão, 2017–2025), redes de países e de artistas, difusão de músicas entre países e conteúdo
explícito/tóxico.

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
# GPU: instale o torch com CUDA primeiro (https://pytorch.org/get-started/locally/), depois:
pip install -r requirements.txt
./rodar.sh                     # tudo, na ordem certa (as saídas ficam salvas dentro dos notebooks)
./rodar.sh toxicidade difusao  # só algumas etapas
```

O primeiro uso baixa os modelos: LaBSE (~1,9 GB) e Detoxify `multilingual` + `unbiased` (~1,6 GB).

| Ordem | Notebook | O que faz | Custo |
|---|---|---|---|
| 1 | `pipelineAnalise.ipynb` | Embeddings (LaBSE), clusters, similaridade entre países, H1/H2, Louvain com modelo nulo | Leve se `dados/embeddings/` existir (~5 min CPU); sem cache, gera os embeddings (GPU recomendada) |
| 2 | `pipelineToxicidade.ipynb` | Detoxify nas letras originais e traduzidas, validação contra a flag `explicit`, toxicidade por país/ano | **Pesado: GPU** (~1 h em CPU, minutos em GPU) |
| 3 | `pipelineRedes.ipynb` | Rede bipartida país–música (37 países) com nulo *curveball*, centralidade, comunidades; rede de colaborações e homofilia | CPU, ~5 min |
| 4 | `pipelineDifusao.ipynb` | Modelo de Bass (duas fases), rede de liderança, alcance × toxicidade, semântica × difusão | CPU, ~5 min (usa as saídas de 1 e 2) |

Os notebooks 3 e 4 podem rodar antes do 2, mas aí o modelo de alcance roda sem toxicidade. Depois que a
toxicidade terminar, rode `./rodar.sh difusao` de novo.

**Teste rápido** antes da execução completa: `TOX_LIMITE=20 N_NULO=50 N_BOOT=50 N_PERM=50 N_PERM_Q=50 N_BOOT_BASS=10 ./rodar.sh`.
Nesse modo, `dados/toxicidade.csv` e os arquivos `resultados/toxicidade_*` saem com só 20 faixas, então rode de
novo sem `TOX_LIMITE` antes de subir.

### O que subir de volta para o GitHub

- `dados/toxicidade.csv` (só escores, sem letras)
- `dados/embeddings/` (vetores; evita recalcular os embeddings)
- `resultados/` e os notebooks executados

`dados/cache/` fica fora do git: é regenerado automaticamente.

## Dados

| Arquivo | Conteúdo | Resolução |
|---|---|---|
| `finalData.csv` | Top 40 anual de 6 países, com letra, idioma e tradução para o inglês (`letra_en`) | anual, 2017–2025 |
| `Datasets/spotify_top200_positions_*.csv` | Top 200 de 37 países | **1 semana a cada 4** (13/ano), 2018-05 a 2026 |
| `Datasets/MGDplus/charts/regional/` | Top 200 de 67 mercados | **diário**, 2017-01 a 2022-03 |
| `Datasets/MGDplus/songs/` | Metadados das faixas, incluindo a flag `explicit` | até 2022 |

Funções compartilhadas (limpeza de letra, identificação da mesma música entre versões, países/idiomas/blocos,
estilo dos gráficos e leitura dos charts) ficam em `comum.py`.

## Decisões de método

- **A versão principal usa as letras traduzidas.** Com a letra original, os clusters separam sobretudo o idioma
  (NMI ≈ 0,5); com a tradução, o que sobra é conteúdo. A versão original aparece como comparação
  (`VERSAO=original` inverte as duas).
- **O H1 passa por uma bateria de robustez** (seção 11 do `pipelineAnalise`): 5 sementes × 4 valores de k,
  composição de idiomas fixa ao longo dos anos, só músicas no idioma local, e as duas métricas (1−JS e cosseno)
  lado a lado. Resultados em `resultados/robustez_H1*.csv`.
- **Toda estrutura de rede é comparada a um modelo nulo:** rótulos de país embaralhados (6 países) e *curveball*
  preservando graus (rede bipartida de 37 países).
- **O Bass tem duas fases.** Cerca de 80% dos países adotam uma música na semana do lançamento (lançamento global
  simultâneo), e um Bass único vira uma exponencial com q = 0. Então separamos a fração simultânea e ajustamos o
  Bass só na difusão gradual que vem depois.
- **A resolução de 4 semanas é calibrada contra o chart diário:** a mesma amostragem é simulada no diário para
  ver o quanto ela distorce as estimativas.
- **A toxicidade é validada contra a flag `explicit`** do Spotify. O japonês usa o escore da tradução, porque não
  está entre as línguas de treino do modelo multilíngue.
