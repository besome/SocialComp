# O mundo canta cada vez mais na própria língua, mas sobre as mesmas coisas?

Projeto de Computação Social: letras do top 100 anual do Spotify de 9 países (EUA, Reino Unido, Canadá, Austrália,
Alemanha, França, Brasil, México e Japão; 2017–2025): convergência semântica entre países, redes de países e de
artistas, difusão de músicas e conteúdo explícito/tóxico.

| Hipótese | Enunciado | Teste principal |
|---|---|---|
| **H1 — glocalização** | A forma fica mais local e o conteúdo fica mais global | Fatia da atenção no idioma local e em artistas domésticos; similaridade (1 − Wasserstein) entre as estreias domésticas, letra traduzida |
| **H2 — gradiente linguístico** | Quanto mais distantes as línguas, mais distantes as letras, e a barreira não cai | Regressão diádica com QAP: similaridade ~ distância linguística × ano |
| **H3 — estilo** | O estilo converge mais rápido que o tema | η² do país ao longo dos anos: estilo (code-switching, repetição, toxicidade, sentimento) × temas |

O resumo das três está em `resultados/hipoteses_resumo.csv`.

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
# GPU: instale o torch com CUDA primeiro (https://pytorch.org/get-started/locally/), depois:
pip install -r requirements.txt
./rodar.sh gpu                 # só precisa rodar uma vez (GPU); os resultados ficam em dados/
./rodar.sh                     # hipoteses, redes e difusao (CPU)
./rodar.sh hipoteses           # só uma etapa
```

| Ordem | Notebook | O que faz | Custo |
|---|---|---|---|
| 1 | `pipelineGPU.ipynb` | Embeddings BGE-M3 (principal) e LaBSE, vetores dos 10 temas, toxicidade (Detoxify) e sentimento (XLM-T) | **GPU**: ~1 h (RTX 4050 ou T4 do Kaggle); 8–12 h em CPU. Baixa ~5 GB de modelos na 1ª vez |
| 2 | `pipelineHipoteses.ipynb` | H1, H2, H3, tópicos (BERTopic) e robustez | CPU, ~4 min |
| 3 | `pipelineRedes.ipynb` | Rede bipartida país–música com nulo *curveball*, centralidade, comunidades; rede de colaborações e homofilia | CPU, ~2 min |
| 4 | `pipelineDifusao.ipynb` | Modelo de Bass (duas fases), rede de liderança, alcance × toxicidade, semântica × difusão | CPU, ~1 min |

Os caches são por `track_id` + hash do texto: rodar de novo só calcula o que mudou. O `pipelineGPU` também roda
no Kaggle (instruções na primeira célula).

**Teste rápido** das hipóteses: `N_BOOT=20 N_BOOT_ROB=10 N_PERM=5 N_QAP=100 ./rodar.sh hipoteses`.

### O que subir para o GitHub

- `dados/embeddings/`, `dados/toxicidade.csv`, `dados/sentimento.csv` (só números, sem letras)
- `resultados/` e os notebooks executados

`dados/cache/` fica fora do git: é regenerado automaticamente (e a planilha de anotação de lá tem letras).

### Versão anterior (6 países, top 40)

`pipelineAnalise.ipynb`, `pipelineToxicidade.ipynb` e `secao12_analise_lexical.ipynb` são da primeira versão
(EUA, Brasil, México, Espanha, Austrália e Japão; relatório em `relatorio.pdf`). Os resultados deles estão em
`resultados/v1/`. Eles não foram adaptados para os 9 países.

## Dados

| Arquivo | Conteúdo | Resolução |
|---|---|---|
| `finalData.csv` | Top 120 anual de 9 países, com letra, idioma e tradução para o inglês (`letra_en`); as análises usam o top 100 | anual, 2017–2025 |
| `Datasets/spotify_top200_positions_*.csv` | Top 200 de 37 países | **1 semana a cada 4** (13/ano), 2018-05 a 2026 |
| `Datasets/MGDplus/charts/regional/` | Top 200 de 69 mercados | **diário**, 2017-01 a 2022-03 |
| `Datasets/MGDplus/songs/` | Metadados das faixas, incluindo a flag `explicit` | até 2022 |

O top 100 tem 8.100 linhas (país × ano × música) e 4.075 faixas únicas. `comum.carrega_top()` lê o top 100 e
acrescenta as colunas derivadas: `peso_atencao` (fatia dos streams na célula), `estreia`, `repetida`, `catalogo`,
`compartilhada`, origem do artista e `domestico`.

## Decisões de método

- **A similaridade entre países não usa clusters.** Cada célula (país, ano) é uma nuvem de embeddings, e a
  similaridade é 1 − Wasserstein (transporte ótimo, custo = distância do cosseno). MMD, cosseno entre centroides e
  1 − JS sobre tópicos ficam como robustez.
- **Versão principal: letras traduzidas.** A letra original e a original "centralizada" (embedding menos o vetor
  médio do idioma) aparecem na robustez.
- **Bootstrap de Poisson pareado:** cada música recebe um peso Poisson(1), o mesmo em todos os países em que aparece
  no ano. Reamostrar cada país separadamente quebrava os pares de músicas idênticas e enviesava a tendência.
- **Tópicos com K-Means (30 finos, 8 grossos) dentro do BERTopic.** O HDBSCAN juntava quase tudo em 2 grupos ou
  deixava ~70% como ruído: as letras formam um contínuo, sem temas separados.
- **Alucinações da tradução são removidas** (`comum.limpa_traducao`): o NLLB inventou frases de regulamento da UE
  em 139 linhas de 69 letras.
- **Artista doméstico só é confiável fora do bloco anglófono:** nos EUA, Reino Unido, Canadá e Austrália, 60–70%
  das músicas são de artistas "globais". A forma local do H1 é testada nos 5 países não anglófonos.
- **Toda estrutura de rede é comparada a um modelo nulo:** *curveball* preservando graus na rede país–música e
  permutação de nós (QAP) na regressão diádica.
- **O Bass tem duas fases:** a fração adotada no lançamento (simultânea) e o contágio gradual depois dela; a
  resolução de 4 semanas é calibrada contra o chart diário.
- **Toxicidade:** Detoxify `multilingual` na letra original (en, es, pt, fr); alemão e japonês usam o escore da
  tradução. Validada contra a flag `explicit` do Spotify.
- **Limitação:** a taxonomia de 10 temas fixos não foi validada por anotação humana; os temas só descrevem o
  conteúdo, e as conclusões das hipóteses não dependem deles.
