#!/usr/bin/env bash
# Executa os notebooks em ordem e salva as saídas dentro deles (--inplace).
#
#   ./rodar.sh                        # tudo: analise toxicidade redes difusao
#   ./rodar.sh toxicidade difusao     # só as etapas indicadas
#   PYTHON=.venv/bin/python ./rodar.sh
#
# Parâmetros por variável de ambiente (os padrões já são os da versão final):
#   N_BOOT, N_PERM, N_PERM_Q   pipelineAnalise  (bootstrap e permutações)
#   TOX_BATCH, TOX_LIMITE      pipelineToxicidade (lote na GPU; TOX_LIMITE=20 = teste rápido)
#   N_NULO                     pipelineRedes    (amostras do nulo curveball)
#   N_BOOT_BASS, H_DIAS        pipelineDifusao  (bootstrap do Bass; janela de difusão)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

roda() {
    echo "==================== $1  ($(date +%H:%M:%S))"
    # "python -m nbconvert" (e não "jupyter nbconvert") garante que o kernel é o desse mesmo Python
    "$PY" -m nbconvert --execute --inplace --ExecutePreprocessor.timeout=-1 "$1"
}

ETAPAS=("$@")
[ ${#ETAPAS[@]} -eq 0 ] && ETAPAS=(analise toxicidade redes difusao)

for e in "${ETAPAS[@]}"; do
    case "$e" in
        analise)    roda pipelineAnalise.ipynb ;;
        toxicidade) roda pipelineToxicidade.ipynb ;;
        redes)      roda pipelineRedes.ipynb ;;
        difusao)    roda pipelineDifusao.ipynb ;;
        *) echo "etapa desconhecida: $e (use analise, toxicidade, redes, difusao)"; exit 1 ;;
    esac
done
echo "==================== fim ($(date +%H:%M:%S)) — resultados em resultados/"
