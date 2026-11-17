#!/usr/bin/env bash
# Executa os notebooks em ordem e salva as saídas dentro deles (--inplace).
#
#   ./rodar.sh                        # tudo: hipoteses redes difusao (a etapa gpu roda à parte, no Kaggle)
#   ./rodar.sh toxicidade difusao     # só as etapas indicadas
#   PYTHON=.venv/bin/python ./rodar.sh
#
# Parâmetros por variável de ambiente (os padrões já são os da versão final):
#   N_BOOT, N_BOOT_ROB, N_PERM, N_QAP  pipelineHipoteses (bootstrap, robustez, nulo, QAP)
#   N_BOOT, N_PERM, N_PERM_Q   pipelineAnalise  (versão antiga, 6 países)
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
[ ${#ETAPAS[@]} -eq 0 ] && ETAPAS=(hipoteses redes difusao)

for e in "${ETAPAS[@]}"; do
    case "$e" in
        gpu)        roda pipelineGPU.ipynb ;;
        hipoteses)  roda pipelineHipoteses.ipynb ;;
        analise)    roda pipelineAnalise.ipynb ;;
        toxicidade) roda pipelineToxicidade.ipynb ;;
        redes)      roda pipelineRedes.ipynb ;;
        difusao)    roda pipelineDifusao.ipynb ;;
        *) echo "etapa desconhecida: $e (use gpu, hipoteses, analise, toxicidade, redes, difusao)"; exit 1 ;;
    esac
done
echo "==================== fim ($(date +%H:%M:%S)) — resultados em resultados/"
