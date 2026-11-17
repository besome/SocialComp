#!/usr/bin/env bash
# Executa os notebooks em ordem e salva as saídas dentro deles (--inplace).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

roda() {
    echo "==================== $1  ($(date +%H:%M:%S))"
    # "python -m nbconvert" (e não "jupyter nbconvert") garante que o kernel é o desse mesmo Python
    "$PY" -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$1"
}

ETAPAS=("$@")
[ ${#ETAPAS[@]} -eq 0 ] && ETAPAS=(Tratamento Translate )

for e in "${ETAPAS[@]}"; do
    case "$e" in
        Tratamento)    roda pipelineTratamento.ipynb ;;
        Translate) roda pipelineTranslate.ipynb ;;
        *) echo "etapa desconhecida: $e (use Tratamento, Translate)"; exit 1 ;;
    esac
done
echo "==================== fim ($(date +%H:%M:%S))"
