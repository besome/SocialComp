"""Funções e constantes compartilhadas pelos notebooks de análise.

Tudo o que precisa ser idêntico entre os notebooks (limpeza de letra, identificação da mesma música
entre versões, mapa de países/idiomas, estilo dos gráficos e leitura dos charts) fica aqui, para que
uma mudança valha para todos ao mesmo tempo.
"""
import glob
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RAIZ = Path(__file__).resolve().parent
PASTA_DATASETS = RAIZ / "Datasets"
PASTA_RES = RAIZ / "resultados"
PASTA_CACHE = RAIZ / "dados" / "cache"          # tabelas intermediárias (sem letras)

# ---------------------------------------------------------------- países
# os 9 países com letras (top 100 anual): 8 dos 10 maiores mercados da IFPI (sem China e Coreia do Sul) + México
PAISES = ["us", "gb", "ca", "au", "de", "fr", "br", "mx", "jp"]
TOP_N = 100          # o finalData.csv traz 120 por célula; as análises usam as 100 primeiras pelo score

# idioma -> família (H2: distância linguística em 3 níveis)
FAMILIA_LINGUA = {"en": "germânica", "de": "germânica", "fr": "românica", "pt": "românica", "es": "românica",
                  "ja": "japônica"}


def dist_ling(a, b):
    """0 = mesmo idioma, 1 = mesma família, 2 = famílias diferentes (entre os mercados a e b)."""
    la, lb = LINGUA_PAIS[a], LINGUA_PAIS[b]
    if la == lb:
        return 0
    return 1 if FAMILIA_LINGUA.get(la) == FAMILIA_LINGUA.get(lb) else 2

NOMES = {
    "ar": "Argentina", "at": "Áustria", "au": "Austrália", "be": "Bélgica", "br": "Brasil", "ca": "Canadá",
    "ch": "Suíça", "cl": "Chile", "co": "Colômbia", "cz": "Tchéquia", "de": "Alemanha", "dk": "Dinamarca",
    "ee": "Estônia", "es": "Espanha", "fi": "Finlândia", "fr": "França", "gb": "Reino Unido", "ie": "Irlanda",
    "is": "Islândia", "it": "Itália", "jp": "Japão", "kr": "Coreia do Sul", "lu": "Luxemburgo", "mx": "México",
    "ng": "Nigéria", "nl": "Países Baixos", "no": "Noruega", "nz": "Nova Zelândia", "pe": "Peru", "pl": "Polônia",
    "pt": "Portugal", "ro": "Romênia", "se": "Suécia", "tr": "Turquia", "ua": "Ucrânia", "us": "EUA",
    "za": "África do Sul", "global": "Global",
}

# língua dominante do mercado (usada para dizer se uma música "atravessa" a barreira do idioma)
LINGUA_PAIS = {
    "us": "en", "gb": "en", "au": "en", "ca": "en", "ie": "en", "nz": "en", "za": "en", "ng": "en",
    "ar": "es", "cl": "es", "co": "es", "mx": "es", "pe": "es", "es": "es",
    "br": "pt", "pt": "pt",
    "de": "de", "at": "de", "ch": "de",
    "fr": "fr", "be": "fr", "lu": "fr",          # Bélgica e Luxemburgo são multilíngues; o chart é mais próximo do francês
    "se": "sv", "no": "no", "dk": "da", "fi": "fi", "is": "is",
    "it": "it", "nl": "nl", "pl": "pl", "cz": "cs", "ro": "ro", "ua": "uk", "ee": "et", "tr": "tr",
    "jp": "ja", "kr": "ko",
}

# blocos culturais/linguísticos (para homofilia e para comparar com as comunidades encontradas)
BLOCO_PAIS = {
    **{c: "anglo" for c in ["us", "gb", "au", "ca", "ie", "nz", "za", "ng"]},
    **{c: "hispano" for c in ["ar", "cl", "co", "mx", "pe", "es"]},
    **{c: "luso" for c in ["br", "pt"]},
    **{c: "germânico" for c in ["de", "at", "ch"]},
    **{c: "franco" for c in ["fr", "be", "lu"]},
    **{c: "nórdico" for c in ["se", "no", "dk", "fi", "is"]},
    **{c: "leste europeu" for c in ["pl", "cz", "ro", "ua", "ee"]},
    "it": "itálico", "nl": "neerlandês", "tr": "turco", "jp": "japonês", "kr": "coreano",
}
ORDEM_BLOCOS = ["anglo", "hispano", "luso", "germânico", "franco", "nórdico", "leste europeu",
                "itálico", "neerlandês", "turco", "japonês", "coreano"]

# ---------------------------------------------------------------- estilo
COR = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#9b5de5", "neutro": "#b9b8b3",
       "texto": "#0b0b0b", "texto2": "#52514e", "grade": "#e6e5e1", "fundo": "#fcfcfb"}
SEQ = LinearSegmentedColormap.from_list("seq_azul", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])  # vermelho = afastando · azul = aproximando
CORES_BLOCO = {"anglo": "#2a78d6", "hispano": "#eb6834", "luso": "#1baf7a", "germânico": "#9b5de5",
               "franco": "#e3a21a", "nórdico": "#3cb8c9", "leste europeu": "#8a6e4b", "itálico": "#d64f8f",
               "neerlandês": "#5b7f2a", "turco": "#c0392b", "japonês": "#555555", "coreano": "#999999",
               "global": "#0b0b0b", "outro": "#cccccc"}


def aplica_estilo():
    plt.rcParams.update({
        "figure.facecolor": COR["fundo"], "axes.facecolor": COR["fundo"], "savefig.facecolor": COR["fundo"],
        "axes.edgecolor": COR["grade"], "axes.labelcolor": COR["texto2"], "text.color": COR["texto"],
        "xtick.color": COR["texto2"], "ytick.color": COR["texto2"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": COR["grade"], "grid.linewidth": 0.8, "axes.axisbelow": True,
        "lines.linewidth": 2, "lines.markersize": 6, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def plota_matriz(M, titulo, ax, cmap, vmin=None, vmax=None, fmt="{:.2f}", nomes=NOMES, anotar=True, fonte=8.5):
    im = ax.imshow(M.values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(M.columns)), [nomes.get(c, c) for c in M.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(M.index)), [nomes.get(i, i) for i in M.index])
    ax.grid(False)
    if anotar:
        lo = vmin if vmin is not None else np.nanmin(M.values)
        hi = vmax if vmax is not None else np.nanmax(M.values)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                if np.isnan(v):
                    continue
                escuro = abs(v - (lo + hi) / 2) > 0.3 * (hi - lo) if cmap is DIV else (v - lo) > 0.55 * (hi - lo)
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fonte,
                        color="#ffffff" if escuro else COR["texto"])
    ax.set_title(titulo, loc="left")
    return im


def tendencia(x, y):
    """Inclinação OLS, IC 95% e p-valor; com poucos anos, é a leitura mais honesta que dá para fazer."""
    from scipy.stats import linregress
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~np.isnan(y)
    if ok.sum() < 3:
        return {"inclinacao": np.nan, "ic95_lo": np.nan, "ic95_hi": np.nan, "p": np.nan}
    r = linregress(x[ok], y[ok])
    from scipy.stats import t as tdist
    q = tdist.ppf(0.975, ok.sum() - 2)
    return {"inclinacao": r.slope, "ic95_lo": r.slope - q * r.stderr, "ic95_hi": r.slope + q * r.stderr, "p": r.pvalue}


# ---------------------------------------------------------------- letras e identidade da música
_LRC = re.compile(r"^(?:\s*\[(?:\d{1,2}:\d{2}(?:[.:]\d{1,3})?|[a-z]{2,6}:[^\]\n]*)\])+[ \t]*", re.I | re.M)
_ROTULO = re.compile(r"^\s*[\[\(](?:refr[aã]o|chorus|pre-?chorus|verse|verso|coro|estribillo|bridge|ponte|puente|"
                     r"intro|outro|hook|interlude)[^\]\)\n]*[\]\)]\s*$", re.I | re.M)


def limpa_letra(t):
    """Remove timestamps/cabeçalhos de LRC ("[03:17.76]", "[ar:Artista]") e rótulos estruturais ("[Refrão]")."""
    if not isinstance(t, str):
        return ""
    t = _LRC.sub("", t)
    t = _ROTULO.sub("", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def chave_musica(title, artists):
    """Identifica a mesma música com track_ids diferentes (remix, ao vivo, relançamento, versões regionais)."""
    t = _sem_acento(str(title).lower())
    t = re.sub(r"[\(\[].*?[\)\]]", " ", t)        # (feat. X), [Remix], (Ao Vivo)
    t = re.split(r"\s+-\s+", t)[0]                # "- Ao Vivo", "- Remix", "- From ..."
    t = re.sub(r"[^\w]+", " ", t).strip()
    a = re.split(r"\s*(?:\||,|&|\bfeat\.?|\bft\.?)\s*", _sem_acento(str(artists).lower()))[0]
    a = re.sub(r"[^\w]+", " ", a).strip()
    return f"{t}::{a}"


# Alucinações do NLLB: em trechos curtos ("Ih, oh!") ele às vezes inventa frases de regulamento da União Europeia
# ("The Commission shall adopt delegated acts..."). São linhas soltas no meio de traduções boas (~140 linhas em ~75
# letras, quase todas do Brasil); a linha inteira sai.
_ALUCINACAO = re.compile(r"\b(?:this Regulation|Commission shall|Official Journal|Member States?\b|delegated acts|"
                         r"implementing acts|European Parliament|competent authority|following definitions shall apply|"
                         r"Standing Committee)", re.I)


def limpa_traducao(t):
    """limpa_letra + remoção das linhas alucinadas pelo tradutor."""
    if not isinstance(t, str) or not t.strip():
        return np.nan
    t = "\n".join(l for l in t.split("\n") if not _ALUCINACAO.search(l))
    t = limpa_letra(t)
    return t if t else np.nan


def carrega_letras(arq=None):
    """finalData.csv com letra limpa e chave. Uma linha por (país, ano, música)."""
    arq = Path(arq) if arq else RAIZ / "finalData.csv"
    d = pd.read_csv(arq)
    d["letra_limpa"] = d["letra"].map(limpa_letra)
    if "letra_en" in d.columns:
        d["letra_en_limpa"] = d["letra_en"].map(limpa_traducao)
    d["chave"] = [chave_musica(t, a) for t, a in zip(d["title"], d["artists"])]
    return d


def idioma_simples(i):
    return i if i in ("en", "es", "pt", "ja", "fr", "de") else "outro"


# ---------------------------------------------------------------- charts
def carrega_4semanas(min_paises=30):
    """Top 200 semanal de ~37 países (2018-05 a 2026), amostrado 1 semana a cada 4.

    As semanas intermediárias só têm 1-2 mercados (global, Turquia); elas são descartadas para que todos os
    países tenham a mesma amostragem. 'global' também sai: não é um país.
    """
    arqs = sorted(glob.glob(str(PASTA_DATASETS / "spotify_top200_positions_*.csv")))
    d = pd.concat([pd.read_csv(a) for a in arqs], ignore_index=True)
    d = d[d["cc"] != "global"]
    n = d.groupby("week_end")["cc"].nunique()
    d = d[d["week_end"].isin(n[n >= min_paises].index)].copy()
    d["week_end"] = pd.to_datetime(d["week_end"])
    d["release_date"] = pd.to_datetime(d["release_date"], errors="coerce", format="%Y-%m-%d")
    d["year"] = d["week_end"].dt.year
    tk = d.drop_duplicates("track_id")[["track_id", "title", "artists"]]
    tk["chave"] = [chave_musica(t, a) for t, a in zip(tk["title"], tk["artists"])]
    return d.merge(tk[["track_id", "chave"]], on="track_id", how="left")


def carrega_diario_entradas(paises=None, cache=True):
    """MGDplus: Top 200 DIÁRIO de 69 mercados (2017-01-01 a 2022-03-13), resumido por (mercado, música).

    Ler os 69 arquivos (~2 GB) leva alguns minutos, então o resumo vai para dados/cache/ e é reaproveitado.
    Colunas: cc, track_id, chave, primeira (1º dia no top 200), ultima, dias (nº de dias no chart), melhor (posição).
    """
    arq_cache = PASTA_CACHE / "mgd_entradas_diarias.csv.gz"
    if cache and arq_cache.exists():
        r = pd.read_csv(arq_cache, parse_dates=["primeira", "ultima"])
    else:
        partes = []
        for a in sorted(glob.glob(str(PASTA_DATASETS / "MGDplus/charts/regional/spotify_charts_regional_*.csv"))):
            cc = Path(a).stem.split("_")[-1]
            if cc == "global":
                continue
            x = pd.read_csv(a, sep="\t", usecols=["ID", "Track", "Artist", "Position", "Date"])
            g = x.groupby("ID").agg(primeira=("Date", "min"), ultima=("Date", "max"), dias=("Date", "nunique"),
                                    melhor=("Position", "min"), title=("Track", "first"), artists=("Artist", "first"))
            partes.append(g.reset_index().assign(cc=cc))
        r = pd.concat(partes, ignore_index=True).rename(columns={"ID": "track_id"})
        tk = r.drop_duplicates("track_id")
        mapa = dict(zip(tk["track_id"], [chave_musica(t, a) for t, a in zip(tk["title"], tk["artists"])]))
        r["chave"] = r["track_id"].map(mapa)
        r = r.drop(columns=["title", "artists"])
        r["primeira"] = pd.to_datetime(r["primeira"])
        r["ultima"] = pd.to_datetime(r["ultima"])
        if cache:
            PASTA_CACHE.mkdir(parents=True, exist_ok=True)
            r.to_csv(arq_cache, index=False)
    if paises is not None:
        r = r[r["cc"].isin(paises)]
    return r


def datas_lancamento():
    """release_date por track_id, juntando o MGDplus (até 2022) e os charts de 4 semanas (até 2026).
    Datas só com ano ('2019') são descartadas: não servem para medir atraso em dias."""
    s = pd.read_csv(PASTA_DATASETS / "MGDplus/songs/spotify_hit_songs_dataset_complete.csv", sep="\t",
                    usecols=["song_id", "release_date", "explicit"]).rename(columns={"song_id": "track_id"})
    arqs = sorted(glob.glob(str(PASTA_DATASETS / "spotify_top200_positions_*.csv")))
    w = pd.concat([pd.read_csv(a, usecols=["track_id", "release_date"]) for a in arqs]).drop_duplicates("track_id")
    r = pd.concat([s[["track_id", "release_date"]], w]).drop_duplicates("track_id")
    r = r[r["release_date"].astype(str).str.len() == 10]
    r["release_date"] = pd.to_datetime(r["release_date"], errors="coerce")
    return r.dropna().set_index("track_id")["release_date"], s.drop_duplicates("track_id").set_index("track_id")["explicit"]


# ---------------------------------------------------------------- top 100 com colunas derivadas
def _primeiro_artista(a):
    """O finalData mistura separadores: ' | ' (charts de 4 semanas) e ', ' (chart diário de 2017)."""
    return re.split(r"\s*\|\s*|,\s*", str(a))[0].strip().lower()


def origem_artistas(h_global=0.80, min_solo=10, cache=True):
    """País de origem de cada artista (pelo nome do artista principal, em minúsculas), estimado pelos charts.

    Mesma regra do pipelineRedes, mas sobre os ~37 países dos charts de 4 semanas: origem = país onde o artista
    mais aparece (contando só músicas solo quando ele tem >= min_solo entradas solo). Quem está espalhado por quase
    todos os países (entropia normalizada >= h_global) recebe global = True. Colunas: origem, share, entropia, global.
    """
    arq = PASTA_CACHE / "origem_artistas.csv"
    if cache and arq.exists():
        return pd.read_csv(arq, index_col=0)
    w = carrega_4semanas()
    e = w[["cc", "week_end", "artists"]].dropna().copy()
    e["solo"] = ~e["artists"].str.contains("|", regex=False)
    e = e.assign(art=e["artists"].str.split(r"\s*\|\s*")).explode("art")
    e["art"] = e["art"].str.strip().str.lower()
    todas = e.groupby(["art", "cc"]).size().unstack(fill_value=0)
    solo = e[e["solo"]].groupby(["art", "cc"]).size().unstack(fill_value=0).reindex(columns=todas.columns, fill_value=0)
    usa_solo = solo.index[solo.sum(axis=1) >= min_solo]
    cont = todas.copy()
    cont.loc[usa_solo] = solo.loc[usa_solo]
    p = cont.div(cont.sum(axis=1), axis=0)
    H = -(p * np.log(p.where(p > 0, 1))).sum(axis=1) / np.log(cont.shape[1])
    r = pd.DataFrame({"origem": cont.idxmax(axis=1), "share": p.max(axis=1), "entropia": H})
    r["global"] = r["entropia"] >= h_global
    if cache:
        PASTA_CACHE.mkdir(parents=True, exist_ok=True)
        r.to_csv(arq)
    return r


def primeira_aparicao(cache=True):
    """Primeira data em que cada música (chave) aparece em QUALQUER chart: diário (69 mercados, 2017-01 a 2022-03)
    ou de 4 semanas (~37 países, 2018-05 em diante)."""
    arq = PASTA_CACHE / "primeira_aparicao.csv"
    if cache and arq.exists():
        return pd.read_csv(arq, index_col=0, parse_dates=["primeira"])["primeira"]
    d = carrega_diario_entradas().groupby("chave")["primeira"].min()
    w = carrega_4semanas().groupby("chave")["week_end"].min()
    r = pd.concat([d, w], axis=1).min(axis=1).rename("primeira")
    if cache:
        PASTA_CACHE.mkdir(parents=True, exist_ok=True)
        r.to_frame().to_csv(arq)
    return r


def carrega_top(n=TOP_N, arq=None, derivadas=True):
    """finalData.csv -> top n por (país, ano), uma linha por (país, ano, música), com as colunas derivadas:

    - pos: posição pelo score (pontos) dentro da célula
    - peso_atencao: streams da música / streams da célula (soma 1 em cada célula)
    - repetida: a música estava no top n do mesmo país no ano anterior (NaN no 1º ano)
    - estreia: 1º ano da música no top n do país (NaN no 1º ano, em que não dá para saber)
    - compartilhada: está no top n de 2+ países no mesmo ano
    - catalogo: 1ª aparição em qualquer chart 2+ anos antes (NaN antes de 2019: os charts começam em 2017)
    - artista, origem, global_artista, domestico: origem do artista principal (ver origem_artistas)
    """
    d = carrega_letras(arq)
    d = d[d["cc"].isin(PAISES) & (d["letra_limpa"].str.len() >= 30)].copy()
    d = (d.sort_values("pontos", ascending=False)
          .drop_duplicates(["cc", "year", "chave"])                 # mesma música 2x na mesma célula
          .sort_values(["cc", "year", "pontos"], ascending=[True, True, False]))
    d["pos"] = d.groupby(["cc", "year"]).cumcount() + 1
    d = d[d["pos"] <= n].copy()
    d["idioma"] = d["idioma"].map(idioma_simples)
    d["peso_atencao"] = d["streams"] / d.groupby(["cc", "year"])["streams"].transform("sum")
    if not derivadas:
        return d.reset_index(drop=True)

    ano0 = d["year"].min()
    ant = set(zip(d["cc"], d["year"] + 1, d["chave"]))
    d["repetida"] = [float((c, y, k) in ant) for c, y, k in zip(d["cc"], d["year"], d["chave"])]
    d.loc[d["year"] == ano0, "repetida"] = np.nan
    primeiro = d.groupby(["cc", "chave"])["year"].transform("min")
    d["estreia"] = (d["year"] == primeiro).astype(float)
    d.loc[d["year"] == ano0, "estreia"] = np.nan
    d["compartilhada"] = d.groupby(["year", "chave"])["cc"].transform("nunique") >= 2

    prim = primeira_aparicao()
    anos = d["year"] - d["chave"].map(prim).dt.year
    d["catalogo"] = (anos >= 2).astype(float)
    d.loc[d["year"] < ano0 + 2, "catalogo"] = np.nan

    org = origem_artistas()
    d["artista"] = d["artists"].map(_primeiro_artista)
    d["origem"] = d["artista"].map(org["origem"])
    d["global_artista"] = d["artista"].map(org["global"]).fillna(False).astype(bool)
    d["domestico"] = (d["origem"] == d["cc"]).astype(float)
    d.loc[d["origem"].isna(), "domestico"] = np.nan
    return d.reset_index(drop=True)
