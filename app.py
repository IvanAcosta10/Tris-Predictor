#estable
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="TRIS Predictor Estable",
    page_icon="🎯",
    layout="wide",
)

LOCAL_CSV = Path("tris_historico.csv")
SORTEOS = ["Medio Día", "De las Tres", "Extra", "De las Siete", "Clásico"]

# Mejor configuración encontrada en 300 pruebas.
VENTANA = 100
RECENCIA = 0.0
TRANSICION = 0.0
SUAVIZADO = 1.0

COMBINACIONES = np.array(
    list(itertools.product(range(10), repeat=4)),
    dtype=np.int8,
)


@st.cache_data
def leer_csv(origen):
    df = pd.read_csv(origen, dtype={"numero": str})

    requeridas = {"fecha", "sorteo", "numero"}
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError("Faltan columnas: " + ", ".join(sorted(faltantes)))

    df = df[["fecha", "sorteo", "numero"]].copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["numero"] = (
        df["numero"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    df = df.dropna(subset=["fecha"])
    df = df[df["numero"].str.fullmatch(r"\d{5}", na=False)]
    df = df.drop_duplicates(subset=["fecha", "sorteo"], keep="last")
    return df.sort_values(["fecha", "sorteo"]).reset_index(drop=True)


def preparar_directa4(df, lado):
    serie = df["numero"].astype(str).str.zfill(5)
    if lado == "Primeros 4":
        return serie.str[:4].tolist()
    return serie.str[-4:].tolist()


def probabilidades_posicion(numeros):
    muestra = numeros[-min(VENTANA, len(numeros)):]
    salida = np.zeros((4, 10), dtype=float)

    for pos in range(4):
        conteo = np.full(10, SUAVIZADO, dtype=float)

        for numero in muestra:
            conteo[int(numero[pos])] += 1.0

        salida[pos] = conteo / conteo.sum()

    return salida


def calcular_ranking(numeros, top_n):
    probs = probabilidades_posicion(numeros)

    score = np.ones(10000, dtype=float)

    for pos in range(4):
        score *= probs[pos, COMBINACIONES[:, pos]]

    score = np.power(score, 0.25)

    indices = np.argsort(score)[::-1][:top_n]

    filas = []

    for ranking, indice in enumerate(indices, start=1):
        numero = f"{indice:04d}"
        filas.append(
            {
                "Ranking": ranking,
                "Número": numero,
                "Puntuación": round(float(score[indice]) * 100, 4),
                "P1": round(float(probs[0, int(numero[0])]) * 100, 4),
                "P2": round(float(probs[1, int(numero[1])]) * 100, 4),
                "P3": round(float(probs[2, int(numero[2])]) * 100, 4),
                "P4": round(float(probs[3, int(numero[3])]) * 100, 4),
            }
        )

    return pd.DataFrame(filas), probs


def buscar_resultado_en_ranking(numero_real, ranking):
    coincidencia = ranking[ranking["Número"] == numero_real]

    if coincidencia.empty:
        return None

    return int(coincidencia.iloc[0]["Ranking"])


st.title("🎯 TRIS Predictor Estable")
st.caption(
    "Modelo rápido basado en frecuencia por posición sobre una ventana de 100 sorteos."
)

archivo_subido = st.sidebar.file_uploader(
    "Cargar tris_historico.csv",
    type=["csv"],
)

try:
    if archivo_subido is not None:
        datos = leer_csv(archivo_subido)
        origen = "Archivo cargado manualmente"
    elif LOCAL_CSV.exists():
        datos = leer_csv(LOCAL_CSV)
        origen = "Archivo incluido en la aplicación"
    else:
        datos = pd.DataFrame()
        origen = None
except Exception as error:
    st.error(f"No se pudo leer el archivo: {error}")
    st.stop()

if datos.empty:
    st.warning("Sube `tris_historico.csv` desde el panel lateral.")
    st.stop()

st.sidebar.success(f"{len(datos):,} resultados cargados")
st.sidebar.caption(origen)

m1, m2, m3 = st.columns(3)
m1.metric("Resultados", f"{len(datos):,}")
m2.metric("Primera fecha", datos["fecha"].min().strftime("%d/%m/%Y"))
m3.metric("Última fecha", datos["fecha"].max().strftime("%d/%m/%Y"))

tab_prediccion, tab_verificacion, tab_modelo, tab_base = st.tabs(
    ["🎯 Ranking", "✅ Verificar resultado", "📊 Modelo", "🗂️ Base"]
)

with tab_prediccion:
    c1, c2, c3 = st.columns(3)

    sorteo = c1.selectbox("Sorteo", SORTEOS)
    lado = c2.selectbox("Directa 4", ["Últimos 4", "Primeros 4"])
    top_n = c3.slider("Cantidad de candidatos", 10, 100, 30, 10)

    filtrados = (
        datos[datos["sorteo"] == sorteo]
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    numeros = preparar_directa4(filtrados, lado)
    ranking, probabilidades = calcular_ranking(numeros, top_n)

    mejor = ranking.iloc[0]

    r1, r2, r3 = st.columns(3)
    r1.metric("Mejor clasificado", mejor["Número"])
    r2.metric("Puntuación interna", f'{mejor["Puntuación"]:.4f}')
    r3.metric("Ventana usada", VENTANA)

    st.dataframe(
        ranking,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Descargar ranking CSV",
        ranking.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"ranking_{sorteo}_{lado}.csv",
        mime="text/csv",
    )

    st.info(
        "La puntuación sirve únicamente para ordenar candidatos. "
        "No representa una probabilidad real de que el número vaya a salir."
    )

with tab_verificacion:
    st.subheader("Comprobar un resultado real")

    v1, v2, v3 = st.columns(3)

    sorteo_v = v1.selectbox(
        "Sorteo",
        SORTEOS,
        key="verificar_sorteo",
    )
    lado_v = v2.selectbox(
        "Directa 4",
        ["Últimos 4", "Primeros 4"],
        key="verificar_lado",
    )
    top_v = v3.slider(
        "Top que quieres comprobar",
        10,
        100,
        100,
        10,
    )

    numero_real = st.text_input(
        "Resultado real de Directa 4",
        max_chars=4,
        placeholder="Ejemplo: 3272",
    )

    if st.button(
        "Verificar resultado",
        type="primary",
        use_container_width=True,
    ):
        if not numero_real.isdigit() or len(numero_real) != 4:
            st.error("Escribe exactamente 4 dígitos.")
        else:
            filtrados_v = (
                datos[datos["sorteo"] == sorteo_v]
                .sort_values("fecha")
                .reset_index(drop=True)
            )
            numeros_v = preparar_directa4(filtrados_v, lado_v)
            ranking_v, _ = calcular_ranking(numeros_v, top_v)
            posicion = buscar_resultado_en_ranking(numero_real, ranking_v)

            if posicion is None:
                st.error(
                    f"{numero_real} no apareció dentro del Top {top_v}."
                )
            else:
                st.success(
                    f"{numero_real} apareció en la posición {posicion} "
                    f"dentro del Top {top_v}."
                )

with tab_modelo:
    st.subheader("Configuración validada")

    st.code(
        """
Ventana: 100 sorteos
Peso de recencia: 0
Peso de transición: 0
Suavizado: 1
        """.strip()
    )

    st.write(
        "En una prueba de 300 sorteos de Medio Día, esta configuración obtuvo:"
    )

    resultados_modelo = pd.DataFrame(
        {
            "Métrica": ["Top 10", "Top 20", "Top 50", "Top 100"],
            "Aciertos": ["0/300", "0/300", "2/300", "6/300"],
            "Porcentaje": ["0.0%", "0.0%", "0.667%", "2.0%"],
            "Referencia aleatoria": ["0.1%", "0.2%", "0.5%", "1.0%"],
        }
    )

    st.dataframe(
        resultados_modelo,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "El resultado de 6/300 todavía no demuestra una ventaja estable. "
        "Debe seguir midiéndose con sorteos nuevos fuera de la muestra usada."
    )

with tab_base:
    st.dataframe(
        datos.sort_values("fecha", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
