from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="TRIS Predictor V3",
    page_icon="🎯",
    layout="wide",
)

LOCAL_CSV = Path("tris_historico.csv")
SORTEOS = ["Medio Día", "De las Tres", "Extra", "De las Siete", "Clásico"]


@st.cache_data
def leer_csv(origen):
    df = pd.read_csv(origen, dtype={"numero": str})

    requeridas = {"fecha", "sorteo", "numero"}
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(
            "El archivo no contiene las columnas requeridas: "
            + ", ".join(sorted(faltantes))
        )

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


def normalizar(contador, claves):
    valores = np.array([contador.get(k, 0.0) for k in claves], dtype=float)
    if valores.max() == valores.min():
        return {k: 0.5 for k in claves}
    valores = (valores - valores.min()) / (valores.max() - valores.min())
    return dict(zip(claves, valores))


def preparar_directa4(df, lado):
    numeros = df["numero"].astype(str).str.zfill(5)
    if lado == "Primeros 4":
        return numeros.str[:4].tolist()
    return numeros.str[-4:].tolist()


def calcular_ranking(
    numeros,
    top_n=30,
    ventana_reciente=150,
    peso_recencia=1.5,
    excluir_vistos=False,
):
    if len(numeros) < 10:
        return pd.DataFrame()

    cantidad = len(numeros)
    pesos = np.ones(cantidad)
    ventana = min(ventana_reciente, cantidad)

    if ventana:
        pesos[-ventana:] = np.linspace(1.0, 1.0 + peso_recencia, ventana)

    digitos = [str(i) for i in range(10)]
    pares = [f"{i:02d}" for i in range(100)]

    frecuencia_global = Counter()
    frecuencia_posicion = [Counter() for _ in range(4)]
    frecuencia_pares = [Counter() for _ in range(3)]
    transiciones = [defaultdict(Counter) for _ in range(3)]

    for numero, peso in zip(numeros, pesos):
        for pos, digito in enumerate(numero):
            frecuencia_global[digito] += peso
            frecuencia_posicion[pos][digito] += peso

        for pos in range(3):
            frecuencia_pares[pos][numero[pos:pos + 2]] += peso
            transiciones[pos][numero[pos]][numero[pos + 1]] += peso

    global_norm = normalizar(frecuencia_global, digitos)
    posicion_norm = [
        normalizar(contador, digitos)
        for contador in frecuencia_posicion
    ]
    pares_norm = [
        normalizar(contador, pares)
        for contador in frecuencia_pares
    ]

    transiciones_norm = []
    for pos in range(3):
        por_digito = {}
        for digito in digitos:
            por_digito[digito] = normalizar(
                transiciones[pos][digito],
                digitos,
            )
        transiciones_norm.append(por_digito)

    vistos = set(numeros)
    filas = []

    for valor in range(10000):
        numero = f"{valor:04d}"

        if excluir_vistos and numero in vistos:
            continue

        d = list(numero)

        puntuacion_posicion = sum(
            posicion_norm[pos][d[pos]]
            for pos in range(4)
        ) / 4

        puntuacion_global = sum(
            global_norm[digito]
            for digito in d
        ) / 4

        puntuacion_pares = sum(
            pares_norm[pos][numero[pos:pos + 2]]
            for pos in range(3)
        ) / 3

        puntuacion_transiciones = sum(
            transiciones_norm[pos][d[pos]][d[pos + 1]]
            for pos in range(3)
        ) / 3

        repeticion = (4 - len(set(d))) / 3
        espejo = (
            int(d[0] == d[3]) +
            int(d[1] == d[2])
        ) / 2

        puntuacion = (
            0.38 * puntuacion_posicion
            + 0.18 * puntuacion_global
            + 0.22 * puntuacion_pares
            + 0.17 * puntuacion_transiciones
            + 0.03 * repeticion
            + 0.02 * espejo
        )

        filas.append(
            {
                "Número": numero,
                "Puntuación": puntuacion,
                "Posición": puntuacion_posicion,
                "Pares": puntuacion_pares,
                "Transiciones": puntuacion_transiciones,
                "Frecuencia": puntuacion_global,
            }
        )

    ranking = (
        pd.DataFrame(filas)
        .sort_values("Puntuación", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    ranking.insert(0, "Ranking", np.arange(1, len(ranking) + 1))

    for columna in [
        "Puntuación",
        "Posición",
        "Pares",
        "Transiciones",
        "Frecuencia",
    ]:
        ranking[columna] = (ranking[columna] * 100).round(2)

    return ranking


def tabla_frecuencias(numeros):
    filas = []
    for posicion in range(4):
        contador = Counter(numero[posicion] for numero in numeros)
        total = sum(contador.values())

        for digito in range(10):
            valor = contador.get(str(digito), 0)
            filas.append(
                {
                    "Posición": posicion + 1,
                    "Dígito": str(digito),
                    "Apariciones": valor,
                    "Porcentaje": round(valor / total * 100, 2) if total else 0,
                }
            )

    return pd.DataFrame(filas)


st.title("🎯 TRIS Predictor V4")
st.caption("Analizador estadístico de Directa 4 usando tu histórico descargado.")

archivo_subido = st.sidebar.file_uploader(
    "Cargar tris_historico.csv",
    type=["csv"],
    help="Sube el archivo generado por actualizar_tris.py.",
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
    st.warning(
        "Falta el archivo de resultados. Usa el panel lateral para subir "
        "`tris_historico.csv`."
    )
    st.stop()

st.sidebar.success(f"{len(datos):,} resultados cargados")
st.sidebar.caption(origen)

ultima_fecha = datos["fecha"].max()
primera_fecha = datos["fecha"].min()

m1, m2, m3 = st.columns(3)
m1.metric("Resultados cargados", f"{len(datos):,}")
m2.metric("Primera fecha", primera_fecha.strftime("%d/%m/%Y"))
m3.metric("Última fecha", ultima_fecha.strftime("%d/%m/%Y"))

tab_prediccion, tab_backtest, tab_analisis, tab_base = st.tabs(
    [
        "🎯 Ranking Directa 4",
        "🧪 Backtesting",
        "📊 Análisis",
        "🗂️ Base de datos",
    ]
)

with tab_prediccion:
    c1, c2, c3 = st.columns(3)

    sorteo = c1.selectbox("Sorteo", SORTEOS)
    lado = c2.selectbox("Directa 4", ["Últimos 4", "Primeros 4"])
    top_n = c3.slider("Cantidad de candidatos", 10, 100, 30, 10)

    c4, c5, c6 = st.columns(3)

    ventana = c4.slider(
        "Resultados recientes con mayor peso",
        20,
        500,
        150,
        10,
    )
    recencia = c5.slider(
        "Peso de resultados recientes",
        0.0,
        3.0,
        1.5,
        0.1,
    )
    excluir = c6.checkbox(
        "Excluir combinaciones ya vistas",
        value=False,
    )

    filtrados = datos[datos["sorteo"] == sorteo].sort_values("fecha")
    numeros = preparar_directa4(filtrados, lado)

    ranking = calcular_ranking(
        numeros,
        top_n=top_n,
        ventana_reciente=ventana,
        peso_recencia=recencia,
        excluir_vistos=excluir,
    )

    st.write(
        f"Analizando **{len(numeros):,} resultados** del sorteo "
        f"**{sorteo}**."
    )

    if ranking.empty:
        st.warning("No hay suficientes resultados para calcular el ranking.")
    else:
        mejor = ranking.iloc[0]

        r1, r2, r3 = st.columns(3)
        r1.metric("Mejor clasificado", mejor["Número"])
        r2.metric("Puntuación interna", f'{mejor["Puntuación"]:.2f}')
        r3.metric("Candidatos mostrados", len(ranking))

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
            "La puntuación sirve para ordenar patrones históricos. "
            "No es una probabilidad real ni garantiza un resultado."
        )


with tab_backtest:
    st.subheader("Prueba histórica del algoritmo")
    st.write(
        "Oculta cada resultado real, genera un Top 100 usando solamente "
        "los sorteos anteriores y revisa si el ganador habría aparecido."
    )

    bt1, bt2, bt3 = st.columns(3)
    sorteo_bt = bt1.selectbox(
        "Sorteo",
        SORTEOS,
        key="sorteo_backtest",
    )
    lado_bt = bt2.selectbox(
        "Directa 4",
        ["Últimos 4", "Primeros 4"],
        key="lado_backtest",
    )
    pruebas_bt = bt3.slider(
        "Últimos sorteos a probar",
        10,
        100,
        30,
        10,
        help="Empieza con 30. Una prueba grande tardará más.",
    )

    bt4, bt5 = st.columns(2)
    ventana_bt = bt4.slider(
        "Resultados recientes con mayor peso",
        20,
        500,
        150,
        10,
        key="ventana_backtest",
    )
    recencia_bt = bt5.slider(
        "Peso de resultados recientes",
        0.0,
        3.0,
        1.5,
        0.1,
        key="recencia_backtest",
    )

    if st.button(
        "▶ Ejecutar backtesting",
        type="primary",
        use_container_width=True,
    ):
        filtrados_bt = (
            datos[datos["sorteo"] == sorteo_bt]
            .sort_values("fecha")
            .reset_index(drop=True)
        )
        numeros_bt = preparar_directa4(filtrados_bt, lado_bt)
        fechas_bt = filtrados_bt["fecha"].tolist()

        if len(numeros_bt) < pruebas_bt + 200:
            st.error("No hay suficientes resultados para esta prueba.")
        else:
            inicio = len(numeros_bt) - pruebas_bt
            barra = st.progress(0)
            estado = st.empty()
            filas_bt = []

            for contador, indice in enumerate(
                range(inicio, len(numeros_bt)),
                start=1,
            ):
                entrenamiento = numeros_bt[:indice]
                numero_real = numeros_bt[indice]

                ranking_bt = calcular_ranking(
                    entrenamiento,
                    top_n=100,
                    ventana_reciente=ventana_bt,
                    peso_recencia=recencia_bt,
                    excluir_vistos=False,
                )

                lista_top100 = ranking_bt["Número"].tolist()
                posicion_real = (
                    lista_top100.index(numero_real) + 1
                    if numero_real in lista_top100
                    else None
                )

                filas_bt.append(
                    {
                        "Fecha": fechas_bt[indice],
                        "Resultado real": numero_real,
                        "Ranking": posicion_real if posicion_real else "Fuera del Top 100",
                        "Top 10": posicion_real is not None and posicion_real <= 10,
                        "Top 20": posicion_real is not None and posicion_real <= 20,
                        "Top 50": posicion_real is not None and posicion_real <= 50,
                        "Top 100": posicion_real is not None,
                    }
                )

                barra.progress(contador / pruebas_bt)
                estado.write(
                    f"Probando {contador} de {pruebas_bt}: "
                    f"{pd.to_datetime(fechas_bt[indice]).strftime('%d/%m/%Y')}"
                )

            barra.empty()
            estado.empty()

            resultado_bt = pd.DataFrame(filas_bt)
            st.session_state["resultado_backtest"] = resultado_bt

    if "resultado_backtest" in st.session_state:
        resultado_bt = st.session_state["resultado_backtest"]
        total_bt = len(resultado_bt)

        aciertos10 = int(resultado_bt["Top 10"].sum())
        aciertos20 = int(resultado_bt["Top 20"].sum())
        aciertos50 = int(resultado_bt["Top 50"].sum())
        aciertos100 = int(resultado_bt["Top 100"].sum())

        r1, r2, r3, r4 = st.columns(4)
        r1.metric(
            "Aciertos Top 10",
            f"{aciertos10}/{total_bt}",
            f"{aciertos10 / total_bt * 100:.1f}%",
        )
        r2.metric(
            "Aciertos Top 20",
            f"{aciertos20}/{total_bt}",
            f"{aciertos20 / total_bt * 100:.1f}%",
        )
        r3.metric(
            "Aciertos Top 50",
            f"{aciertos50}/{total_bt}",
            f"{aciertos50 / total_bt * 100:.1f}%",
        )
        r4.metric(
            "Aciertos Top 100",
            f"{aciertos100}/{total_bt}",
            f"{aciertos100 / total_bt * 100:.1f}%",
        )

        st.caption(
            "Como referencia, un Top 100 elegido completamente al azar "
            "cubriría aproximadamente 1% de las 10,000 combinaciones."
        )

        if aciertos100 / total_bt > 0.01:
            st.success(
                "En esta muestra, el Top 100 superó la referencia aleatoria del 1%."
            )
        else:
            st.warning(
                "En esta muestra, el Top 100 no superó claramente "
                "la referencia aleatoria del 1%."
            )

        st.dataframe(
            resultado_bt.sort_values("Fecha", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Descargar backtesting CSV",
            resultado_bt.to_csv(index=False).encode("utf-8-sig"),
            file_name="backtesting_tris.csv",
            mime="text/csv",
        )


with tab_analisis:
    c1, c2 = st.columns(2)
    sorteo_analisis = c1.selectbox(
        "Sorteo para analizar",
        SORTEOS,
        key="sorteo_analisis",
    )
    lado_analisis = c2.selectbox(
        "Sección de Directa 4",
        ["Últimos 4", "Primeros 4"],
        key="lado_analisis",
    )

    filtrados = datos[datos["sorteo"] == sorteo_analisis].sort_values("fecha")
    numeros = preparar_directa4(filtrados, lado_analisis)
    frecuencias = tabla_frecuencias(numeros)

    pivot = frecuencias.pivot(
        index="Dígito",
        columns="Posición",
        values="Porcentaje",
    )

    st.subheader("Frecuencia porcentual por posición")
    st.dataframe(pivot, use_container_width=True)

    globales = Counter("".join(numeros))
    global_df = pd.DataFrame(
        {
            "Dígito": [str(i) for i in range(10)],
            "Apariciones": [globales.get(str(i), 0) for i in range(10)],
        }
    ).sort_values("Apariciones", ascending=False)

    st.subheader("Dígitos más frecuentes")
    st.bar_chart(global_df.set_index("Dígito"))

    st.subheader("Últimos 30 resultados analizados")
    recientes = filtrados.sort_values("fecha", ascending=False).head(30).copy()
    recientes["Directa 4"] = preparar_directa4(
        recientes.sort_values("fecha"),
        lado_analisis,
    )[::-1]

    st.dataframe(
        recientes[["fecha", "sorteo", "numero", "Directa 4"]],
        use_container_width=True,
        hide_index=True,
    )

with tab_base:
    st.dataframe(
        datos.sort_values("fecha", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Descargar base completa",
        datos.to_csv(index=False).encode("utf-8-sig"),
        file_name="tris_historico_limpio.csv",
        mime="text/csv",
    )
