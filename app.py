
from pathlib import Path
import itertools
import time

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

st.set_page_config(
    page_title="TRIS AI V1",
    page_icon="🧠",
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
            "Faltan columnas: " + ", ".join(sorted(faltantes))
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


def preparar_directa4(df, lado):
    serie = df["numero"].astype(str).str.zfill(5)
    if lado == "Primeros 4":
        return serie.str[:4].tolist()
    return serie.str[-4:].tolist()


def caracteristicas_numero(numero):
    d = np.array([int(x) for x in numero], dtype=float)
    return [
        d.sum(),
        d.mean(),
        d.std(),
        len(set(numero)),
        sum(x % 2 == 0 for x in d),
        sum(x >= 5 for x in d),
        abs(d[0] - d[1]),
        abs(d[1] - d[2]),
        abs(d[2] - d[3]),
        int(d[0] == d[3]),
        int(d[1] == d[2]),
    ]


def crear_dataset(numeros, lags=10):
    X = []
    y = [[], [], [], []]

    for indice in range(lags, len(numeros)):
        historial = numeros[indice - lags:indice]
        fila = []

        # Dígitos de los últimos sorteos.
        for numero in historial:
            fila.extend(int(x) for x in numero)

        # Características de cada sorteo anterior.
        for numero in historial:
            fila.extend(caracteristicas_numero(numero))

        # Cambios entre sorteos consecutivos.
        for j in range(1, len(historial)):
            anterior = historial[j - 1]
            actual = historial[j]
            fila.extend(
                (int(actual[pos]) - int(anterior[pos])) % 10
                for pos in range(4)
            )

        X.append(fila)

        objetivo = numeros[indice]
        for posicion in range(4):
            y[posicion].append(int(objetivo[posicion]))

    return np.asarray(X, dtype=np.float32), [
        np.asarray(valores, dtype=np.int64)
        for valores in y
    ]


def crear_fila_prediccion(numeros, lags=10):
    historial = numeros[-lags:]
    fila = []

    for numero in historial:
        fila.extend(int(x) for x in numero)

    for numero in historial:
        fila.extend(caracteristicas_numero(numero))

    for j in range(1, len(historial)):
        anterior = historial[j - 1]
        actual = historial[j]
        fila.extend(
            (int(actual[pos]) - int(anterior[pos])) % 10
            for pos in range(4)
        )

    return np.asarray([fila], dtype=np.float32)


def crear_modelos(
    numeros,
    lags=10,
    max_entrenamiento=3000,
    arboles=150,
    profundidad=12,
    semilla=42,
):
    X, ys = crear_dataset(numeros, lags=lags)

    if len(X) < 200:
        raise ValueError("No hay suficientes sorteos para entrenar.")

    if len(X) > max_entrenamiento:
        X = X[-max_entrenamiento:]
        ys = [y[-max_entrenamiento:] for y in ys]

    modelos = []

    for posicion in range(4):
        modelo = RandomForestClassifier(
            n_estimators=arboles,
            max_depth=profundidad,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=semilla + posicion,
            n_jobs=-1,
        )
        modelo.fit(X, ys[posicion])
        modelos.append(modelo)

    return modelos


def probabilidades_posicion(modelos, numeros, lags):
    fila = crear_fila_prediccion(numeros, lags=lags)
    probabilidades = []

    for modelo in modelos:
        probs_modelo = modelo.predict_proba(fila)[0]
        probs = np.full(10, 1e-9, dtype=float)

        for clase, prob in zip(modelo.classes_, probs_modelo):
            probs[int(clase)] = float(prob)

        probs = probs / probs.sum()
        probabilidades.append(probs)

    return probabilidades


def ranking_combinaciones(probabilidades, top_n=100):
    filas = []

    for digitos in itertools.product(range(10), repeat=4):
        numero = "".join(str(x) for x in digitos)
        probs = [
            probabilidades[pos][digitos[pos]]
            for pos in range(4)
        ]

        # Producto geométrico para evitar que una posición domine demasiado.
        score = float(np.prod(probs) ** 0.25)

        filas.append({
            "Número": numero,
            "Score IA": score,
            "P1": probs[0],
            "P2": probs[1],
            "P3": probs[2],
            "P4": probs[3],
        })

    ranking = (
        pd.DataFrame(filas)
        .sort_values("Score IA", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    ranking.insert(0, "Ranking", range(1, len(ranking) + 1))

    for columna in ["Score IA", "P1", "P2", "P3", "P4"]:
        ranking[columna] = (ranking[columna] * 100).round(4)

    return ranking


def ejecutar_backtesting(
    numeros,
    fechas,
    pruebas,
    lags,
    max_entrenamiento,
    arboles,
    profundidad,
):
    inicio = len(numeros) - pruebas
    filas = []
    barra = st.progress(0)
    estado = st.empty()

    for contador, indice in enumerate(range(inicio, len(numeros)), start=1):
        entrenamiento = numeros[:indice]
        real = numeros[indice]

        modelos = crear_modelos(
            entrenamiento,
            lags=lags,
            max_entrenamiento=max_entrenamiento,
            arboles=arboles,
            profundidad=profundidad,
            semilla=1000 + indice,
        )
        probabilidades = probabilidades_posicion(
            modelos,
            entrenamiento,
            lags,
        )
        ranking = ranking_combinaciones(probabilidades, top_n=100)
        top100 = ranking["Número"].tolist()

        posicion = top100.index(real) + 1 if real in top100 else None

        prob_real = np.prod([
            probabilidades[pos][int(real[pos])]
            for pos in range(4)
        ]) ** 0.25

        filas.append({
            "Fecha": fechas[indice],
            "Resultado real": real,
            "Ranking": posicion if posicion else "Fuera del Top 100",
            "Score real": round(float(prob_real) * 100, 4),
            "Top 10": posicion is not None and posicion <= 10,
            "Top 20": posicion is not None and posicion <= 20,
            "Top 50": posicion is not None and posicion <= 50,
            "Top 100": posicion is not None,
        })

        barra.progress(contador / pruebas)
        estado.write(
            f"Entrenando y probando {contador} de {pruebas}: "
            f"{pd.to_datetime(fechas[indice]).strftime('%d/%m/%Y')}"
        )

    barra.empty()
    estado.empty()
    return pd.DataFrame(filas)


st.title("🧠 TRIS AI V1")
st.caption(
    "Random Forest temporal: aprende de secuencias anteriores y estima "
    "cada dígito de la siguiente Directa 4."
)

archivo_subido = st.sidebar.file_uploader(
    "Cargar tris_historico.csv",
    type=["csv"],
)

try:
    if archivo_subido is not None:
        datos = leer_csv(archivo_subido)
    elif LOCAL_CSV.exists():
        datos = leer_csv(LOCAL_CSV)
    else:
        datos = pd.DataFrame()
except Exception as error:
    st.error(f"No se pudo leer el archivo: {error}")
    st.stop()

if datos.empty:
    st.warning("Sube `tris_historico.csv` desde el panel lateral.")
    st.stop()

st.sidebar.success(f"{len(datos):,} resultados cargados")

with st.sidebar.expander("⚙️ Configuración IA", expanded=True):
    lags = st.slider(
        "Sorteos anteriores usados",
        5,
        30,
        10,
        1,
    )
    max_entrenamiento = st.slider(
        "Máximo de muestras de entrenamiento",
        500,
        4000,
        2500,
        250,
    )
    arboles = st.slider(
        "Árboles por modelo",
        50,
        300,
        120,
        10,
    )
    profundidad = st.slider(
        "Profundidad máxima",
        4,
        20,
        10,
        1,
    )

m1, m2, m3 = st.columns(3)
m1.metric("Resultados", f"{len(datos):,}")
m2.metric("Primera fecha", datos["fecha"].min().strftime("%d/%m/%Y"))
m3.metric("Última fecha", datos["fecha"].max().strftime("%d/%m/%Y"))

tab_prediccion, tab_backtest, tab_modelo, tab_base = st.tabs(
    ["🎯 Predicción IA", "🧪 Backtesting IA", "🔍 Modelo", "🗂️ Base"]
)

with tab_prediccion:
    c1, c2, c3 = st.columns(3)
    sorteo = c1.selectbox("Sorteo", SORTEOS)
    lado = c2.selectbox("Directa 4", ["Últimos 4", "Primeros 4"])
    top_n = c3.slider("Candidatos mostrados", 10, 100, 30, 10)

    filtrados = (
        datos[datos["sorteo"] == sorteo]
        .sort_values("fecha")
        .reset_index(drop=True)
    )
    numeros = preparar_directa4(filtrados, lado)

    if st.button(
        "🧠 Entrenar IA y generar ranking",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Entrenando cuatro modelos Random Forest..."):
            inicio = time.time()
            modelos = crear_modelos(
                numeros,
                lags=lags,
                max_entrenamiento=max_entrenamiento,
                arboles=arboles,
                profundidad=profundidad,
            )
            probabilidades = probabilidades_posicion(
                modelos,
                numeros,
                lags,
            )
            ranking = ranking_combinaciones(
                probabilidades,
                top_n=top_n,
            )
            duracion = time.time() - inicio

        st.session_state["ranking_ai"] = ranking
        st.session_state["probs_ai"] = probabilidades
        st.session_state["tiempo_ai"] = duracion

    if "ranking_ai" in st.session_state:
        ranking = st.session_state["ranking_ai"]
        mejor = ranking.iloc[0]

        p1, p2, p3 = st.columns(3)
        p1.metric("Mejor clasificado", mejor["Número"])
        p2.metric("Score IA", f'{mejor["Score IA"]:.4f}')
        p3.metric(
            "Tiempo de entrenamiento",
            f'{st.session_state.get("tiempo_ai", 0):.1f} s',
        )

        st.dataframe(
            ranking,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Descargar ranking IA",
            ranking.to_csv(index=False).encode("utf-8-sig"),
            file_name="ranking_tris_ai.csv",
            mime="text/csv",
        )

        st.info(
            "El Score IA no es una probabilidad garantizada. Es una puntuación "
            "derivada de las probabilidades estimadas para cada posición."
        )

with tab_backtest:
    st.subheader("Validación temporal de la IA")
    st.write(
        "Para cada prueba, el modelo solo ve resultados anteriores. "
        "Después se comprueba si el resultado real quedó en el Top 100."
    )

    b1, b2, b3 = st.columns(3)
    sorteo_bt = b1.selectbox("Sorteo", SORTEOS, key="bt_sorteo")
    lado_bt = b2.selectbox(
        "Directa 4",
        ["Últimos 4", "Primeros 4"],
        key="bt_lado",
    )
    pruebas = b3.slider(
        "Sorteos a probar",
        5,
        30,
        10,
        5,
        help="Random Forest tarda más. Empieza con 10.",
    )

    if st.button(
        "▶ Ejecutar backtesting IA",
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

        if len(numeros_bt) < max(300, pruebas + lags + 50):
            st.error("No hay suficientes resultados para entrenar.")
        else:
            inicio = time.time()
            resultado = ejecutar_backtesting(
                numeros_bt,
                fechas_bt,
                pruebas,
                lags,
                max_entrenamiento,
                arboles,
                profundidad,
            )
            st.session_state["bt_ai"] = resultado
            st.session_state["bt_ai_tiempo"] = time.time() - inicio

    if "bt_ai" in st.session_state:
        resultado = st.session_state["bt_ai"]
        total = len(resultado)

        a10 = int(resultado["Top 10"].sum())
        a20 = int(resultado["Top 20"].sum())
        a50 = int(resultado["Top 50"].sum())
        a100 = int(resultado["Top 100"].sum())

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Top 10", f"{a10}/{total}", f"{a10 / total * 100:.1f}%")
        q2.metric("Top 20", f"{a20}/{total}", f"{a20 / total * 100:.1f}%")
        q3.metric("Top 50", f"{a50}/{total}", f"{a50 / total * 100:.1f}%")
        q4.metric("Top 100", f"{a100}/{total}", f"{a100 / total * 100:.1f}%")

        st.caption(
            f'Tiempo total: {st.session_state.get("bt_ai_tiempo", 0):.1f} segundos. '
            "La referencia aleatoria para Top 100 es aproximadamente 1%."
        )

        if a100 / total > 0.01:
            st.success(
                "En esta muestra, la IA superó la referencia aleatoria del 1%."
            )
        else:
            st.warning(
                "En esta muestra, la IA no superó la referencia aleatoria del 1%."
            )

        st.dataframe(
            resultado.sort_values("Fecha", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Descargar backtesting IA",
            resultado.to_csv(index=False).encode("utf-8-sig"),
            file_name="backtesting_tris_ai.csv",
            mime="text/csv",
        )

with tab_modelo:
    st.subheader("Cómo funciona")
    st.write(
        "Se entrenan cuatro Random Forest independientes: uno para cada "
        "posición de Directa 4. Cada modelo recibe los últimos sorteos, "
        "sus dígitos, sumas, paridad, repeticiones y cambios entre posiciones."
    )
    st.write(
        "Después calcula 10 probabilidades por posición y combina las cuatro "
        "para ordenar las 10,000 combinaciones posibles."
    )
    st.warning(
        "Un sorteo bien diseñado debe comportarse como un proceso aleatorio. "
        "El aprendizaje automático puede encontrar correlaciones históricas, "
        "pero no garantiza que se repitan."
    )

with tab_base:
    st.dataframe(
        datos.sort_values("fecha", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
