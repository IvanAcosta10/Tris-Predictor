
from pathlib import Path
import itertools
import time

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier

st.set_page_config(
    page_title="TRIS Predictor AI V2",
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


def perfil(numero):
    d = [int(x) for x in numero]
    return {
        "pares": sum(x % 2 == 0 for x in d),
        "altos": sum(x >= 5 for x in d),
        "repetidos": 4 - len(set(numero)),
        "rango_suma": min(sum(d) // 5, 7),
        "consecutivos": min(
            sum(abs(d[i] - d[i + 1]) == 1 for i in range(3)),
            2,
        ),
    }


def features_numero(numero):
    d = np.array([int(x) for x in numero], dtype=float)
    p = perfil(numero)
    return [
        *d.tolist(),
        d.sum(),
        d.mean(),
        d.std(),
        d.max(),
        d.min(),
        len(set(numero)),
        p["pares"],
        p["altos"],
        p["repetidos"],
        p["rango_suma"],
        p["consecutivos"],
        int(d[0] == d[3]),
        int(d[1] == d[2]),
    ]


def crear_dataset(numeros, lags):
    X = []
    y_digitos = [[], [], [], []]
    y_perfiles = []

    for i in range(lags, len(numeros)):
        historial = numeros[i - lags:i]
        fila = []

        for numero in historial:
            fila.extend(features_numero(numero))

        for j in range(1, len(historial)):
            anterior = historial[j - 1]
            actual = historial[j]
            fila.extend(
                (int(actual[pos]) - int(anterior[pos])) % 10
                for pos in range(4)
            )
            fila.extend(
                int(actual[pos] == anterior[pos])
                for pos in range(4)
            )

        # Frecuencias recientes por posición.
        for pos in range(4):
            conteo = np.bincount(
                [int(n[pos]) for n in historial],
                minlength=10,
            ) / len(historial)
            fila.extend(conteo.tolist())

        X.append(fila)

        objetivo = numeros[i]
        for pos in range(4):
            y_digitos[pos].append(int(objetivo[pos]))

        p = perfil(objetivo)
        y_perfiles.append([
            p["pares"],
            p["altos"],
            p["repetidos"],
            p["rango_suma"],
            p["consecutivos"],
        ])

    return (
        np.asarray(X, dtype=np.float32),
        [np.asarray(y, dtype=np.int64) for y in y_digitos],
        np.asarray(y_perfiles, dtype=np.int64),
    )


def fila_prediccion(numeros, lags):
    historial = numeros[-lags:]
    fila = []

    for numero in historial:
        fila.extend(features_numero(numero))

    for j in range(1, len(historial)):
        anterior = historial[j - 1]
        actual = historial[j]
        fila.extend(
            (int(actual[pos]) - int(anterior[pos])) % 10
            for pos in range(4)
        )
        fila.extend(
            int(actual[pos] == anterior[pos])
            for pos in range(4)
        )

    for pos in range(4):
        conteo = np.bincount(
            [int(n[pos]) for n in historial],
            minlength=10,
        ) / len(historial)
        fila.extend(conteo.tolist())

    return np.asarray([fila], dtype=np.float32)


def entrenar_modelos(
    numeros,
    lags,
    max_muestras,
    arboles,
    profundidad,
    semilla,
):
    X, ys_digitos, y_perfiles = crear_dataset(numeros, lags)

    if len(X) < 250:
        raise ValueError("No hay suficientes datos para entrenar.")

    if len(X) > max_muestras:
        X = X[-max_muestras:]
        ys_digitos = [y[-max_muestras:] for y in ys_digitos]
        y_perfiles = y_perfiles[-max_muestras:]

    modelos_rf = []
    modelos_et = []

    for pos in range(4):
        rf = RandomForestClassifier(
            n_estimators=arboles,
            max_depth=profundidad,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=semilla + pos,
            n_jobs=-1,
        )
        et = ExtraTreesClassifier(
            n_estimators=arboles,
            max_depth=profundidad,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=semilla + 100 + pos,
            n_jobs=-1,
        )
        rf.fit(X, ys_digitos[pos])
        et.fit(X, ys_digitos[pos])
        modelos_rf.append(rf)
        modelos_et.append(et)

    perfil_base = RandomForestClassifier(
        n_estimators=arboles,
        max_depth=profundidad,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=semilla + 500,
        n_jobs=-1,
    )
    modelo_perfil = MultiOutputClassifier(perfil_base, n_jobs=-1)
    modelo_perfil.fit(X, y_perfiles)

    return modelos_rf, modelos_et, modelo_perfil


def probs_clases(modelo, fila, total_clases):
    salida = np.full(total_clases, 1e-9, dtype=float)
    probs = modelo.predict_proba(fila)[0]

    for clase, prob in zip(modelo.classes_, probs):
        salida[int(clase)] = float(prob)

    salida /= salida.sum()
    return salida


def obtener_predicciones(
    modelos_rf,
    modelos_et,
    modelo_perfil,
    numeros,
    lags,
):
    fila = fila_prediccion(numeros, lags)

    probs_posiciones = []
    for pos in range(4):
        rf = probs_clases(modelos_rf[pos], fila, 10)
        et = probs_clases(modelos_et[pos], fila, 10)
        probs_posiciones.append(0.5 * rf + 0.5 * et)

    probs_perfil = []
    tamanos = [5, 5, 4, 8, 3]

    for estimador, tamano in zip(modelo_perfil.estimators_, tamanos):
        probs_perfil.append(probs_clases(estimador, fila, tamano))

    return probs_posiciones, probs_perfil


def score_perfil(numero, probs_perfil):
    p = perfil(numero)
    valores = [
        p["pares"],
        p["altos"],
        p["repetidos"],
        p["rango_suma"],
        p["consecutivos"],
    ]
    return float(np.mean([
        probs_perfil[i][valores[i]]
        for i in range(len(valores))
    ]))


def ranking_hibrido(
    probs_posiciones,
    probs_perfil,
    top_n,
    supervivencia,
):
    filas = []

    for digitos in itertools.product(range(10), repeat=4):
        numero = "".join(str(x) for x in digitos)

        score_digitos = float(np.prod([
            probs_posiciones[pos][digitos[pos]]
            for pos in range(4)
        ]) ** 0.25)

        score_p = score_perfil(numero, probs_perfil)

        filas.append({
            "Número": numero,
            "Score dígitos": score_digitos,
            "Score perfil": score_p,
        })

    df = pd.DataFrame(filas)

    sobrevivientes = max(
        top_n,
        int(10000 * supervivencia / 100),
    )

    # Primero descarta por perfil.
    df = (
        df.sort_values("Score perfil", ascending=False)
        .head(sobrevivientes)
        .copy()
    )

    # Después combina ambos modelos.
    df["Score final"] = (
        0.68 * df["Score dígitos"]
        + 0.32 * df["Score perfil"]
    )

    ranking = (
        df.sort_values("Score final", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    ranking.insert(0, "Ranking", range(1, len(ranking) + 1))

    for columna in ["Score dígitos", "Score perfil", "Score final"]:
        ranking[columna] = (ranking[columna] * 100).round(4)

    return ranking


def ejecutar_backtesting(
    numeros,
    fechas,
    pruebas,
    lags,
    max_muestras,
    arboles,
    profundidad,
    supervivencia,
):
    inicio = len(numeros) - pruebas
    filas = []
    barra = st.progress(0)
    estado = st.empty()

    for contador, indice in enumerate(range(inicio, len(numeros)), start=1):
        entrenamiento = numeros[:indice]
        real = numeros[indice]

        modelos_rf, modelos_et, modelo_perfil = entrenar_modelos(
            entrenamiento,
            lags,
            max_muestras,
            arboles,
            profundidad,
            semilla=2000 + indice,
        )

        probs_posiciones, probs_perfil = obtener_predicciones(
            modelos_rf,
            modelos_et,
            modelo_perfil,
            entrenamiento,
            lags,
        )

        ranking = ranking_hibrido(
            probs_posiciones,
            probs_perfil,
            top_n=100,
            supervivencia=supervivencia,
        )

        top100 = ranking["Número"].tolist()
        posicion = top100.index(real) + 1 if real in top100 else None

        filas.append({
            "Fecha": fechas[indice],
            "Resultado real": real,
            "Ranking": posicion if posicion else "Fuera del Top 100",
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


st.title("🧠 TRIS Predictor AI V2")
st.caption(
    "Ensamble de Random Forest + Extra Trees + predictor de perfiles."
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

with st.sidebar.expander("⚙️ Configuración", expanded=True):
    lags = st.slider("Sorteos anteriores usados", 5, 25, 12, 1)
    max_muestras = st.slider(
        "Muestras máximas",
        500,
        4000,
        2500,
        250,
    )
    arboles = st.slider("Árboles por modelo", 50, 250, 100, 10)
    profundidad = st.slider("Profundidad máxima", 4, 18, 10, 1)
    supervivencia = st.slider(
        "Porcentaje que sobrevive al filtro",
        1,
        20,
        5,
        1,
    )

m1, m2, m3 = st.columns(3)
m1.metric("Resultados", f"{len(datos):,}")
m2.metric("Primera fecha", datos["fecha"].min().strftime("%d/%m/%Y"))
m3.metric("Última fecha", datos["fecha"].max().strftime("%d/%m/%Y"))

tab_pred, tab_bt, tab_explicacion, tab_base = st.tabs(
    ["🎯 Predicción", "🧪 Backtesting", "🔍 Cómo funciona", "🗂️ Base"]
)

with tab_pred:
    c1, c2, c3 = st.columns(3)
    sorteo = c1.selectbox("Sorteo", SORTEOS)
    lado = c2.selectbox("Directa 4", ["Últimos 4", "Primeros 4"])
    top_n = c3.slider("Candidatos", 10, 100, 30, 10)

    filtrados = (
        datos[datos["sorteo"] == sorteo]
        .sort_values("fecha")
        .reset_index(drop=True)
    )
    numeros = preparar_directa4(filtrados, lado)

    if st.button(
        "🧠 Entrenar y generar ranking",
        type="primary",
        use_container_width=True,
    ):
        inicio = time.time()

        with st.spinner("Entrenando ensamble y predictor de perfiles..."):
            modelos_rf, modelos_et, modelo_perfil = entrenar_modelos(
                numeros,
                lags,
                max_muestras,
                arboles,
                profundidad,
                semilla=42,
            )
            probs_posiciones, probs_perfil = obtener_predicciones(
                modelos_rf,
                modelos_et,
                modelo_perfil,
                numeros,
                lags,
            )
            ranking = ranking_hibrido(
                probs_posiciones,
                probs_perfil,
                top_n,
                supervivencia,
            )

        st.session_state["ranking_ai_v2"] = ranking
        st.session_state["tiempo_ai_v2"] = time.time() - inicio

    if "ranking_ai_v2" in st.session_state:
        ranking = st.session_state["ranking_ai_v2"]
        mejor = ranking.iloc[0]

        r1, r2, r3 = st.columns(3)
        r1.metric("Mejor clasificado", mejor["Número"])
        r2.metric("Score final", f'{mejor["Score final"]:.4f}')
        r3.metric(
            "Tiempo",
            f'{st.session_state.get("tiempo_ai_v2", 0):.1f} s',
        )

        st.dataframe(ranking, use_container_width=True, hide_index=True)

        st.download_button(
            "Descargar ranking",
            ranking.to_csv(index=False).encode("utf-8-sig"),
            file_name="ranking_tris_ai_v2.csv",
            mime="text/csv",
        )

with tab_bt:
    st.subheader("Backtesting temporal")

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

        inicio = time.time()
        resultado = ejecutar_backtesting(
            numeros_bt,
            fechas_bt,
            pruebas,
            lags,
            max_muestras,
            arboles,
            profundidad,
            supervivencia,
        )
        st.session_state["bt_ai_v2"] = resultado
        st.session_state["bt_ai_v2_tiempo"] = time.time() - inicio

    if "bt_ai_v2" in st.session_state:
        resultado = st.session_state["bt_ai_v2"]
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
            f'Tiempo: {st.session_state.get("bt_ai_v2_tiempo", 0):.1f} s. '
            "Referencia aleatoria Top 100: aproximadamente 1%."
        )

        if a100 / total > 0.01:
            st.success("Superó la referencia aleatoria en esta muestra.")
        else:
            st.warning("No superó la referencia aleatoria en esta muestra.")

        st.dataframe(
            resultado.sort_values("Fecha", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

with tab_explicacion:
    st.write(
        "La V2 usa dos modelos distintos para cada posición: Random Forest "
        "y Extra Trees. Sus probabilidades se promedian."
    )
    st.write(
        "Además entrena un modelo separado para predecir el perfil: cantidad "
        "de pares, dígitos altos, repeticiones, rango de suma y consecutivos."
    )
    st.write(
        "Primero sobreviven las combinaciones con perfil más compatible y "
        "después se ordenan combinando el score de dígitos y el score del perfil."
    )
    st.warning(
        "El backtesting es el criterio principal. Una mejora visual o un score "
        "alto no significa que el modelo tenga poder predictivo."
    )

with tab_base:
    st.dataframe(
        datos.sort_values("fecha", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
