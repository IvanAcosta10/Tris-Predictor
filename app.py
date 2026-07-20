
from pathlib import Path
import itertools
import time
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="TRIS Predictor VNext",
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


def rasgos_numero(numero):
    d = np.array([int(x) for x in numero], dtype=float)
    diffs = np.diff(d)
    moddiffs = (np.diff(d) % 10)

    return [
        *d.tolist(),
        d.sum(),
        d.mean(),
        d.std(),
        d.max(),
        d.min(),
        np.ptp(d),
        len(set(numero)),
        sum(x % 2 == 0 for x in d),
        sum(x >= 5 for x in d),
        sum(d[i] == d[j] for i in range(4) for j in range(i + 1, 4)),
        int(d[0] == d[3]),
        int(d[1] == d[2]),
        int(d[0] == d[1]),
        int(d[1] == d[2]),
        int(d[2] == d[3]),
        *np.abs(diffs).tolist(),
        *moddiffs.tolist(),
        sum(abs(diffs) == 1),
        sum(abs(diffs) <= 2),
        sum(diffs > 0),
        sum(diffs < 0),
        int(np.all(diffs >= 0)),
        int(np.all(diffs <= 0)),
        int(sum(d) % 2),
        int(sum(d) % 3),
        int(sum(d) % 5),
        int(int(numero) % 7),
        int(int(numero) % 9),
        int(numero[:2]),
        int(numero[2:]),
        abs(int(numero[:2]) - int(numero[2:])),
    ]


def frecuencia_ventana(historial, ventana):
    muestra = historial[-min(ventana, len(historial)):]
    salida = []

    for pos in range(4):
        conteo = np.bincount(
            [int(n[pos]) for n in muestra],
            minlength=10,
        ) / max(len(muestra), 1)
        salida.extend(conteo.tolist())

    return salida


def construir_fila(historial, lags):
    recientes = historial[-lags:]
    fila = []

    for numero in recientes:
        fila.extend(rasgos_numero(numero))

    # Cambios entre sorteos.
    for i in range(1, len(recientes)):
        anterior = recientes[i - 1]
        actual = recientes[i]

        for pos in range(4):
            a = int(anterior[pos])
            b = int(actual[pos])
            fila.extend([
                (b - a) % 10,
                abs(b - a),
                int(a == b),
                int(b > a),
            ])

        fila.extend([
            abs(int(actual) - int(anterior)) % 10000,
            int(actual[:2] == anterior[:2]),
            int(actual[2:] == anterior[2:]),
            len(set(actual) & set(anterior)),
        ])

    # Frecuencias en varias escalas temporales.
    for ventana in [10, 25, 50, 100, 200]:
        fila.extend(frecuencia_ventana(historial, ventana))

    # Frecuencia de pares por posición en ventana reciente.
    muestra = historial[-min(200, len(historial)):]
    for pos in range(3):
        conteo = np.zeros(100, dtype=float)
        for n in muestra:
            conteo[int(n[pos:pos + 2])] += 1
        conteo /= max(len(muestra), 1)
        fila.extend(conteo.tolist())

    # Ausencia de dígitos por posición.
    for pos in range(4):
        for digito in range(10):
            distancia = len(historial)
            for offset, n in enumerate(reversed(historial)):
                if int(n[pos]) == digito:
                    distancia = offset
                    break
            fila.append(min(distancia, 300) / 300)

    return np.asarray(fila, dtype=np.float32)


def crear_dataset(numeros, lags):
    X = []
    ys = [[], [], [], []]

    for i in range(lags, len(numeros)):
        historial = numeros[:i]
        X.append(construir_fila(historial, lags))
        objetivo = numeros[i]

        for pos in range(4):
            ys[pos].append(int(objetivo[pos]))

    return (
        np.asarray(X, dtype=np.float32),
        [np.asarray(y, dtype=np.int64) for y in ys],
    )


def crear_modelos(arboles, profundidad, semilla):
    return [
        RandomForestClassifier(
            n_estimators=arboles,
            max_depth=profundidad,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=semilla,
            n_jobs=-1,
        ),
        ExtraTreesClassifier(
            n_estimators=arboles,
            max_depth=profundidad,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=semilla + 100,
            n_jobs=-1,
        ),
        HistGradientBoostingClassifier(
            max_depth=max(3, profundidad // 2),
            learning_rate=0.05,
            max_iter=150,
            l2_regularization=1.0,
            random_state=semilla + 200,
        ),
        Pipeline([
            ("selector", SelectKBest(mutual_info_classif, k=120)),
            ("scaler", StandardScaler()),
            ("modelo", LogisticRegression(
                max_iter=600,
                class_weight="balanced",
                C=0.5,
                random_state=semilla + 300,
            )),
        ]),
    ]


def entrenar_ensamble(
    numeros,
    lags,
    max_muestras,
    arboles,
    profundidad,
    semilla,
):
    X, ys = crear_dataset(numeros, lags)

    if len(X) < 300:
        raise ValueError("No hay suficientes datos para entrenar.")

    if len(X) > max_muestras:
        X = X[-max_muestras:]
        ys = [y[-max_muestras:] for y in ys]

    modelos_por_posicion = []

    for pos in range(4):
        modelos = crear_modelos(arboles, profundidad, semilla + pos * 1000)
        entrenados = []

        for modelo in modelos:
            try:
                modelo.fit(X, ys[pos])
                entrenados.append(modelo)
            except Exception:
                pass

        if not entrenados:
            raise RuntimeError(f"No se pudo entrenar la posición {pos + 1}.")

        modelos_por_posicion.append(entrenados)

    return modelos_por_posicion


def probabilidades_modelo(modelo, fila):
    probs = np.full(10, 1e-9, dtype=float)
    pred = modelo.predict_proba(fila)[0]
    clases = modelo.classes_

    for clase, prob in zip(clases, pred):
        probs[int(clase)] = float(prob)

    probs /= probs.sum()
    return probs


def predecir_probabilidades(modelos_por_posicion, numeros, lags):
    fila = construir_fila(numeros, lags).reshape(1, -1)
    salida = []

    for modelos in modelos_por_posicion:
        probabilidades = [
            probabilidades_modelo(modelo, fila)
            for modelo in modelos
        ]
        promedio = np.mean(probabilidades, axis=0)
        promedio /= promedio.sum()
        salida.append(promedio)

    return salida


def ranking_combinaciones(probabilidades, top_n):
    filas = []

    for digitos in itertools.product(range(10), repeat=4):
        numero = "".join(str(x) for x in digitos)
        probs = np.array([
            probabilidades[pos][digitos[pos]]
            for pos in range(4)
        ])

        score_geo = float(np.prod(probs) ** 0.25)
        score_min = float(np.min(probs))
        score_prom = float(np.mean(probs))

        score_final = (
            0.60 * score_geo
            + 0.25 * score_prom
            + 0.15 * score_min
        )

        filas.append({
            "Número": numero,
            "Score final": score_final,
            "Score geométrico": score_geo,
            "Probabilidad mínima": score_min,
            "P1": probs[0],
            "P2": probs[1],
            "P3": probs[2],
            "P4": probs[3],
        })

    ranking = (
        pd.DataFrame(filas)
        .sort_values("Score final", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    ranking.insert(0, "Ranking", range(1, len(ranking) + 1))

    for col in [
        "Score final",
        "Score geométrico",
        "Probabilidad mínima",
        "P1",
        "P2",
        "P3",
        "P4",
    ]:
        ranking[col] = (ranking[col] * 100).round(4)

    return ranking


def ejecutar_backtesting(
    numeros,
    fechas,
    pruebas,
    lags,
    max_muestras,
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

        modelos = entrenar_ensamble(
            entrenamiento,
            lags,
            max_muestras,
            arboles,
            profundidad,
            semilla=5000 + indice,
        )

        probs = predecir_probabilidades(modelos, entrenamiento, lags)
        ranking = ranking_combinaciones(probs, 100)

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
            f"Entrenando {contador} de {pruebas}: "
            f"{pd.to_datetime(fechas[indice]).strftime('%d/%m/%Y')}"
        )

    barra.empty()
    estado.empty()
    return pd.DataFrame(filas)


st.title("🧠 TRIS Predictor VNext")
st.caption(
    "Motor ampliado de características + ensamble de cuatro modelos."
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
    lags = st.slider("Sorteos anteriores usados", 5, 20, 10, 1)
    max_muestras = st.slider(
        "Muestras máximas",
        500,
        3500,
        2000,
        250,
    )
    arboles = st.slider("Árboles", 50, 200, 90, 10)
    profundidad = st.slider("Profundidad", 4, 16, 9, 1)

m1, m2, m3 = st.columns(3)
m1.metric("Resultados", f"{len(datos):,}")
m2.metric("Primera fecha", datos["fecha"].min().strftime("%d/%m/%Y"))
m3.metric("Última fecha", datos["fecha"].max().strftime("%d/%m/%Y"))

tab_pred, tab_bt, tab_info, tab_base = st.tabs(
    ["🎯 Predicción", "🧪 Backtesting", "🔍 Motor", "🗂️ Base"]
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
        "🧠 Entrenar VNext",
        type="primary",
        use_container_width=True,
    ):
        inicio = time.time()

        with st.spinner("Construyendo variables y entrenando ensamble..."):
            modelos = entrenar_ensamble(
                numeros,
                lags,
                max_muestras,
                arboles,
                profundidad,
                semilla=42,
            )
            probs = predecir_probabilidades(modelos, numeros, lags)
            ranking = ranking_combinaciones(probs, top_n)

        st.session_state["ranking_vnext"] = ranking
        st.session_state["tiempo_vnext"] = time.time() - inicio

    if "ranking_vnext" in st.session_state:
        ranking = st.session_state["ranking_vnext"]
        mejor = ranking.iloc[0]

        r1, r2, r3 = st.columns(3)
        r1.metric("Mejor clasificado", mejor["Número"])
        r2.metric("Score final", f'{mejor["Score final"]:.4f}')
        r3.metric(
            "Tiempo",
            f'{st.session_state.get("tiempo_vnext", 0):.1f} s',
        )

        st.dataframe(ranking, use_container_width=True, hide_index=True)

        st.download_button(
            "Descargar ranking",
            ranking.to_csv(index=False).encode("utf-8-sig"),
            file_name="ranking_vnext.csv",
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
        20,
        5,
        5,
        help="Esta versión tarda más. Empieza con 5.",
    )

    if st.button(
        "▶ Ejecutar backtesting VNext",
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
        )

        st.session_state["bt_vnext"] = resultado
        st.session_state["bt_vnext_tiempo"] = time.time() - inicio

    if "bt_vnext" in st.session_state:
        resultado = st.session_state["bt_vnext"]
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
            f'Tiempo: {st.session_state.get("bt_vnext_tiempo", 0):.1f} s. '
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

with tab_info:
    st.write(
        "Esta versión construye cientos de características: frecuencias en "
        "múltiples ventanas, pares por posición, ausencias, diferencias, "
        "módulos, simetrías, sumas, rango y cambios entre sorteos."
    )
    st.write(
        "Después entrena Random Forest, Extra Trees, HistGradientBoosting y "
        "Regresión Logística con selección automática de variables."
    )
    st.warning(
        "El backtesting sigue siendo la única medida válida. Si no mejora, "
        "significa que el histórico no contiene una señal explotable con este enfoque."
    )

with tab_base:
    st.dataframe(
        datos.sort_values("fecha", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
