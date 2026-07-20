
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="TRIS Predictor V7",
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


def normalizar_diccionario(diccionario, claves):
    valores = np.array([diccionario.get(k, 0.0) for k in claves], dtype=float)

    if valores.max() == valores.min():
        return {k: 0.5 for k in claves}

    valores = (valores - valores.min()) / (valores.max() - valores.min())
    return {k: float(v) for k, v in zip(claves, valores)}


def crear_modelo(numeros, ventana_reciente, peso_recencia):
    cantidad = len(numeros)
    pesos_temporales = np.ones(cantidad, dtype=float)
    ventana = min(ventana_reciente, cantidad)

    if ventana > 0:
        pesos_temporales[-ventana:] = np.linspace(
            1.0,
            1.0 + peso_recencia,
            ventana,
        )

    digitos = [str(i) for i in range(10)]
    pares = [f"{i:02d}" for i in range(100)]

    frecuencia_global = Counter()
    frecuencia_posicion = [Counter() for _ in range(4)]
    frecuencia_pares = [Counter() for _ in range(3)]
    transiciones = [defaultdict(Counter) for _ in range(3)]

    # Aprende cómo cambia cada posición de un sorteo al siguiente.
    siguiente_por_digito = [defaultdict(Counter) for _ in range(4)]
    delta_por_posicion = [Counter() for _ in range(4)]
    coincidencias_con_anterior = Counter()

    ultima_aparicion_numero = {}
    ultima_aparicion_digito_posicion = [
        {d: None for d in digitos}
        for _ in range(4)
    ]

    for indice, (numero, peso) in enumerate(zip(numeros, pesos_temporales)):
        ultima_aparicion_numero[numero] = indice

        if indice > 0:
            anterior = numeros[indice - 1]
            coincidencias = 0

            for posicion in range(4):
                digito_anterior = anterior[posicion]
                digito_actual = numero[posicion]

                siguiente_por_digito[posicion][digito_anterior][digito_actual] += peso

                delta = (
                    int(digito_actual) - int(digito_anterior)
                ) % 10
                delta_por_posicion[posicion][str(delta)] += peso

                if digito_actual == digito_anterior:
                    coincidencias += 1

            coincidencias_con_anterior[str(coincidencias)] += peso

        for posicion, digito in enumerate(numero):
            frecuencia_global[digito] += peso
            frecuencia_posicion[posicion][digito] += peso
            ultima_aparicion_digito_posicion[posicion][digito] = indice

        for posicion in range(3):
            par = numero[posicion:posicion + 2]
            frecuencia_pares[posicion][par] += peso
            transiciones[posicion][numero[posicion]][numero[posicion + 1]] += peso

    global_norm = normalizar_diccionario(frecuencia_global, digitos)
    posicion_norm = [
        normalizar_diccionario(contador, digitos)
        for contador in frecuencia_posicion
    ]
    pares_norm = [
        normalizar_diccionario(contador, pares)
        for contador in frecuencia_pares
    ]

    transiciones_norm = []
    for posicion in range(3):
        por_digito = {}
        for digito in digitos:
            por_digito[digito] = normalizar_diccionario(
                transiciones[posicion][digito],
                digitos,
            )
        transiciones_norm.append(por_digito)

    # Ausencia de cada dígito por posición.
    ausencia_posicion = []
    for posicion in range(4):
        ausencias = {}
        for digito in digitos:
            ultima = ultima_aparicion_digito_posicion[posicion][digito]
            ausencias[digito] = cantidad if ultima is None else cantidad - 1 - ultima
        ausencia_posicion.append(
            normalizar_diccionario(ausencias, digitos)
        )

    siguiente_norm = []
    for posicion in range(4):
        por_digito = {}
        for digito in digitos:
            por_digito[digito] = normalizar_diccionario(
                siguiente_por_digito[posicion][digito],
                digitos,
            )
        siguiente_norm.append(por_digito)

    delta_norm = [
        normalizar_diccionario(contador, digitos)
        for contador in delta_por_posicion
    ]

    coincidencias_norm = normalizar_diccionario(
        coincidencias_con_anterior,
        [str(i) for i in range(5)],
    )

    ultimo_numero = numeros[-1] if numeros else "0000"

    return {
        "global": global_norm,
        "posicion": posicion_norm,
        "pares": pares_norm,
        "transiciones": transiciones_norm,
        "ausencia_posicion": ausencia_posicion,
        "ultima_aparicion_numero": ultima_aparicion_numero,
        "cantidad": cantidad,
        "ultimo_numero": ultimo_numero,
        "siguiente_por_digito": siguiente_norm,
        "delta_por_posicion": delta_norm,
        "coincidencias_con_anterior": coincidencias_norm,
    }


def calcular_componentes(numero, modelo):
    d = list(numero)

    posicion = sum(
        modelo["posicion"][pos][d[pos]]
        for pos in range(4)
    ) / 4

    frecuencia = sum(
        modelo["global"][digito]
        for digito in d
    ) / 4

    pares = sum(
        modelo["pares"][pos][numero[pos:pos + 2]]
        for pos in range(3)
    ) / 3

    transiciones = sum(
        modelo["transiciones"][pos][d[pos]][d[pos + 1]]
        for pos in range(3)
    ) / 3

    ausencia = sum(
        modelo["ausencia_posicion"][pos][d[pos]]
        for pos in range(4)
    ) / 4

    repeticion = (4 - len(set(d))) / 3

    espejo = (
        int(d[0] == d[3]) +
        int(d[1] == d[2])
    ) / 2

    ultimo = modelo["ultimo_numero"]
    coincidencias = sum(
        int(d[pos] == ultimo[pos])
        for pos in range(4)
    )
    similitud_ultimo = coincidencias / 4

    # Probabilidad condicional empírica:
    # dado el dígito anterior en cada posición, qué dígito tendió a seguir.
    influencia_ultimo = sum(
        modelo["siguiente_por_digito"][pos][ultimo[pos]][d[pos]]
        for pos in range(4)
    ) / 4

    # Cambios modulares observados, por ejemplo 7 -> 2 equivale a +5 módulo 10.
    patron_delta = sum(
        modelo["delta_por_posicion"][pos][
            str((int(d[pos]) - int(ultimo[pos])) % 10)
        ]
        for pos in range(4)
    ) / 4

    # Cantidad histórica de posiciones que suelen conservarse entre sorteos.
    patron_coincidencias = modelo["coincidencias_con_anterior"][
        str(coincidencias)
    ]

    ultima_aparicion = modelo["ultima_aparicion_numero"].get(numero)
    if ultima_aparicion is None:
        ciclo_numero = 1.0
    else:
        distancia = modelo["cantidad"] - 1 - ultima_aparicion
        ciclo_numero = min(distancia / 500, 1.0)

    suma = sum(int(x) for x in d)
    equilibrio_suma = 1.0 - min(abs(suma - 18) / 18, 1.0)

    pares_impares = sum(int(x) % 2 for x in d)
    equilibrio_paridad = 1.0 if pares_impares == 2 else 0.5 if pares_impares in (1, 3) else 0.0

    return {
        "Posición": posicion,
        "Frecuencia": frecuencia,
        "Pares": pares,
        "Transiciones": transiciones,
        "Ausencia": ausencia,
        "Ciclo": ciclo_numero,
        "Repetición": repeticion,
        "Espejo": espejo,
        "Similitud último": similitud_ultimo,
        "Influencia último": influencia_ultimo,
        "Patrón delta": patron_delta,
        "Coincidencias históricas": patron_coincidencias,
        "Suma": equilibrio_suma,
        "Paridad": equilibrio_paridad,
    }


def puntuar_componentes(componentes, pesos):
    total_pesos = sum(pesos.values())
    if total_pesos <= 0:
        return 0.0

    puntuacion = sum(
        componentes[nombre] * peso
        for nombre, peso in pesos.items()
    )
    return puntuacion / total_pesos



def perfil_numero(numero):
    digitos = [int(x) for x in numero]
    pares = sum(d % 2 == 0 for d in digitos)
    impares = 4 - pares
    suma = sum(digitos)
    repetidos = 4 - len(set(numero))
    empieza_cero = numero[0] == "0"
    consecutivos = sum(
        abs(digitos[i] - digitos[i + 1]) == 1
        for i in range(3)
    )
    altos = sum(d >= 5 for d in digitos)
    bajos = 4 - altos

    return {
        "pares": pares,
        "impares": impares,
        "suma": suma,
        "repetidos": repetidos,
        "empieza_cero": empieza_cero,
        "consecutivos": consecutivos,
        "altos": altos,
        "bajos": bajos,
    }


def aprender_perfil(numeros, ventana_perfil=200):
    muestra = numeros[-min(ventana_perfil, len(numeros)):]
    perfiles = [perfil_numero(numero) for numero in muestra]

    if not perfiles:
        return None

    df = pd.DataFrame(perfiles)

    conteo_pares = df["pares"].value_counts(normalize=True).to_dict()
    conteo_repetidos = df["repetidos"].value_counts(normalize=True).to_dict()
    conteo_altos = df["altos"].value_counts(normalize=True).to_dict()
    conteo_consecutivos = df["consecutivos"].value_counts(normalize=True).to_dict()

    suma_media = float(df["suma"].mean())
    suma_std = float(df["suma"].std(ddof=0))
    if suma_std == 0:
        suma_std = 1.0

    prob_cero = float(df["empieza_cero"].mean())

    return {
        "pares": conteo_pares,
        "repetidos": conteo_repetidos,
        "altos": conteo_altos,
        "consecutivos": conteo_consecutivos,
        "suma_media": suma_media,
        "suma_std": suma_std,
        "prob_cero": prob_cero,
    }


def score_perfil(numero, modelo_perfil):
    perfil = perfil_numero(numero)

    score_pares = modelo_perfil["pares"].get(perfil["pares"], 0.0)
    score_repetidos = modelo_perfil["repetidos"].get(perfil["repetidos"], 0.0)
    score_altos = modelo_perfil["altos"].get(perfil["altos"], 0.0)
    score_consecutivos = modelo_perfil["consecutivos"].get(perfil["consecutivos"], 0.0)

    z = abs(perfil["suma"] - modelo_perfil["suma_media"]) / modelo_perfil["suma_std"]
    score_suma = max(0.0, 1.0 - min(z / 3.0, 1.0))

    score_cero = (
        modelo_perfil["prob_cero"]
        if perfil["empieza_cero"]
        else 1.0 - modelo_perfil["prob_cero"]
    )

    total = (
        0.24 * score_pares
        + 0.22 * score_repetidos
        + 0.18 * score_altos
        + 0.10 * score_consecutivos
        + 0.20 * score_suma
        + 0.06 * score_cero
    )

    return total, perfil


def calcular_ranking(
    numeros,
    top_n,
    ventana_reciente,
    peso_recencia,
    pesos,
    excluir_vistos=False,
    usar_filtro_perfil=True,
    ventana_perfil=200,
    porcentaje_supervivencia=5,
):
    if len(numeros) < 10:
        return pd.DataFrame()

    modelo = crear_modelo(
        numeros,
        ventana_reciente=ventana_reciente,
        peso_recencia=peso_recencia,
    )

    vistos = set(numeros)
    filas = []

    modelo_perfil = aprender_perfil(
        numeros,
        ventana_perfil=ventana_perfil,
    )

    for valor in range(10000):
        numero = f"{valor:04d}"

        if excluir_vistos and numero in vistos:
            continue

        componentes = calcular_componentes(numero, modelo)
        puntuacion_base = puntuar_componentes(componentes, pesos)

        if usar_filtro_perfil and modelo_perfil is not None:
            perfil_score, perfil = score_perfil(numero, modelo_perfil)
        else:
            perfil_score = 1.0
            perfil = perfil_numero(numero)

        puntuacion = (
            0.55 * puntuacion_base
            + 0.45 * perfil_score
        )

        filas.append(
            {
                "Número": numero,
                "Puntuación": puntuacion,
                "Score perfil": perfil_score,
                "Pares perfil": perfil["pares"],
                "Suma perfil": perfil["suma"],
                "Repetidos perfil": perfil["repetidos"],
                **componentes,
            }
        )

    df_filas = pd.DataFrame(filas)

    if usar_filtro_perfil and not df_filas.empty:
        limite = max(
            top_n,
            int(len(df_filas) * porcentaje_supervivencia / 100)
        )
        df_filas = (
            df_filas
            .sort_values("Score perfil", ascending=False)
            .head(limite)
        )

    ranking = (
        df_filas
        .sort_values("Puntuación", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    ranking.insert(0, "Ranking", range(1, len(ranking) + 1))

    columnas_score = [
        "Puntuación",
        "Posición",
        "Frecuencia",
        "Pares",
        "Transiciones",
        "Ausencia",
        "Ciclo",
        "Repetición",
        "Espejo",
        "Similitud último",
        "Influencia último",
        "Patrón delta",
        "Coincidencias históricas",
        "Suma",
        "Paridad",
        "Score perfil",
    ]

    for columna in columnas_score:
        ranking[columna] = (ranking[columna] * 100).round(2)

    return ranking


def ejecutar_backtesting(
    numeros,
    fechas,
    cantidad_pruebas,
    ventana,
    recencia,
    pesos,
    usar_filtro_perfil,
    ventana_perfil,
    porcentaje_supervivencia,
):
    inicio = len(numeros) - cantidad_pruebas
    filas = []
    barra = st.progress(0)
    estado = st.empty()

    for contador, indice in enumerate(
        range(inicio, len(numeros)),
        start=1,
    ):
        entrenamiento = numeros[:indice]
        real = numeros[indice]

        ranking = calcular_ranking(
            entrenamiento,
            top_n=100,
            ventana_reciente=ventana,
            peso_recencia=recencia,
            pesos=pesos,
            excluir_vistos=False,
            usar_filtro_perfil=usar_filtro_perfil,
            ventana_perfil=ventana_perfil,
            porcentaje_supervivencia=porcentaje_supervivencia,
        )

        top100 = ranking["Número"].tolist()
        posicion = top100.index(real) + 1 if real in top100 else None

        filas.append(
            {
                "Fecha": fechas[indice],
                "Resultado real": real,
                "Ranking": posicion if posicion else "Fuera del Top 100",
                "Top 10": posicion is not None and posicion <= 10,
                "Top 20": posicion is not None and posicion <= 20,
                "Top 50": posicion is not None and posicion <= 50,
                "Top 100": posicion is not None,
            }
        )

        barra.progress(contador / cantidad_pruebas)
        estado.write(
            f"Probando {contador} de {cantidad_pruebas}: "
            f"{pd.to_datetime(fechas[indice]).strftime('%d/%m/%Y')}"
        )

    barra.empty()
    estado.empty()
    return pd.DataFrame(filas)


st.title("🎯 TRIS Predictor V7")
st.caption("Filtro inteligente por perfil antes del ranking estadístico.")

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
    st.warning("Sube el archivo `tris_historico.csv` desde el panel lateral.")
    st.stop()

st.sidebar.success(f"{len(datos):,} resultados cargados")

m1, m2, m3 = st.columns(3)
m1.metric("Resultados", f"{len(datos):,}")
m2.metric("Primera fecha", datos["fecha"].min().strftime("%d/%m/%Y"))
m3.metric("Última fecha", datos["fecha"].max().strftime("%d/%m/%Y"))

tab_ranking, tab_backtest, tab_perfil, tab_base = st.tabs(
    ["🎯 Ranking", "🧪 Backtesting", "🧬 Perfil", "🗂️ Base de datos"]
)

with st.sidebar.expander("⚙️ Pesos del modelo", expanded=True):
    peso_posicion = st.slider("Posición", 0.0, 1.0, 0.25, 0.01)
    peso_frecuencia = st.slider("Frecuencia", 0.0, 1.0, 0.10, 0.01)
    peso_pares = st.slider("Pares", 0.0, 1.0, 0.15, 0.01)
    peso_transiciones = st.slider("Transiciones", 0.0, 1.0, 0.15, 0.01)
    peso_ausencia = st.slider("Ausencia", 0.0, 1.0, 0.10, 0.01)
    peso_ciclo = st.slider("Ciclo del número", 0.0, 1.0, 0.10, 0.01)
    peso_repeticion = st.slider("Repetición", 0.0, 1.0, 0.03, 0.01)
    peso_espejo = st.slider("Espejo", 0.0, 1.0, 0.02, 0.01)
    peso_similitud = st.slider("Similitud simple con último", 0.0, 1.0, 0.01, 0.01)
    peso_influencia = st.slider("Influencia aprendida del último", 0.0, 1.0, 0.20, 0.01)
    peso_delta = st.slider("Patrón de cambio por posición", 0.0, 1.0, 0.10, 0.01)
    peso_coincidencias = st.slider("Posiciones conservadas", 0.0, 1.0, 0.05, 0.01)
    peso_suma = st.slider("Equilibrio de suma", 0.0, 1.0, 0.04, 0.01)
    peso_paridad = st.slider("Paridad", 0.0, 1.0, 0.03, 0.01)

pesos = {
    "Posición": peso_posicion,
    "Frecuencia": peso_frecuencia,
    "Pares": peso_pares,
    "Transiciones": peso_transiciones,
    "Ausencia": peso_ausencia,
    "Ciclo": peso_ciclo,
    "Repetición": peso_repeticion,
    "Espejo": peso_espejo,
    "Similitud último": peso_similitud,
    "Influencia último": peso_influencia,
    "Patrón delta": peso_delta,
    "Coincidencias históricas": peso_coincidencias,
    "Suma": peso_suma,
    "Paridad": peso_paridad,
}

with tab_ranking:
    c1, c2, c3 = st.columns(3)
    sorteo = c1.selectbox("Sorteo", SORTEOS)
    lado = c2.selectbox("Directa 4", ["Últimos 4", "Primeros 4"])
    top_n = c3.slider("Cantidad de candidatos", 10, 100, 30, 10)

    c4, c5, c6 = st.columns(3)
    ventana = c4.slider("Ventana reciente", 20, 500, 150, 10)
    recencia = c5.slider("Peso de recencia", 0.0, 3.0, 1.5, 0.1)
    excluir = c6.checkbox("Excluir números ya vistos", value=False)

    f1, f2, f3 = st.columns(3)
    usar_filtro = f1.checkbox("Usar filtro inteligente", value=True)
    ventana_perfil = f2.slider("Ventana del perfil", 50, 500, 200, 25)
    supervivencia = f3.slider(
        "Porcentaje que sobrevive",
        1,
        20,
        5,
        1,
        help="5% equivale a aproximadamente 500 candidatos antes del ranking final.",
    )

    filtrados = datos[datos["sorteo"] == sorteo].sort_values("fecha")
    numeros = preparar_directa4(filtrados, lado)

    ranking = calcular_ranking(
        numeros,
        top_n=top_n,
        ventana_reciente=ventana,
        peso_recencia=recencia,
        pesos=pesos,
        excluir_vistos=excluir,
        usar_filtro_perfil=usar_filtro,
        ventana_perfil=ventana_perfil,
        porcentaje_supervivencia=supervivencia,
    )

    if not ranking.empty:
        mejor = ranking.iloc[0]

        r1, r2, r3 = st.columns(3)
        r1.metric("Mejor clasificado", mejor["Número"])
        r2.metric("Puntuación interna", f'{mejor["Puntuación"]:.2f}')
        r3.metric("Resultados analizados", f"{len(numeros):,}")

        st.dataframe(
            ranking,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Descargar ranking CSV",
            ranking.to_csv(index=False).encode("utf-8-sig"),
            file_name="ranking_v7.csv",
            mime="text/csv",
        )

with tab_backtest:
    st.subheader("Validación histórica")

    b1, b2, b3 = st.columns(3)
    sorteo_bt = b1.selectbox("Sorteo", SORTEOS, key="bt_sorteo")
    lado_bt = b2.selectbox(
        "Directa 4",
        ["Últimos 4", "Primeros 4"],
        key="bt_lado",
    )
    pruebas = b3.slider("Sorteos a probar", 10, 100, 30, 10)

    b4, b5 = st.columns(2)
    ventana_bt = b4.slider(
        "Ventana reciente",
        20,
        500,
        150,
        10,
        key="bt_ventana",
    )
    recencia_bt = b5.slider(
        "Peso de recencia",
        0.0,
        3.0,
        1.5,
        0.1,
        key="bt_recencia",
    )

    bf1, bf2, bf3 = st.columns(3)
    usar_filtro_bt = bf1.checkbox(
        "Usar filtro inteligente",
        value=True,
        key="bt_usar_filtro",
    )
    ventana_perfil_bt = bf2.slider(
        "Ventana del perfil",
        50,
        500,
        200,
        25,
        key="bt_ventana_perfil",
    )
    supervivencia_bt = bf3.slider(
        "Porcentaje que sobrevive",
        1,
        20,
        5,
        1,
        key="bt_supervivencia",
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

        if len(numeros_bt) < pruebas + 200:
            st.error("No hay suficientes resultados.")
        else:
            resultado = ejecutar_backtesting(
                numeros_bt,
                fechas_bt,
                pruebas,
                ventana_bt,
                recencia_bt,
                pesos,
                usar_filtro_bt,
                ventana_perfil_bt,
                supervivencia_bt,
            )
            st.session_state["bt_v7"] = resultado

    if "bt_v7" in st.session_state:
        resultado = st.session_state["bt_v7"]
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

        if a100 / total > 0.01:
            st.success("El Top 100 superó la referencia aleatoria del 1% en esta muestra.")
        else:
            st.warning("El Top 100 no superó la referencia aleatoria del 1%.")

        st.dataframe(
            resultado.sort_values("Fecha", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Descargar backtesting CSV",
            resultado.to_csv(index=False).encode("utf-8-sig"),
            file_name="backtesting_v7.csv",
            mime="text/csv",
        )


with tab_perfil:
    st.subheader("Perfil histórico reciente")

    p1, p2 = st.columns(2)
    sorteo_perfil = p1.selectbox(
        "Sorteo",
        SORTEOS,
        key="perfil_sorteo",
    )
    lado_perfil = p2.selectbox(
        "Directa 4",
        ["Últimos 4", "Primeros 4"],
        key="perfil_lado",
    )

    p3, p4 = st.columns(2)
    ventana_perfil_tab = p3.slider(
        "Resultados usados para aprender el perfil",
        50,
        500,
        200,
        25,
        key="perfil_ventana",
    )
    mostrar_perfiles = p4.slider(
        "Cantidad de perfiles probables",
        5,
        30,
        10,
        5,
    )

    filtrados_perfil = (
        datos[datos["sorteo"] == sorteo_perfil]
        .sort_values("fecha")
    )
    numeros_perfil = preparar_directa4(
        filtrados_perfil,
        lado_perfil,
    )

    modelo_perfil = aprender_perfil(
        numeros_perfil,
        ventana_perfil=ventana_perfil_tab,
    )

    perfiles_posibles = []
    for pares in range(5):
        for repetidos in range(4):
            for altos in range(5):
                score = (
                    0.4 * modelo_perfil["pares"].get(pares, 0)
                    + 0.35 * modelo_perfil["repetidos"].get(repetidos, 0)
                    + 0.25 * modelo_perfil["altos"].get(altos, 0)
                )
                perfiles_posibles.append(
                    {
                        "Pares": pares,
                        "Impares": 4 - pares,
                        "Repetidos": repetidos,
                        "Dígitos altos": altos,
                        "Dígitos bajos": 4 - altos,
                        "Score": round(score * 100, 2),
                    }
                )

    perfiles_df = (
        pd.DataFrame(perfiles_posibles)
        .sort_values("Score", ascending=False)
        .head(mostrar_perfiles)
        .reset_index(drop=True)
    )
    perfiles_df.insert(0, "Ranking", range(1, len(perfiles_df) + 1))

    st.dataframe(
        perfiles_df,
        use_container_width=True,
        hide_index=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Suma promedio", f'{modelo_perfil["suma_media"]:.1f}')
    m2.metric("Desviación de suma", f'{modelo_perfil["suma_std"]:.1f}')
    m3.metric(
        "Probabilidad histórica de iniciar con 0",
        f'{modelo_perfil["prob_cero"] * 100:.1f}%',
    )


with tab_base:
    st.dataframe(
        datos.sort_values("fecha", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
