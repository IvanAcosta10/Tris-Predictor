from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="TRIS Predictor V6",
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


def calcular_ranking(
    numeros,
    top_n,
    ventana_reciente,
    peso_recencia,
    pesos,
    excluir_vistos=False,
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

    for valor in range(10000):
        numero = f"{valor:04d}"

        if excluir_vistos and numero in vistos:
            continue

        componentes = calcular_componentes(numero, modelo)
        puntuacion = puntuar_componentes(componentes, pesos)

        filas.append(
            {
                "Número": numero,
                "Puntuación": puntuacion,
                **componentes,
            }
        )

    ranking = (
        pd.DataFrame(filas)
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


st.title("🎯 TRIS Predictor V6")
st.caption("Motor modular con influencia aprendida del último resultado.")

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

tab_ranking, tab_backtest, tab_base = st.tabs(
    ["🎯 Ranking", "🧪 Backtesting", "🗂️ Base de datos"]
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

    filtrados = datos[datos["sorteo"] == sorteo].sort_values("fecha")
    numeros = preparar_directa4(filtrados, lado)

    ranking = calcular_ranking(
        numeros,
        top_n=top_n,
        ventana_reciente=ventana,
        peso_recencia=recencia,
        pesos=pesos,
        excluir_vistos=excluir,
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
            file_name="ranking_v6.csv",
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
            )
            st.session_state["bt_v6"] = resultado

    if "bt_v6" in st.session_state:
        resultado = st.session_state["bt_v6"]
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
            file_name="backtesting_v6.csv",
            mime="text/csv",
        )

with tab_base:
    st.dataframe(
        datos.sort_values("fecha", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
