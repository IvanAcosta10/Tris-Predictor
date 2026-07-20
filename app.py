import re
import sqlite3
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title='TRIS Predictor', page_icon='🎯', layout='wide')
DB_PATH = Path('tris_resultados.db')
BASE_URL = 'https://www.resultadostris.com/resultados.php'
SORTEOS = ['Medio Día','De las Tres','Extra','De las Siete','Clásico']

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS resultados (
        fecha TEXT NOT NULL,
        sorteo TEXT NOT NULL,
        numero TEXT NOT NULL,
        fuente TEXT,
        actualizado_en TEXT,
        PRIMARY KEY (fecha, sorteo)
    )''')
    conn.commit()
    return conn

def fetch_year(year):
    url = BASE_URL if year == datetime.now().year else f'{BASE_URL}?anno={year}'
    r = requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, timeout=30)
    r.raise_for_status()
    text = BeautifulSoup(r.text, 'html.parser').get_text(' ', strip=True)
    pat = re.compile(r'(\d{2}/\d{2}/\d{4})\s+(\d{5}|-----)\s+(\d{5}|-----)\s+(\d{5}|-----)\s+(\d{5}|-----)\s+(\d{5}|-----)')
    rows=[]
    for m in pat.finditer(text):
        fecha, a,b,c,d,e = m.groups()
        for sorteo, numero in zip(SORTEOS,[a,b,c,d,e]):
            if numero != '-----':
                rows.append({'fecha':pd.to_datetime(fecha,format='%d/%m/%Y').date().isoformat(),'sorteo':sorteo,'numero':numero,'fuente':url})
    if not rows:
        raise ValueError(f'No se encontraron resultados para {year}')
    return pd.DataFrame(rows)

def save_results(df):
    conn=init_db(); before=conn.execute('SELECT COUNT(*) FROM resultados').fetchone()[0]
    now=datetime.now().isoformat(timespec='seconds')
    rec=[(r.fecha,r.sorteo,str(r.numero).zfill(5),r.fuente,now) for r in df.itertuples(index=False)]
    conn.executemany('''INSERT INTO resultados(fecha,sorteo,numero,fuente,actualizado_en)
    VALUES(?,?,?,?,?) ON CONFLICT(fecha,sorteo) DO UPDATE SET numero=excluded.numero,fuente=excluded.fuente,actualizado_en=excluded.actualizado_en''',rec)
    conn.commit(); after=conn.execute('SELECT COUNT(*) FROM resultados').fetchone()[0]; conn.close(); return after-before

def load_results():
    conn=init_db(); df=pd.read_sql_query('SELECT * FROM resultados',conn,dtype={'numero':str}); conn.close()
    if not df.empty:
        df['fecha']=pd.to_datetime(df['fecha']); df['numero']=df['numero'].astype(str).str.zfill(5)
    return df

def normalize(counter, keys):
    vals=np.array([counter.get(k,0.0) for k in keys],float)
    if vals.max()==vals.min(): return {k:0.5 for k in keys}
    vals=(vals-vals.min())/(vals.max()-vals.min())
    return dict(zip(keys,vals))

def rank_direct4(numbers, top_n, recent_window, recency_strength, exclude_seen):
    n=len(numbers)
    if n<10: return pd.DataFrame()
    weights=np.ones(n); w=min(recent_window,n)
    weights[-w:]=np.linspace(1.0,1.0+recency_strength,w)
    digits=[str(i) for i in range(10)]
    pos=[Counter() for _ in range(4)]; glob=Counter(); pair=[Counter() for _ in range(3)]; trans=[defaultdict(Counter) for _ in range(3)]
    for num,wt in zip(numbers,weights):
        for i,d in enumerate(num): pos[i][d]+=wt; glob[d]+=wt
        for i in range(3): pair[i][num[i:i+2]]+=wt; trans[i][num[i]][num[i+1]]+=wt
    posn=[normalize(c,digits) for c in pos]; globn=normalize(glob,digits); pkeys=[f'{i:02d}' for i in range(100)]; pairn=[normalize(c,pkeys) for c in pair]
    transn=[{d:normalize(trans[i][d],digits) for d in digits} for i in range(3)]
    seen=set(numbers); rows=[]
    for v in range(10000):
        num=f'{v:04d}'
        if exclude_seen and num in seen: continue
        d=list(num)
        ps=sum(posn[i][d[i]] for i in range(4))/4
        gs=sum(globn[x] for x in d)/4
        prs=sum(pairn[i][num[i:i+2]] for i in range(3))/3
        ts=sum(transn[i][d[i]][d[i+1]] for i in range(3))/3
        rep=(4-len(set(d)))/3; mir=(int(d[0]==d[3])+int(d[1]==d[2]))/2
        score=.38*ps+.18*gs+.22*prs+.17*ts+.03*rep+.02*mir
        rows.append({'Número':num,'Puntuación':score,'Posición':ps,'Pares':prs,'Transiciones':ts,'Frecuencia':gs})
    out=pd.DataFrame(rows).sort_values('Puntuación',ascending=False).head(top_n).reset_index(drop=True)
    out.insert(0,'Ranking',range(1,len(out)+1))
    for c in ['Puntuación','Posición','Pares','Transiciones','Frecuencia']: out[c]=(out[c]*100).round(2)
    return out

st.title('🎯 TRIS Predictor V2')
st.caption('Histórico automático y ranking estadístico para Directa 4.')

t1,t2,t3=st.tabs(['Actualizar resultados','Analizar Directa 4','Base de datos'])
with t1:
    st.subheader('Descargar histórico automáticamente')
    y=datetime.now().year
    c1,c2=st.columns(2)
    start=int(c1.number_input('Desde el año',2015,y,2015)); end=int(c2.number_input('Hasta el año',2015,y,y))
    if st.button('🔄 Actualizar resultados',type='primary',use_container_width=True):
        if start>end: st.error('El año inicial no puede ser mayor.')
        else:
            frames=[]; errors=[]; years=list(range(start,end+1)); prog=st.progress(0); status=st.empty()
            for i,yr in enumerate(years,1):
                status.write(f'Descargando {yr}...')
                try: frames.append(fetch_year(yr))
                except Exception as e: errors.append(f'{yr}: {e}')
                prog.progress(i/len(years))
            status.empty(); prog.empty()
            if frames:
                all_df=pd.concat(frames,ignore_index=True); added=save_results(all_df)
                st.success(f'Listo: {len(all_df):,} registros leídos y {added:,} nuevos guardados.')
            if errors: st.warning(' | '.join(errors))
    df=load_results()
    if not df.empty:
        a,b,c=st.columns(3); a.metric('Registros',f'{len(df):,}'); b.metric('Primera fecha',df.fecha.min().strftime('%d/%m/%Y')); c.metric('Última fecha',df.fecha.max().strftime('%d/%m/%Y'))
with t2:
    df=load_results()
    if df.empty: st.info('Primero actualiza los resultados.')
    else:
        a,b,c=st.columns(3)
        sorteo=a.selectbox('Sorteo',SORTEOS); lado=b.selectbox('Directa 4',['Últimos 4','Primeros 4']); topn=c.slider('Cantidad',10,100,30,10)
        d,e,f=st.columns(3); win=d.slider('Ventana reciente',20,500,120,10); strength=e.slider('Peso de recencia',0.0,3.0,1.5,.1); exclude=f.checkbox('Excluir ya vistos')
        nums=df[df.sorteo==sorteo].sort_values('fecha').numero.astype(str).str.zfill(5)
        nums=(nums.str[-4:] if lado=='Últimos 4' else nums.str[:4]).tolist()
        out=rank_direct4(nums,topn,win,strength,exclude)
        st.write(f'Analizando **{len(nums):,} sorteos** de **{sorteo}**.')
        if not out.empty:
            x,y,z=st.columns(3); x.metric('Mejor clasificado',out.iloc[0]['Número']); y.metric('Puntuación',f"{out.iloc[0]['Puntuación']:.2f}"); z.metric('Top',len(out))
            st.dataframe(out,use_container_width=True,hide_index=True)
            st.download_button('Descargar ranking CSV',out.to_csv(index=False).encode('utf-8-sig'),'ranking_directa4.csv','text/csv')
            st.caption('La puntuación ordena patrones históricos; no es una probabilidad real ni garantiza un resultado.')
with t3:
    df=load_results()
    if df.empty: st.info('No hay datos todavía.')
    else:
        st.dataframe(df.sort_values('fecha',ascending=False)[['fecha','sorteo','numero']],use_container_width=True,hide_index=True)
        st.download_button('Descargar toda la base CSV',df[['fecha','sorteo','numero']].to_csv(index=False).encode('utf-8-sig'),'tris_historico.csv','text/csv')
        up=st.file_uploader('Restaurar/importar CSV',type=['csv'])
        if up is not None:
            imp=pd.read_csv(up,dtype={'numero':str})
            if {'fecha','sorteo','numero'}.issubset(imp.columns):
                imp['numero']=imp.numero.astype(str).str.zfill(5); imp['fuente']='CSV importado'; added=save_results(imp[['fecha','sorteo','numero','fuente']]); st.success(f'Importado. Nuevos: {added}')
            else: st.error('El CSV debe contener fecha, sorteo y numero.')
