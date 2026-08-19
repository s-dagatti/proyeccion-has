import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import datetime

# ----------------------------------------------------------------------
# --- CONFIGURACIÓN DE LA PÁGINA ---
# ----------------------------------------------------------------------
st.set_page_config(page_title="Proyector Prophet", page_icon="🚜", layout="wide")

st.title("🚜 Proyección de Hectáreas - Modelo Ajustable")
st.markdown(
    "Sube tu archivo para entrenar el modelo con los **últimos 30 meses**. Usá la palanca manual para forzar la inclinación comercial de la proyección.")

# --- BARRA LATERAL (CONTROLES) ---
with st.sidebar:
    st.header("1. 📂 Cargar Datos")
    uploaded_file = st.file_uploader("Cargar archivo 'DIG Hisotorico - Hoja 2.csv'", type=["csv"])

    st.markdown("---")
    st.header("2. 🛠️ Inclinación Manual")

    ajuste_pendiente = st.slider(
        "Multiplicador de Crecimiento Futuro",
        min_value=-2.0, max_value=5.0, value=1.0, step=0.1,
        help="1.0 mantiene el ritmo de crecimiento histórico. 2.0 duplica la velocidad de crecimiento. 0.0 congela el crecimiento (línea plana)."
    )

    st.markdown("---")
    st.header("3. ⚙️ Parámetros Matemáticos")

    flexibilidad_tendencia = st.slider(
        "Flexibilidad de Tendencia Histórica",
        min_value=0.001, max_value=0.5, value=0.05, format="%.3f"
    )

    fuerza_estacionalidad = st.slider(
        "Fuerza de Estacionalidad Mensual",
        min_value=0.1, max_value=20.0, value=10.0, format="%.1f"
    )

    modo_estacionalidad = st.radio(
        "Comportamiento de Estacionalidad",
        ['multiplicative', 'additive'],
        format_func=lambda x: "Multiplicativa (Proporcional)" if x == 'multiplicative' else "Aditiva (Fija)"
    )

    intervalo_confianza = st.slider(
        "Margen Máx/Mín (Confianza)",
        min_value=0.50, max_value=0.99, value=0.90, format="%.2f"
    )

# --- ÁREA PRINCIPAL ---
if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)

        # Ojo: No consideramos valores nulos
        df_raw['Fecha'] = pd.to_datetime(df_raw['Fecha'], dayfirst=True, errors='coerce')
        df_raw = df_raw.dropna(subset=['Fecha']).sort_values('Fecha')

        opciones_columnas = [col for col in df_raw.columns if col != 'Fecha']

        col1, col2 = st.columns([1, 2])
        with col1:
            target_col = st.selectbox("📌 ¿Qué indicador querés proyectar?", opciones_columnas)

        df_prophet = df_raw[['Fecha', target_col]].rename(columns={'Fecha': 'ds', target_col: 'y'}).dropna()

        # Filtro estricto: Últimos 30 meses
        fecha_maxima = df_prophet['ds'].max()
        fecha_inicio = fecha_maxima - pd.DateOffset(months=30)
        df_entrenamiento = df_prophet[df_prophet['ds'] >= fecha_inicio].copy()

        if df_entrenamiento.shape[0] < 5:
            st.error("No hay suficientes datos en los últimos 30 meses para generar una proyección.")
        else:
            with st.spinner("Calculando proyección e inyectando ajustes de pendiente..."):

                # 1. Entrenar el modelo
                modelo = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=flexibilidad_tendencia,
                    seasonality_prior_scale=fuerza_estacionalidad,
                    seasonality_mode=modo_estacionalidad,
                    interval_width=intervalo_confianza
                )

                modelo.fit(df_entrenamiento)
                futuro = modelo.make_future_dataframe(periods=14, freq='MS')
                proyeccion = modelo.predict(futuro)

                # -------------------------------------------------------------
                # NUEVA LÓGICA DE INCLINACIÓN (Basada en la pendiente real)
                # -------------------------------------------------------------
                mask_futuro = proyeccion['ds'] > fecha_maxima

                if mask_futuro.any():
                    # Calculamos cuánto creció la tendencia en el periodo histórico
                    trend_inicial = proyeccion.loc[proyeccion['ds'] >= fecha_inicio, 'trend'].iloc[0]
                    trend_final = proyeccion.loc[proyeccion['ds'] <= fecha_maxima, 'trend'].iloc[-1]

                    meses_historia = df_entrenamiento.shape[0]
                    # Hectáreas promedio que creció por mes históricamente
                    pendiente_mensual_historica = (trend_final - trend_inicial) / max(1, meses_historia)

                    # Le aplicamos tu multiplicador a esa tasa de crecimiento real
                    nueva_pendiente_mensual = pendiente_mensual_historica * ajuste_pendiente

                    # Reconstruimos la tendencia futura mes a mes
                    meses_futuros = pd.Series(range(1, mask_futuro.sum() + 1), index=proyeccion[mask_futuro].index)
                    nuevo_trend_futuro = trend_final + (nueva_pendiente_mensual * meses_futuros)

                    # Le sumamos/multiplicamos la estacionalidad (los picos de cada mes) a la nueva línea
                    multipliers = proyeccion.loc[mask_futuro, 'multiplicative_terms']
                    additives = proyeccion.loc[mask_futuro, 'additive_terms']

                    nuevo_yhat = nuevo_trend_futuro * (1 + multipliers) + additives

                    # Ajustamos los márgenes de máximos y mínimos para que sigan a la nueva línea
                    diferencia_absoluta = nuevo_yhat - proyeccion.loc[mask_futuro, 'yhat']
                    proyeccion.loc[mask_futuro, 'yhat'] = nuevo_yhat
                    proyeccion.loc[mask_futuro, 'yhat_lower'] += diferencia_absoluta
                    proyeccion.loc[mask_futuro, 'yhat_upper'] += diferencia_absoluta

                # Limpiar valores negativos por si le ponés un multiplicador muy negativo
                for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                    proyeccion[col] = proyeccion[col].apply(lambda x: max(0, x))

                # -------------------------------------------------------------
                # GRAFICAR CON PLOTLY
                # -------------------------------------------------------------
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=pd.concat([proyeccion['ds'], proyeccion['ds'].iloc[::-1]]),
                    y=pd.concat([proyeccion['yhat_upper'], proyeccion['yhat_lower'].iloc[::-1]]),
                    fill='toself',
                    fillcolor='rgba(31, 119, 180, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'Escenarios (Máx/Mín al {int(intervalo_confianza * 100)}%)',
                    hoverinfo="skip"
                ))

                fig.add_trace(go.Scatter(
                    x=proyeccion['ds'],
                    y=proyeccion['yhat'],
                    mode='lines',
                    line=dict(color='rgb(31, 119, 180)', width=3, dash='dash'),
                    name='Proyección (Ajustada)',
                    hovertemplate='Fecha: %{x}<br>Valor: %{y:,.0f}<extra></extra>'
                ))

                fig.add_trace(go.Scatter(
                    x=df_entrenamiento['ds'],
                    y=df_entrenamiento['y'],
                    mode='lines+markers',
                    line=dict(color='black', width=2),
                    marker=dict(size=6, color='black'),
                    name='Histórico (30 meses)',
                    hovertemplate='Fecha: %{x}<br>Valor Real: %{y:,.0f}<extra></extra>'
                ))

                fig.update_layout(
                    title=f"Análisis de {target_col}",
                    xaxis_title='Fecha',
                    yaxis_title='Valor',
                    hovermode="x unified",
                    template='plotly_white',
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )

                st.plotly_chart(fig, use_container_width=True)

                # --- TABLA Y DESCARGA ---
                st.markdown("### 📊 Detalle de los próximos 14 meses")
                df_tabla_futuro = proyeccion[proyeccion['ds'] > fecha_maxima].copy()
                df_tabla_futuro = df_tabla_futuro[['ds', 'yhat_lower', 'yhat', 'yhat_upper']]
                df_tabla_futuro.columns = ['Fecha', 'Escenario Pesimista', 'Proyección Ajustada', 'Escenario Optimista']

                df_display = df_tabla_futuro.copy()
                df_display['Fecha'] = df_display['Fecha'].dt.strftime('%B %Y')
                st.dataframe(
                    df_display.style.format({
                        'Escenario Pesimista': '{:,.0f}',
                        'Proyección Ajustada': '{:,.0f}',
                        'Escenario Optimista': '{:,.0f}'
                    }),
                    use_container_width=True
                )

                csv = df_tabla_futuro.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar Proyección Ajustada (CSV)",
                    data=csv,
                    file_name='Proyeccion_Ajustada_DIG.csv',
                    mime='text/csv',
                    type='secondary'
                )

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
else:
    st.info("👈 Esperando que cargues el archivo CSV en el menú lateral.")
