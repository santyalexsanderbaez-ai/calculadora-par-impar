import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# --- Configuración inicial de página ---
st.set_page_config(
    page_title="Análisis de Señales: Par e Impar",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Estilos CSS personalizados ---
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .badge-par {
        background-color: #1f77b420;
        color: #1f77b4;
        padding: 0.4rem 0.8rem;
        border-radius: 0.5rem;
        font-weight: 600;
        border: 1px solid #1f77b4;
        display: inline-block;
    }
    .badge-impar {
        background-color: #ff7f0e20;
        color: #ff7f0e;
        padding: 0.4rem 0.8rem;
        border-radius: 0.5rem;
        font-weight: 600;
        border: 1px solid #ff7f0e;
        display: inline-block;
    }
    .badge-mixta {
        background-color: #2ca02c20;
        color: #2ca02c;
        padding: 0.4rem 0.8rem;
        border-radius: 0.5rem;
        font-weight: 600;
        border: 1px solid #2ca02c;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Configuración simbólica ---
t = sp.Symbol("t", real=True)
TRANSFORMACIONES = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)
FUNCIONES = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "ln": sp.log,
    "log": sp.log,
    "pi": sp.pi,
    "e": sp.E,
    "u": lambda x: sp.Heaviside(x, 1),
    "sign": sp.sign,
    "oo": sp.oo,
    "inf": sp.oo,
}

EJEMPLOS = {
    "Escribir mi propia señal": "",
    "Mixta: t^2 + sin(t) + exp(t)": "t^2 + sin(t) + exp(t)",
    "Seno: sin(t)": "sin(t)",
    "Coseno: cos(t)": "cos(t)",
    "Seno + coseno: sin(t) + cos(t)": "sin(t) + cos(t)",
    "Polinomio: t^3 + t^2": "t^3 + t^2",
    "Exponencial: exp(t)": "exp(t)",
    "Exponencial causal: exp(-t)*u(t)": "exp(-t)*u(t)",
    "Escalón unitario: u(t)": "u(t)",
    "Rampa: t*u(t)": "t*u(t)",
    "Pulso: u(t) - u(t-2)": "u(t) - u(t-2)",
    "Valor absoluto + lineal: abs(t) + t": "abs(t) + t",
    "Seno amortiguado: exp(-t)*sin(t)*u(t)": "exp(-t)*sin(t)*u(t)",
}


def parsear_texto(texto: str):
    return parse_expr(
        texto.strip().replace("^", "**"),
        local_dict={**FUNCIONES, "t": t},
        transformations=TRANSFORMACIONES,
    )


def construir_desde_intervalos(df_tramos: pd.DataFrame):
    pares_piecewise = []
    for _, fila in df_tramos.iterrows():
        expr_str = str(fila.get("Expresión x(t)", "")).strip()
        lim_inf_str = str(fila.get("Límite inferior", "-oo")).strip()
        lim_sup_str = str(fila.get("Límite superior", "oo")).strip()

        if not expr_str:
            continue

        expr_sym = parsear_texto(expr_str)
        lim_inf = parsear_texto(lim_inf_str) if lim_inf_str else -sp.oo
        lim_sup = parsear_texto(lim_sup_str) if lim_sup_str else sp.oo

        if lim_inf != -sp.oo and lim_sup != sp.oo:
            if float(lim_inf) >= float(lim_sup):
                raise ValueError(
                    f"El límite inferior ({lim_inf}) debe ser menor que el superior ({lim_sup})."
                )
            condicion = sp.And(t >= lim_inf, t < lim_sup)
        elif lim_inf == -sp.oo and lim_sup != sp.oo:
            condicion = t < lim_sup
        elif lim_inf != -sp.oo and lim_sup == sp.oo:
            condicion = t >= lim_inf
        else:
            condicion = True

        pares_piecewise.append((expr_sym, condicion))

    if not pares_piecewise:
        raise ValueError("Debes definir al menos un intervalo válido.")

    # Fuera de los intervalos definidos, la señal vale 0
    pares_piecewise.append((0, True))
    return sp.Piecewise(*pares_piecewise)


def descomponer_expresion(x):
    x_inv = x.subs(t, -t)
    par = sp.simplify((x + x_inv) / 2)
    impar = sp.simplify((x - x_inv) / 2)

    if sp.simplify(x - par) == 0:
        tipo = "par"
    elif sp.simplify(x - impar) == 0:
        tipo = "impar"
    else:
        tipo = "ni par ni impar"

    return {"x": x, "par": par, "impar": impar, "tipo": tipo}


def a_numerico(expr, valores):
    f = sp.lambdify(
        t,
        expr,
        modules=[{"Heaviside": lambda x, h=1: np.heaviside(x, h)}, "numpy"],
    )
    resultado = f(valores)
    return np.real(np.broadcast_to(resultado, valores.shape)).astype(float)


def crear_grafica(valores, y, titulo, color, nombre_eje_y):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=valores,
            y=y,
            mode="lines",
            name=nombre_eje_y,
            line=dict(color=color, width=2.8),
            hovertemplate="t = %{x:.3f}<br>" + f"{nombre_eje_y} = " + "%{y:.3f}<extra></extra>",
        )
    )
    # Líneas de referencia en los ejes
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="rgba(150, 150, 150, 0.5)")
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="rgba(150, 150, 150, 0.5)")

    fig.update_layout(
        title=dict(text=titulo, font=dict(size=16)),
        xaxis_title="Tiempo (t)",
        yaxis_title=nombre_eje_y,
        template="plotly_white",
        height=340,
        margin=dict(l=20, r=20, t=45, b=20),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.2)")
    return fig


def cargar_ejemplo():
    elegido = EJEMPLOS[st.session_state["ejemplo"]]
    if elegido:
        st.session_state["expresion"] = elegido


# --- Estado inicial ---
if "expresion" not in st.session_state:
    st.session_state["expresion"] = "t^2 + sin(t) + exp(t)"

if "tabla_intervalos" not in st.session_state:
    st.session_state["tabla_intervalos"] = pd.DataFrame(
        [
            {"Límite inferior": "-2", "Límite superior": "0", "Expresión x(t)": "t + 2"},
            {"Límite inferior": "0", "Límite superior": "2", "Expresión x(t)": "2 - t"},
        ]
    )

# --- Barra lateral (Controles) ---
with st.sidebar:
    st.header("⚙️ Configuración de la Señal")
    modo = st.radio(
        "Tipo de definición",
        ["Expresión única", "Por intervalos (a trozos)"],
        index=0,
    )

    st.divider()

    if modo == "Expresión única":
        st.selectbox(
            "Señales predeterminadas",
            list(EJEMPLOS.keys()),
            key="ejemplo",
            on_change=cargar_ejemplo,
        )
        expresion_input = st.text_input("Señal x(t)", key="expresion")
    else:
        st.markdown("**Editor de tramos $x(t)$**")
        st.caption(
            "Define cada intervalo $[a, b)$. Usa `-oo` o `oo` para infinito. "
            "Fuera de los rangos definidos, $x(t) = 0$."
        )
        df_editado = st.data_editor(
            st.session_state["tabla_intervalos"],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Límite inferior": st.column_config.TextColumn("Desde (≥)", default="0"),
                "Límite superior": st.column_config.TextColumn("Hasta (<)", default="1"),
                "Expresión x(t)": st.column_config.TextColumn("x(t)", default="t"),
            },
        )

    st.divider()
    st.subheader("Rango de visualización")
    col1, col2 = st.columns(2)
    inicio = col1.number_input("t inicial", value=-5.0, step=0.5)
    fin = col2.number_input("t final", value=5.0, step=0.5)

    calcular_btn = st.button("Analizar y Graficar", type="primary", use_container_width=True)

    with st.expander("📖 Sintaxis permitida"):
        st.markdown(
            """
            - **Variable:** `t`
            - **Potencias:** `t^2` o `t**2`
            - **Funciones:** `sin(t)`, `cos(t)`, `tan(t)`, `exp(t)`, `sqrt(t)`, `abs(t)`, `ln(t)`, `sign(t)`
            - **Escalón:** `u(t)` (vale 1 para $t \\ge 0$)
            - **Constantes:** `pi`, `e`, `oo` (infinito)
            """
        )

# --- Área Principal ---
st.title("Descomposición de Señales en Componentes Par e Impar")
st.markdown(
    "Toda señal $x(t)$ puede descomponerse en la suma de una componente **par** $x_e(t)$ y una componente **impar** $x_o(t)$:"
)
st.latex(r"x(t) = x_e(t) + x_o(t) \qquad \text{donde} \qquad x_e(t)=\frac{x(t)+x(-t)}{2}, \quad x_o(t)=\frac{x(t)-x(-t)}{2}")
st.divider()

# Ejecutar cálculo al cargar por primera vez o al presionar el botón
if fin <= inicio:
    st.error("El tiempo final debe ser estrictamente mayor que el tiempo inicial.")
else:
    try:
        if modo == "Expresión única":
            if not expresion_input.strip():
                st.warning("Ingresa una expresión matemática en el panel izquierdo.")
                st.stop()
            x_sym = parsear_texto(expresion_input)
        else:
            x_sym = construir_desde_intervalos(df_editado)

        r = descomponer_expresion(x_sym)
        valores = np.linspace(inicio, fin, 800)

        y_orig = a_numerico(r["x"], valores)
        y_par = a_numerico(r["par"], valores)
        y_impar = a_numerico(r["impar"], valores)

    except Exception as error:
        st.error(f"Error al procesar la señal: {error}")
    else:
        # --- Sección 1: Resultados Analíticos ---
        st.subheader("1. Resultado Analítico")

        if r["tipo"] == "par":
            st.markdown('<span class="badge-par">Clasificación: Señal Puramente PAR</span>', unsafe_allow_html=True)
        elif r["tipo"] == "impar":
            st.markdown('<span class="badge-impar">Clasificación: Señal Puramente IMPAR</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-mixta">Clasificación: Señal MIXTA (ni par ni impar)</span>', unsafe_allow_html=True)

        st.write("")
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.markdown("**Señal Original $x(t)$**")
                st.latex(sp.latex(r["x"]))
        with c2:
            with st.container(border=True):
                st.markdown("**Componente Par $x_e(t)$**")
                st.latex(sp.latex(r["par"]))
        with c3:
            with st.container(border=True):
                st.markdown("**Componente Impar $x_o(t)$**")
                st.latex(sp.latex(r["impar"]))

        st.divider()

        # --- Sección 2: Gráficas Separadas ---
        st.subheader("2. Visualización de Componentes")

        g1, g2, g3 = st.columns(3)
        with g1:
            fig_orig = crear_grafica(valores, y_orig, "Señal Original x(t)", "#2ca02c", "x(t)")
            st.plotly_chart(fig_orig, use_container_width=True)

        with g2:
            fig_par = crear_grafica(valores, y_par, "Componente Par xe(t)", "#1f77b4", "xe(t)")
            st.plotly_chart(fig_par, use_container_width=True)

        with g3:
            fig_impar = crear_grafica(valores, y_impar, "Componente Impar xo(t)", "#ff7f0e", "xo(t)")
            st.plotly_chart(fig_impar, use_container_width=True)
