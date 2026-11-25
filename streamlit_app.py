import numpy as np
import pandas as pd
import sympy as sp
import streamlit as st

from calculadora_par_impar import FUNCIONES, descomponer, t

st.set_page_config(page_title="Componentes par e impar", page_icon="📈")
st.title("Calculadora de componentes par e impar")
st.latex(r"x_e(t)=\frac{x(t)+x(-t)}{2}\qquad x_o(t)=\frac{x(t)-x(-t)}{2}")

expresion = st.text_input("Señal x(t)", value="t^2 + sin(t) + exp(t)")
col1, col2 = st.columns(2)
inicio = col1.number_input("Inicio", value=-5.0)
fin = col2.number_input("Fin", value=5.0)

st.caption(
    "Funciones: sin, cos, tan, exp, sqrt, abs, ln, sign, pi, e, u(t). "
    "Potencias con ^."
)


def a_numerico(texto):
    expr = sp.sympify(texto, locals={"t": t, **FUNCIONES})
    f = sp.lambdify(
        t,
        expr,
        modules=[{"Heaviside": lambda x, h=1: np.heaviside(x, h)}, "numpy"],
    )
    return f


if st.button("Calcular", type="primary"):
    if not expresion.strip():
        st.warning("Escribe una expresión.")
    elif fin <= inicio:
        st.warning("El valor final debe ser mayor que el inicial.")
    else:
        try:
            r = descomponer(expresion)
        except Exception as error:
            st.error(f"Expresión inválida: {error}")
        else:
            st.subheader("Resultado")
            st.latex(f"x(t)={r['latex']['x']}")
            st.latex(f"x_e(t)={r['latex']['par']}")
            st.latex(f"x_o(t)={r['latex']['impar']}")
            st.info(f"La señal es: {r['tipo']}")

            valores = np.linspace(inicio, fin, 400)
            par = np.real(np.broadcast_to(a_numerico(r["par"])(valores), valores.shape))
            impar = np.real(
                np.broadcast_to(a_numerico(r["impar"])(valores), valores.shape)
            )
            original = np.real(
                np.broadcast_to(a_numerico(r["x"])(valores), valores.shape)
            )
            datos = pd.DataFrame(
                {"x(t)": original, "Par": par, "Impar": impar}, index=valores
            )
            st.line_chart(datos)
