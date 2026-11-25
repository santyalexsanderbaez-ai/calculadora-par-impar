import numpy as np
import pandas as pd
import sympy as sp
import streamlit as st
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

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
}


def descomponer(expresion):
    x = parse_expr(
        expresion.replace("^", "**"),
        local_dict={**FUNCIONES, "t": t},
        transformations=TRANSFORMACIONES,
    )
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
    return np.real(np.broadcast_to(f(valores), valores.shape))


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

if st.button("Calcular", type="primary"):
    if not expresion.strip():
        st.warning("Escribe una expresión.")
    elif fin <= inicio:
        st.warning("El valor final debe ser mayor que el inicial.")
    else:
        try:
            r = descomponer(expresion)
            valores = np.linspace(inicio, fin, 400)
            datos = pd.DataFrame(
                {
                    "x(t)": a_numerico(r["x"], valores),
                    "Par": a_numerico(r["par"], valores),
                    "Impar": a_numerico(r["impar"], valores),
                },
                index=valores,
            )
        except Exception as error:
            st.error(f"Expresión inválida: {error}")
        else:
            st.subheader("Resultado")
            st.latex(f"x(t)={sp.latex(r['x'])}")
            st.latex(f"x_e(t)={sp.latex(r['par'])}")
            st.latex(f"x_o(t)={sp.latex(r['impar'])}")
            st.info(f"La señal es: {r['tipo']}")
            st.line_chart(datos)
