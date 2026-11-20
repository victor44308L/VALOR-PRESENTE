"""
Valor presente de una pensión con tasa creciente
Interfaz en Streamlit — Análisis Numérico (PIA)

Ejecutar:
    pip install streamlit numpy scipy matplotlib pandas
    streamlit run app_valor_presente.py
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.integrate import quad
from scipy.special import erfcx

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# =====================================================================
# 1. CÁLCULOS (misma lógica que Valor_Presente_PIA.py)
# =====================================================================

def v(t, a, b):
    """Factor de descuento: cuánto vale hoy $1 recibido en el año t."""
    return np.exp(-(a * t + b * t**2 / 2))


def integral_exacta(a, b, T):
    if b == 0:
        return T if a == 0 else (1 - np.exp(-a * T)) / a
    if b > 0:
        c = np.sqrt(b / 2)
        u0 = c * (a / b)
        u1 = c * (T + a / b)
        return np.sqrt(np.pi / (2 * b)) * (
            erfcx(u0) - erfcx(u1) * np.exp(-(a * T + b * T**2 / 2))
        )
    return quad(lambda t: v(t, a, b), 0, T, epsabs=1e-13, epsrel=1e-13)[0]


def trapecio(f, x0, x1, n):
    h = (x1 - x0) / n
    y = f(x0 + h * np.arange(n + 1))
    return h * (y[0] / 2 + y[1:-1].sum() + y[-1] / 2)


def simpson(f, x0, x1, n):
    if n % 2 != 0:
        raise ValueError("requiere n par")
    h = (x1 - x0) / n
    y = f(x0 + h * np.arange(n + 1))
    return h / 3 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum() + y[-1])


def boole(f, x0, x1, n):
    if n % 4 != 0:
        raise ValueError("requiere n múltiplo de 4")
    h = (x1 - x0) / n
    y = f(x0 + h * np.arange(n + 1))
    return 2 * h / 45 * (7 * (y[0] + y[-1]) + 32 * y[1::2].sum()
                         + 12 * y[2:-1:4].sum() + 14 * y[4:-1:4].sum())


METODOS = {"Trapecio": trapecio, "Simpson 1/3": simpson, "Boole": boole}
ORDEN = {"Trapecio": "h²", "Simpson 1/3": "h⁴", "Boole": "h⁶"}
TEORICO = {"Trapecio": 2, "Simpson 1/3": 4, "Boole": 6}


def integrar_auto(metodo, f, x0, x1, tol, n=4, n_max=2**18):
    """Duplica n hasta que dos resultados seguidos difieran menos que tol."""
    anterior = metodo(f, x0, x1, n)
    while n < n_max:
        n *= 2
        actual = metodo(f, x0, x1, n)
        if abs(actual - anterior) < tol:
            return actual, n
        anterior = actual
    return actual, n


# =====================================================================
# 2. ESTILO: cuaderno cuadriculado en blanco y negro
# =====================================================================

st.set_page_config(page_title="Valor presente de una pensión", page_icon="∫", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600;700&family=Patrick+Hand&display=swap');

html, body, [class*="css"], .stMarkdown, p, label, input, li {
    font-family: 'Quicksand', sans-serif !important;
    color: #111;
}
.stApp {
    background-color: #fafafa;
    background-image:
        linear-gradient(#e6e6e6 1px, transparent 1px),
        linear-gradient(90deg, #e6e6e6 1px, transparent 1px);
    background-size: 26px 26px;
}
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 2px solid #111;
}
h1, h2, h3 { font-family: 'Quicksand', sans-serif !important; color: #111 !important; }

/* Encabezado tipo hoja de cuaderno */
.hoja {
    position: relative;
    background: #fff;
    border: 2px solid #111;
    border-radius: 6px;
    box-shadow: 6px 6px 0 #111;
    padding: 28px 36px 24px 70px;
    margin-bottom: 26px;
    overflow: hidden;
}
.hoja::before {                     /* agujeros de la libreta */
    content: "";
    position: absolute; left: 22px; top: 18px; bottom: 18px; width: 18px;
    background: radial-gradient(circle, #e9e9e9 7px, #bbb 8px, transparent 9px) 0 0 / 18px 42px repeat-y;
}
.hoja .pildora {
    display: inline-block;
    border: 2.5px solid #111;
    border-radius: 50% / 60%;
    padding: 4px 22px;
    font-weight: 700;
    font-size: 15px;
    transform: rotate(-2deg);
}
.hoja h1 { font-size: 44px; margin: 10px 0 4px 0; line-height: 1.1; }
.hoja .sub { font-family: 'Patrick Hand', cursive; font-size: 22px; color: #444; }
.hoja .doodle {
    position: absolute; right: 30px; top: 6px;
    font-family: 'Patrick Hand', cursive; font-size: 90px; color: #e3e3e3;
    letter-spacing: 18px; user-select: none;
}

/* Títulos con subrayado a mano */
.titulo {
    font-size: 28px; font-weight: 700; margin: 18px 0 12px 0;
    display: inline-block;
    text-decoration: underline;
    text-decoration-thickness: 3px;
    text-underline-offset: 9px;
    text-decoration-skip-ink: none;
}

/* Tarjetas de resultados */
.tarjeta {
    background: #fff;
    border: 2px solid #111;
    border-radius: 10px;
    box-shadow: 4px 4px 0 #111;
    padding: 16px 20px;
    height: 100%;
}
.tarjeta .etq { font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #555; }
.tarjeta .num { font-size: 30px; font-weight: 700; margin-top: 4px; }
.tarjeta .nota { font-family: 'Patrick Hand', cursive; font-size: 18px; color: #555; }
.tarjeta.negra { background: #111; color: #fff; }
.tarjeta.negra .etq, .tarjeta.negra .nota { color: #ccc; }

/* Tabla */
table.cuaderno { width: 100%; border-collapse: collapse; background: #fff; border: 2px solid #111; }
table.cuaderno th { background: #111; color: #fff; padding: 10px 12px; text-align: left; font-weight: 600; }
table.cuaderno td { padding: 10px 12px; border-top: 1px solid #ccc; }
table.cuaderno td.n { text-align: right; font-variant-numeric: tabular-nums; }

/* Pestañas y botones */
button[data-baseweb="tab"] { font-family: 'Quicksand', sans-serif !important; font-weight: 700; }
div[data-baseweb="tab-highlight"] { background-color: #111 !important; }
.stButton button, .stDownloadButton button {
    border: 2px solid #111 !important; border-radius: 30px !important;
    background: #fff !important; color: #111 !important; font-weight: 700 !important;
    box-shadow: 3px 3px 0 #111;
}
.aviso {
    font-family: 'Patrick Hand', cursive; font-size: 19px;
    border: 2px dashed #111; border-radius: 8px; padding: 10px 16px; background: #fff;
}
</style>
""",
    unsafe_allow_html=True,
)


def titulo(texto):
    st.markdown(f'<div class="titulo">{texto}</div>', unsafe_allow_html=True)


def tarjeta(etq, num, nota="", negra=False):
    clase = "tarjeta negra" if negra else "tarjeta"
    st.markdown(
        f'<div class="{clase}"><div class="etq">{etq}</div>'
        f'<div class="num">{num}</div><div class="nota">{nota}</div></div>',
        unsafe_allow_html=True,
    )


def estilo_ejes(ax):
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_color("#111")
        s.set_linewidth(1.5)
    ax.tick_params(colors="#111")
    ax.grid(True, color="#d9d9d9", linewidth=0.8)


# =====================================================================
# 3. ENTRADAS (barra lateral)
# =====================================================================

with st.sidebar:
    st.markdown("## ✎ Datos del problema")
    r0 = st.number_input("Tasa de interés inicial (%)", min_value=0.0, value=6.0, step=0.5) / 100
    r1 = st.number_input("Tasa de interés final (%)", min_value=0.0, value=8.0, step=0.5) / 100
    T = st.number_input("Número de años", min_value=0.5, value=20.0, step=1.0)
    P = st.number_input("Pesos por año", min_value=0.0, value=120_000.0, step=10_000.0, format="%.2f")

    st.markdown("## ✎ Subintervalos")
    auto = st.toggle("Que el programa elija n", value=False,
                     help="Duplica n hasta que dos resultados seguidos difieran menos que la tolerancia.")
    if auto:
        tol_pesos = st.number_input("Tolerancia (pesos)", min_value=1e-6, value=0.01, format="%.4f")
        n = None
    else:
        n = int(st.number_input("Número de subintervalos n", min_value=1, value=80, step=4))
        if n % 4 != 0:
            st.caption("Boole necesita n múltiplo de 4" + ("" if n % 2 == 0 else " y Simpson n par") + ".")

    a_mano = st.toggle("Gráficas con trazo a mano", value=False)

a = r0
b = (r1 - r0) / T

# =====================================================================
# 4. ENCABEZADO
# =====================================================================

st.markdown(
    """
<div class="hoja">
  <div class="doodle">∫ π Σ</div>
  <span class="pildora">Análisis Numérico</span>
  <h1>Valor presente de una pensión</h1>
  <div class="sub">Trapecio · Simpson 1/3 · Boole</div>
</div>
""",
    unsafe_allow_html=True,
)

titulo("El modelo")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Tasa de interés**")
    st.latex(rf"\delta(t) = {a:.4f} + {b:.6f}\,t")
with c2:
    st.markdown("**Factor de descuento**")
    st.latex(r"v(t) = e^{-\left(a t + \frac{b t^2}{2}\right)}")
with c3:
    st.markdown("**Valor presente**")
    st.latex(rf"VP = {P:,.0f}\int_0^{{{T:g}}} v(t)\,dt")

# =====================================================================
# 5. CÁLCULO
# =====================================================================

f = lambda t: v(t, a, b)
exacto = integral_exacta(a, b, T)
vp_exacto = P * exacto
vp_fija = P * integral_exacta(a, 0.0, T)

filas = []
for nombre, metodo in METODOS.items():
    try:
        if auto:
            tol = tol_pesos / P if P > 0 else 1e-12
            I, n_usado = integrar_auto(metodo, f, 0, T, tol)
        else:
            I, n_usado = metodo(f, 0, T, n), n
        filas.append({"Método": nombre, "n": n_usado, "Integral": I,
                      "Valor presente": P * I, "Error (pesos)": P * abs(I - exacto),
                      "Orden": ORDEN[nombre]})
    except ValueError as e:
        filas.append({"Método": nombre, "n": n, "Integral": None, "Valor presente": None,
                      "Error (pesos)": None, "Orden": ORDEN[nombre], "nota": str(e)})

# =====================================================================
# 6. RESULTADOS
# =====================================================================

titulo("Resultados")
k1, k2, k3 = st.columns(3)
with k1:
    tarjeta("Valor presente exacto", f"${vp_exacto:,.2f}",
            f"área = {exacto:.6f} años", negra=True)
with k2:
    tarjeta(f"Con tasa fija de {100 * r0:g}%", f"${vp_fija:,.2f}", "sin el cambio de tasa")
with k3:
    dif = vp_fija - vp_exacto
    pct = 100 * dif / vp_exacto if vp_exacto else 0
    tarjeta("Diferencia", f"${dif:,.2f}",
            f"{'sobrevalora' if dif > 0 else 'subvalora'} {abs(pct):.1f}%")

st.write("")
html = ('<table class="cuaderno"><tr><th>Método</th><th>n</th><th>Integral</th>'
        '<th>Valor presente</th><th>Error (pesos)</th><th>Orden</th></tr>')
for r in filas:
    if r["Integral"] is None:
        html += (f'<tr><td><b>{r["Método"]}</b></td><td class="n">{r["n"]}</td>'
                 f'<td colspan="3"><i>No aplica: {r["nota"]}</i></td><td>{r["Orden"]}</td></tr>')
    else:
        html += (f'<tr><td><b>{r["Método"]}</b></td><td class="n">{r["n"]:,}</td>'
                 f'<td class="n">{r["Integral"]:.12f}</td>'
                 f'<td class="n">${r["Valor presente"]:,.2f}</td>'
                 f'<td class="n">{r["Error (pesos)"]:.2e}</td><td>{r["Orden"]}</td></tr>')
html += "</table>"
st.markdown(html, unsafe_allow_html=True)

if auto:
    st.markdown(
        f'<p class="aviso">✎ Con tolerancia de ${tol_pesos:g}, cada método eligió su propio n: '
        "los de mayor orden necesitan muchos menos subintervalos.</p>",
        unsafe_allow_html=True,
    )

df = pd.DataFrame([{k: r[k] for k in ["Método", "n", "Integral", "Valor presente", "Error (pesos)"]}
                   for r in filas])
st.download_button("⬇ Descargar resultados (CSV)", df.to_csv(index=False).encode("utf-8"),
                   "resultados_valor_presente.csv", "text/csv")

# =====================================================================
# 7. GRÁFICAS
# =====================================================================

titulo("Gráficas")
tab1, tab2, tab3 = st.tabs(["∫  Área bajo la curva", "↘  Convergencia", "δ  Tasa de interés"])

estilo = plt.xkcd if a_mano else plt.rc_context
t = np.linspace(0, T, 400)

with tab1:
    with estilo():
        fig, ax = plt.subplots(figsize=(8, 4.2))
        estilo_ejes(ax)
        ax.fill_between(t, f(t), color="#d0d0d0", hatch="//", edgecolor="#999", linewidth=0)
        ax.plot(t, f(t), color="#111", linewidth=2.5)
        ax.text(0.04, 0.1, f"Área = {exacto:.4f} años\nVP = \\${vp_exacto:,.2f}",
                transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor="white", edgecolor="#111", boxstyle="round,pad=0.5"))
        ax.set_xlabel("t (años)")
        ax.set_ylabel("v(t)  factor de descuento")
        ax.set_xlim(0, T)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.caption(f"Un peso que se recibe en el año {T:g} vale hoy {f(T):.2f} pesos.")

with tab2:
    ns = [4, 8, 16, 32, 64, 128]
    hs = [T / k for k in ns]
    errores = {m: [abs(fn(f, 0, T, k) - exacto) for k in ns] for m, fn in METODOS.items()}
    with estilo():
        fig, ax = plt.subplots(figsize=(8, 4.2))
        estilo_ejes(ax)
        for (m, errs), mk in zip(errores.items(), ["o-", "s--", "D-."]):
            ax.loglog(hs, np.maximum(errs, 1e-17), mk, color="#111", linewidth=2, markersize=7,
                      markerfacecolor="white", label=m)
        ax.minorticks_off()
        ax.set_xticks(hs)
        ax.set_xticklabels([f"{x:g}" for x in hs])
        ax.set_xlabel("tamaño de paso h (años)")
        ax.set_ylabel("error absoluto")
        ax.legend(frameon=True, edgecolor="#111")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    ord_html = '<table class="cuaderno"><tr><th>Método</th>' + "".join(
        f"<th>{ns[i - 1]}→{ns[i]}</th>" for i in range(1, len(ns))) + "<th>Teórico</th></tr>"
    for m, errs in errores.items():
        celdas = []
        for i in range(1, len(ns)):
            if errs[i] > 1e-14 and errs[i - 1] > 1e-14:
                celdas.append(f"{np.log2(errs[i - 1] / errs[i]):.2f}")
            else:
                celdas.append("—")
        ord_html += (f"<tr><td><b>{m}</b></td>" + "".join(f'<td class="n">{c}</td>' for c in celdas)
                     + f'<td class="n">{TEORICO[m]}</td></tr>')
    ord_html += "</table>"
    st.markdown("**Orden observado** = log₂(error con h / error con h/2)")
    st.markdown(ord_html, unsafe_allow_html=True)
    st.caption("“—” significa que el error ya llegó al límite de precisión de la computadora.")

with tab3:
    with estilo():
        fig, ax = plt.subplots(figsize=(8, 4.2))
        estilo_ejes(ax)
        ax.plot(t, 100 * (a + b * t), color="#111", linewidth=2.5, label="Tasa creciente")
        ax.plot(t, np.full_like(t, 100 * a), "--", color="#777", linewidth=2, label="Tasa fija")
        ax.set_xlabel("t (años)")
        ax.set_ylabel("tasa (%)")
        ax.set_xlim(0, T)
        ax.legend(frameon=True, edgecolor="#111")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.caption("Suponer la tasa fija cambia el valor presente por la diferencia mostrada en Resultados.")
