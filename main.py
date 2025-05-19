# app.py  ────────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Całkowanie numeryczne – wariant 3  (∫₀^∞ e^{-x} f(x) dx)")


# ─────────────── HELPER: ładne formatowanie liczb (komunikaty & tabela) ─
def pretty(x, places=8):
    """
    Zwraca liczbę w formacie '0,0001234' zamiast '1.234e-04'
    · places – maks. liczba miejsc po przecinku
    · usuwa zbędne zera i przecinek na końcu
    """
    s = f"{x:.{places}f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


# ───────────────────────────────────────── GAUSS–LAGUERRE  TABLICE ──────
GAUSS_LAGUERRE_DATA = {
    2: {"nodes": np.array([0.58578644, 3.41421356]),
        "weights": np.array([0.85355339, 0.14644661])},
    3: {"nodes": np.array([0.41577456, 2.29428036, 6.28994508]),
        "weights": np.array([0.71109301, 0.27851773, 0.01038926])},
    4: {"nodes": np.array([0.32254769, 1.7457611, 4.5366203, 9.39507091]),
        "weights": np.array([0.6031541, 0.35741869, 0.03888791, 0.000539295])},
    5: {"nodes": np.array([0.26356032, 1.41340306, 3.59642577, 7.08581001, 12.64080084]),
        "weights": np.array([0.52175561, 0.39866681, 0.07594245, 0.00361175, 0.00002337])},
}


# ───────────────────────────────────────── FUNKCJE  TESTOWE ────────────
def f_constant_one(x): return np.ones_like(x) if isinstance(x, np.ndarray) else 1.0


def f_x(x):            return x


def f_x_squared(x):    return x ** 2


def f_x_cubed(x):      return x ** 3


def f_sin_x(x):        return np.sin(x)


def f_exp_neg_x(x):    return np.exp(-x)


FUNCTIONS = {
    "f(x) = 1": (f_constant_one, 1.0),
    "f(x) = x": (f_x, 1.0),  # Γ(2)=1
    "f(x) = x^2": (f_x_squared, 2.0),  # Γ(3)=2
    "f(x) = x^3": (f_x_cubed, 6.0),  # Γ(4)=6
    "f(x) = sin(x)": (f_sin_x, 0.5),
    "f(x) = exp(-x)": (f_exp_neg_x, 0.5),
}


# ───────────────────────────── POMOCNICZA  KLASA  LICZNIKA ─────────────
class FunctionCounter:
    def __init__(self, func):
        self.func = func
        self.calls = 0

    def __call__(self, x):
        self.calls += np.size(x)
        return self.func(x)


# ───────────────────────────── METODA  SIMPSONA  PODSTAWOWA ────────────
def simpson_basic(func, a, b, N):
    if N == 0: return 0.0
    if N % 2 == 1: N += 1
    h = (b - a) / N
    S = func(a) + func(b)
    for i in range(1, N):
        x_i = a + i * h
        S += 4 * func(x_i) if i % 2 else 2 * func(x_i)
    return S * h / 3.0


# ─────────────────── POMOCNICZA: POBIERANIE WĘZŁÓW SIMPSONA  ───────────
def get_simpson_nodes_for_plot(func, a, b, N_actual):
    if N_actual == 0: return np.array([]), np.array([])
    x_nodes = np.linspace(a, b, N_actual + 1)
    y_nodes = func(x_nodes)
    return x_nodes, y_nodes


# ───────────────────────────── ADAPTACYJNY  SIMPSON ────────────────────
def adaptive_simpson(func, a, b, tol, N_start=10, max_iter=15, mode="double"):
    N = N_start if N_start % 2 == 0 else N_start + 1
    val_prev = simpson_basic(func, a, b, N)

    for _ in range(max_iter):
        N_curr = (N * 2) if mode == "double" else (N + 2)
        val_curr = simpson_basic(func, a, b, N_curr)

        if abs(val_curr - val_prev) < tol:
            x_nodes, y_nodes = get_simpson_nodes_for_plot(func, a, b, N_curr)
            return val_curr, N_curr, x_nodes, y_nodes

        val_prev = val_curr
        N = N_curr

        if N > 2 ** 20: break

    x_nodes, y_nodes = get_simpson_nodes_for_plot(func, a, b, N)
    return val_prev, N, x_nodes, y_nodes


# ──────────────────── NEWTON-COTES  z OGONEM  DO  ∞  ───────────────────
def newton_cotes_variant3(
        f_original, eps_tail, A0=5.0, delta=1.0, max_segments=100,
        seg_tol_factor=1e-3, N_start_seg=10, inc_mode="double",
):
    counter = FunctionCounter(f_original)
    g = lambda x: np.exp(-x) * counter(x)

    all_segment_nodes_x = []
    all_segment_nodes_y = []

    val_seg0, _, x_n_s0, y_n_s0 = adaptive_simpson(
        g, 0.0, A0, max(eps_tail * seg_tol_factor, 1e-13),
        N_start_seg, mode=inc_mode
    )
    total = val_seg0
    all_segment_nodes_x.extend(x_n_s0)
    all_segment_nodes_y.extend(y_n_s0)

    upper = A0

    for _ in range(max_segments):
        seg_a = upper
        seg_b = upper + delta

        seg_val, _, x_n_curr, y_n_curr = adaptive_simpson(
            g, seg_a, seg_b,
            max(eps_tail * seg_tol_factor, 1e-13),
            N_start_seg, mode=inc_mode,
        )
        total += seg_val
        all_segment_nodes_x.extend(x_n_curr)
        all_segment_nodes_y.extend(y_n_curr)

        upper = seg_b

        if abs(seg_val) < eps_tail:
            return total, counter.calls, upper, np.array(all_segment_nodes_x), np.array(all_segment_nodes_y)

        if upper > 500: break

    return total, counter.calls, upper, np.array(all_segment_nodes_x), np.array(all_segment_nodes_y)


# ───────────────────────────── GAUSS-LAGUERRE ──────────────────────────
def gauss_laguerre(f_original, n):
    data = GAUSS_LAGUERRE_DATA[n]
    counter = FunctionCounter(f_original)
    val = np.sum(data["weights"] * counter(data["nodes"]))
    return val, counter.calls


# ────────────────────────────────── UI  (SIDEBAR) ───────────────────────
st.sidebar.header("Konfiguracja")

fname = st.sidebar.selectbox("Wybierz funkcję f(x):", list(FUNCTIONS.keys()))
f_original, exact_val = FUNCTIONS[fname]

st.sidebar.subheader("Newton-Cotes (Simpson)")
eps_tail = st.sidebar.number_input("Dokładność ε (ogon):", 1e-12, 1e-1, 1e-6, format="%.1e")
A0 = st.sidebar.slider("Początkowy przedział  [0, A]  (A):", 0.1, 20.0, 5.0, 0.1)
delta = st.sidebar.slider("Krok δ do rozszerzania przedziału:", 0.1, 10.0, 1.0, 0.1)
max_seg = st.sidebar.slider("Maks. liczba segmentów:", 10, 500, 100, 10)
inc_mode_radio = st.sidebar.radio(
    "Jak zwiększać liczbę podprzedziałów w adaptacyjnym Simpsonie?",
    ("x2 (szybciej)", "+2 (wolniej, parzyste N)"), index=0
)
inc_mode = "double" if inc_mode_radio.startswith("x2") else "increment"

st.sidebar.subheader("Gauss-Laguerre (Wykres)")
gauss_laguerre_options = list(GAUSS_LAGUERRE_DATA.keys())

# Inicjalizacja wartości w session_state, jeśli jeszcze nie istnieje
if 'n_plot_gl_selection' not in st.session_state:
    st.session_state.n_plot_gl_selection = gauss_laguerre_options[-1]  # Domyślnie ostatnia opcja

# Kontrolka selectbox używa teraz klucza do powiązania z session_state
n_plot_gl = st.sidebar.selectbox(
    "Wybierz liczbę węzłów (n) dla wykresu Gaussa-Laguerre'a:",
    options=gauss_laguerre_options,
    key='n_plot_gl_selection'  # Powiązanie z st.session_state
)


if st.sidebar.button("OBLICZ", type="primary"):
    st.header(f"Wyniki dla  {fname}")

    g_for_plot = lambda x_val: np.exp(-x_val) * f_original(x_val)

    with st.spinner("Obliczanie  Newton-Cotes…"):
        nc_val, nc_calls, nc_upper_final, nc_nodes_x_all, nc_nodes_y_all = newton_cotes_variant3(
            f_original, eps_tail, A0, delta, max_seg,
            N_start_seg=10, inc_mode=inc_mode,
        )
    st.subheader("1 · Newton-Cotes (Simpson)")
    st.write(f"**Wartość:** {pretty(nc_val)}")
    st.write(f"**Wywołania f(x):** {nc_calls}")
    st.write(f"**Górna granica całkowania (A_final):** {pretty(nc_upper_final)}")
    if exact_val is not None:
        st.write(f"**Błąd bezwzględny NC vs dokładna:** {pretty(abs(nc_val - exact_val))}")

    st.subheader("Wykres dla Newton-Cotes (Simpson)")
    fig_nc, ax_nc = plt.subplots()
    x_plot_nc = np.linspace(0, nc_upper_final, 600)
    y_plot_nc = g_for_plot(x_plot_nc)
    ax_nc.plot(x_plot_nc, y_plot_nc, label=r"$g(x) = e^{-x}f(x)$", color="dodgerblue")
    ax_nc.fill_between(x_plot_nc, y_plot_nc, alpha=0.2, color="skyblue")
    if nc_nodes_x_all.size > 0:
        unique_indices = np.sort(np.unique(nc_nodes_x_all, return_index=True)[1])
        plot_nc_nodes_x = nc_nodes_x_all[unique_indices]
        plot_nc_nodes_y = nc_nodes_y_all[unique_indices]
        max_plot_nodes = 500
        if len(plot_nc_nodes_x) > max_plot_nodes:
            step = len(plot_nc_nodes_x) // max_plot_nodes
            plot_nc_nodes_x = plot_nc_nodes_x[::step]
            plot_nc_nodes_y = plot_nc_nodes_y[::step]
        ax_nc.scatter(plot_nc_nodes_x, plot_nc_nodes_y, color='crimson', s=10, zorder=3, label="Węzły Simpsona (użyte)")
    ax_nc.set_title(f"Newton-Cotes dla $g(x) = e^{{-x}} \cdot ${fname.split('=')[1].strip()}")
    ax_nc.set_xlabel("x")
    ax_nc.set_ylabel(r"$g(x)$")
    ax_nc.legend()
    ax_nc.grid(True, linestyle='--', alpha=0.7)
    ax_nc.set_ylim(bottom=min(0, np.min(y_plot_nc) if y_plot_nc.size > 0 else 0) - 0.1 * (
                (np.max(y_plot_nc) if y_plot_nc.size > 0 else 1) - (
            np.min(y_plot_nc) if y_plot_nc.size > 0 else 0)))
    st.pyplot(fig_nc)

    st.markdown("---")

    st.subheader("2 · Gauss-Laguerre  (n = 2 … 5)")
    rows = []
    gl_plot_data = {}
    for n_gl_calc in GAUSS_LAGUERRE_DATA.keys():
        val_gl, calls_gl = gauss_laguerre(f_original, n_gl_calc)
        gl_plot_data[n_gl_calc] = {"nodes": GAUSS_LAGUERRE_DATA[n_gl_calc]["nodes"],
                                   "value_at_nodes": g_for_plot(GAUSS_LAGUERRE_DATA[n_gl_calc]["nodes"])}
        rows.append({
            "n": n_gl_calc,
            "I Gaussa": pretty(val_gl),
            "Błąd NC vs Gauss": pretty(abs(val_gl - nc_val)),
            "Błąd Gauss vs dokładna" if exact_val is not None else "": (
                pretty(abs(val_gl - exact_val)) if exact_val is not None else ""
            ),
            "Wywołania f(x)": calls_gl,
        })
    df = pd.DataFrame(rows).set_index("n")
    st.table(df)

    # n_plot_gl jest teraz pobierane z session_state poprzez klucz w selectboxie
    st.subheader(f"Wykres dla Gauss-Laguerre (n={n_plot_gl})")
    fig_gl, ax_gl = plt.subplots()
    if n_plot_gl in gl_plot_data:
        current_gl_nodes = gl_plot_data[n_plot_gl]["nodes"]
        current_gl_y_at_nodes = gl_plot_data[n_plot_gl]["value_at_nodes"]
        x_max_gl = max(current_gl_nodes) * 1.5 if current_gl_nodes.size > 0 else 10
        x_plot_upper_bound = nc_upper_final if 'nc_upper_final' in locals() and nc_upper_final > 0 else x_max_gl
        x_plot_gl = np.linspace(0, max(x_max_gl, x_plot_upper_bound * 0.3), 500)
        y_plot_gl = g_for_plot(x_plot_gl)
        ax_gl.plot(x_plot_gl, y_plot_gl, label=r"$g(x) = e^{-x}f(x)$", color="dodgerblue")
        ax_gl.fill_between(x_plot_gl, y_plot_gl, alpha=0.2, color="skyblue")
        ax_gl.scatter(current_gl_nodes, current_gl_y_at_nodes, color='forestgreen', s=60, zorder=3, edgecolor='black',
                      label=f"Węzły G-L (n={n_plot_gl})")
        ax_gl.set_title(f"Gauss-Laguerre (n={n_plot_gl}) dla $g(x) = e^{{-x}} \cdot ${fname.split('=')[1].strip()}")
        ax_gl.set_xlabel("x")
        ax_gl.set_ylabel(r"$g(x)$")
        ax_gl.legend()
        ax_gl.grid(True, linestyle='--', alpha=0.7)
        ax_gl.set_ylim(bottom=min(0, np.min(y_plot_gl) if y_plot_gl.size > 0 else 0) - 0.1 * (
                    (np.max(y_plot_gl) if y_plot_gl.size > 0 else 1) - (np.min(y_plot_gl) if y_plot_gl.size > 0 else 0)))
        st.pyplot(fig_gl)
    else:
        st.warning(f"Brak danych do wyświetlenia wykresu dla n={n_plot_gl}.")

    st.markdown("---")
    st.success("Gotowe ✔")
else:
    st.info("Ustaw parametry po lewej stronie i kliknij **OBLICZ**.")