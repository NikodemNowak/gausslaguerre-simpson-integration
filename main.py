# app.py  ────────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import pandas as pd

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
def f_x_squared(x):    return x**2
def f_x_cubed(x):      return x**3
def f_sin_x(x):        return np.sin(x)
def f_exp_neg_x(x):    return np.exp(-x)

FUNCTIONS = {
    "f(x) = 1":          (f_constant_one, 1.0),
    "f(x) = x":          (f_x, 1.0),       # Γ(2)=1
    "f(x) = x^2":        (f_x_squared, 2.0),  # Γ(3)=2
    "f(x) = x^3":        (f_x_cubed, 6.0),  # Γ(4)=6
    "f(x) = sin(x)":     (f_sin_x, 0.5),
    "f(x) = exp(-x)":    (f_exp_neg_x, 0.5),
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
    if N % 2 == 1: N += 1
    h = (b - a) / N
    S = func(a) + func(b)
    for i in range(1, N):
        x = a + i * h
        S += 4 * func(x) if i % 2 else 2 * func(x)
    return S * h / 3.0

# ───────────────────────────── ADAPTACYJNY  SIMPSON ────────────────────
def adaptive_simpson(func, a, b, tol, N_start=10, max_iter=15, mode="double"):
    N = N_start if N_start % 2 == 0 else N_start + 1
    prev = simpson_basic(func, a, b, N)
    for _ in range(max_iter):
        N = N * 2 if mode == "double" else N + 2
        curr = simpson_basic(func, a, b, N)
        if abs(curr - prev) < tol: return curr
        prev = curr
        if N > 2**20: break
    return curr

# ──────────────────── NEWTON-COTES  z OGONEM  DO  ∞  ───────────────────
def newton_cotes_variant3(
    f_original, eps_tail, A0=5.0, delta=1.0, max_segments=100,
    seg_tol_factor=1e-3, N_start_seg=10, inc_mode="double",
):
    counter = FunctionCounter(f_original)
    g = lambda x: np.exp(-x) * counter(x)

    total = adaptive_simpson(
        g, 0.0, A0, max(eps_tail * seg_tol_factor, 1e-13),
        N_start_seg, mode=inc_mode
    )
    upper = A0

    for _ in range(max_segments):
        seg_val = adaptive_simpson(
            g, upper, upper + delta,
            max(eps_tail * seg_tol_factor, 1e-13),
            N_start_seg, mode=inc_mode,
        )
        total += seg_val
        upper += delta
        if abs(seg_val) < eps_tail:
            return total, counter.calls
        if upper > 500: break
    return total, counter.calls

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
inc_mode = st.sidebar.radio(
    "Jak zwiększać liczbę podprzedziałów w adaptacyjnym Simpsonie?",
    ("x2 (szybciej)", "+2 (wolniej)"),
)
inc_mode = "double" if inc_mode.startswith("x2") else "increment"

if st.sidebar.button("OBLICZ", type="primary"):
    st.header(f"Wyniki dla  {fname}")

    # ─────────── Newton-Cotes
    with st.spinner("Obliczanie  Newton-Cotes…"):
        nc_val, nc_calls = newton_cotes_variant3(
            f_original, eps_tail, A0, delta, max_seg,
            N_start_seg=10, inc_mode=inc_mode,
        )
    st.subheader("1 · Newton-Cotes (Simpson)")
    st.write(f"**Wartość:** {pretty(nc_val)}")
    st.write(f"**Wywołania f(x):** {nc_calls}")
    if exact_val is not None:
        st.write(f"**Błąd bezwzględny NC vs dokładna:** {pretty(abs(nc_val-exact_val))}")

    st.markdown("---")

    # ─────────── Gauss-Laguerre
    st.subheader("2 · Gauss-Laguerre  (n = 2 … 5)")
    rows = []
    for n in range(2, 6):
        val, calls = gauss_laguerre(f_original, n)
        rows.append({
            "n": n,
            "I Gaussa":               pretty(val),
            "Błąd NC vs Gauss":       pretty(abs(val - nc_val)),
            "Błąd Gauss vs dokładna" if exact_val is not None else "": (
                pretty(abs(val - exact_val)) if exact_val is not None else ""
            ),
            "Wywołania f(x)": calls,
        })
    df = pd.DataFrame(rows).set_index("n")
    st.table(df)

    st.markdown("---")
    st.success("Gotowe ✔")
else:
    st.info("Ustaw parametry po lewej stronie i kliknij **OBLICZ**.")
