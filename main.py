import streamlit as st
import numpy as np
from newton_cotes import basic_simpson, singularity_simpson
from gauss_chebyshev import gauss_chebyshev_first_kind_integration


# --- Funkcje testowe f_oryginal(x) dla całki ∫[-1,1] [f_oryginal(x) / √(1-x²)] dx ---

def f_const_one(x):
    # Obsługa wejścia tablicowego z węzłów Gaussa
    if isinstance(x, np.ndarray):
        return np.ones_like(x)
    return 1.0


def f_linear_x(x):
    return x


def f_quadratic_x_sq(x):
    return x ** 2


def f_cubic_x_cubed(x):
    return x ** 3


def f_T2_chebyshev(x):  # T_2(x) = 2x^2 - 1
    return 2 * x ** 2 - 1


def f_exp_x(x):
    return np.exp(x)


# Słownik funkcji do wyboru w interfejsie Streamlit
# Klucz: Nazwa wyświetlana
# Wartość: (obiekt_funkcji_f_oryginal, lambda_dla_rozwiazania_analitycznego_CALKI_CZEBYSZEWA)
FUNCTIONS_CHEBYSHEV = {
    "f(x) = 1": (f_const_one, lambda: np.pi),  # ∫ 1/sqrt(1-x^2) dx = π
    "f(x) = x": (f_linear_x, lambda: 0.0),  # ∫ x/sqrt(1-x^2) dx = 0
    "f(x) = x^2": (f_quadratic_x_sq, lambda: np.pi / 2.0),  # ∫ x^2/sqrt(1-x^2) dx = π/2
    "f(x) = x^3": (f_cubic_x_cubed, lambda: 0.0),  # ∫ x^3/sqrt(1-x^2) dx = 0
    "f(x) = T_2(x) = 2x^2 - 1": (f_T2_chebyshev, lambda: 0.0),  # ∫ T_2(x)/sqrt(1-x^2) dx = 0 (bo T_0=1, ortogonalność)
    "f(x) = exp(x)": (f_exp_x, lambda: np.pi * np.i0(1))  # ∫ exp(x)/sqrt(1-x^2) dx = π * I_0(1) (BesselI)
}

# --- Interfejs Użytkownika Streamlit ---
st.set_page_config(layout="wide")
st.title("Całkowanie Numeryczne: Wariant 1 (Czebyszew)")
st.markdown("Program oblicza całkę `∫[-1, 1] f(x) / √(1-x²) dx` używając dwóch metod:")
st.markdown("1. **Złożona kwadratura Newtona-Cotesa (metoda Simpsona z obsługą osobliwości)**")
st.markdown("2. **Kwadratura Gaussa-Czebyszewa (pierwszego rodzaju)**")

st.sidebar.header("Konfiguracja Globalna")

# Wybór funkcji f(x) (licznika)
nazwa_wybranej_funkcji = st.sidebar.selectbox(
    "Wybierz funkcję licznika f(x):",
    list(FUNCTIONS_CHEBYSHEV.keys())
)
f_oryginal_wybrana, f_rozwiazanie_analityczne_callable = FUNCTIONS_CHEBYSHEV[nazwa_wybranej_funkcji]

# --- Parametry dla Metody Newtona-Cotesa (Simpsona z obsługą osobliwości) ---
st.sidebar.subheader("Ustawienia Newtona-Cotesa (Simpsona)")
st.sidebar.markdown("Używa `singularity_simpson` do obsługi osobliwości na `±1`.")

# Parametry dla singularity_simpson
n_sp_dla_sing_simpson = st.sidebar.number_input(
    "Liczba podprzedziałów Simpsona dla każdego segmentu w `singularity_simpson` (n_segment_per_piece):",
    min_value=2, max_value=200, value=32, step=2
)
tol_dla_sing_simpsona = st.sidebar.number_input(
    "Tolerancja dla 'singularity_simpson' (tolerance):",
    min_value=1e-12, max_value=1e-1, value=1e-6, format="%.1e"
)
max_steps_dla_sing_simpsona = st.sidebar.number_input(
    "Maks. liczba kroków 'podchodzenia' w `singularity_simpson` (max_steps):",
    min_value=10, max_value=500, value=100, step=10
)

# --- Parametry dla Kwadratury Gaussa-Czebyszewa ---
st.sidebar.subheader("Ustawienia Gaussa-Czebyszewa")
liczba_wezlow_gaussa_ui = st.sidebar.number_input(
    "Liczba węzłów dla kwadratury Gaussa-Czebyszewa (num_nodes):",
    min_value=1, max_value=50, value=5, step=1
)

if st.sidebar.button("Oblicz Całkę", type="primary"):
    st.header("Wyniki")

    # Rozwiązanie analityczne
    wartosc_rozwiazania_analitycznego = f_rozwiazanie_analityczne_callable()
    st.subheader(f"Całka: `∫[-1,1] ({nazwa_wybranej_funkcji.split('=')[1].strip()}) / √(1-x²) dx`")
    if wartosc_rozwiazania_analitycznego is not None:
        st.write(f"**Rozwiązanie Analityczne (Dokładne):** `{wartosc_rozwiazania_analitycznego:.12f}`")
    else:
        st.write("**Rozwiązanie Analityczne:** Nie podano dla tej funkcji lub jest złożone.")
    st.markdown("---")

    # --- Obliczenia Newtona-Cotesa (singularity_simpson) ---
    st.subheader("1. Złożona Kwadratura Newtona-Cotesa (Metoda Simpsona z obsługą osobliwości)")


    # Definicja pełnej funkcji podcałkowej dla Simpsona: g(x) = f_oryginal(x) / sqrt(1-x^2)
    def g_for_simpson(x_val):
        if np.isclose(abs(x_val), 1.0):
            # singularity_simpson powinien unikać wywoływania basic_simpson dokładnie na krańcach
            # ale na wszelki wypadek, jeśli func zostanie tam wywołana
            return float('inf')
        if abs(x_val) > 1.0:  # Poza dziedziną
            return float('nan')
        denominator_sq = 1.0 - x_val ** 2
        if denominator_sq <= 1e-15:  # Bardzo blisko lub w osobliwości
            return float('inf')
        return f_oryginal_wybrana(x_val) / np.sqrt(denominator_sq)


    with st.spinner("Obliczanie metodą Newtona-Cotesa (Simpsona)..."):
        # Wywołujemy singularity_simpson z pełną funkcją podcałkową g_for_simpson
        # Funkcja singularity_simpson w Twojej wersji zwraca tylko wynik, bez komunikatów
        # Dla uproszczenia, na razie nie będziemy przechwytywać print() z niej.
        # Jeśli chcesz komunikaty, singularity_simpson musiałby je zwracać.
        nc_wynik = singularity_simpson(
            g_for_simpson,
            -1.0,
            1.0,
            n_segment_per_piece=n_sp_dla_sing_simpson,
            tolerance=tol_dla_sing_simpsona,
            max_steps=max_steps_dla_sing_simpsona
        )

    st.write(f"**Wynik Newtona-Cotesa (singularity_simpson):** `{nc_wynik:.12f}`")
    if wartosc_rozwiazania_analitycznego is not None and nc_wynik is not None and not np.isnan(
            nc_wynik) and not np.isinf(nc_wynik):
        nc_blad = abs(nc_wynik - wartosc_rozwiazania_analitycznego)
        st.write(f"**Błąd Bezwzględny (względem analitycznego):** `{nc_blad:.3e}`")
    st.markdown("---")

    # --- Obliczenia Gaussa-Czebyszewa ---
    st.subheader("2. Kwadratura Gaussa-Czebyszewa")
    with st.spinner(f"Obliczanie kwadraturą Gaussa-Czebyszewa ({liczba_wezlow_gaussa_ui} węzłów)..."):
        # Przekazujemy tylko f_oryginal_wybrana (licznik)
        gc_wynik, gc_komunikaty = gauss_chebyshev_first_kind_integration(
            f_oryginal_wybrana,
            liczba_wezlow_gaussa_ui
        )

    for msg in gc_komunikaty:
        if "Ostrzeżenie" in msg:
            st.warning(msg)
        else:
            st.error(msg)

    if gc_wynik is not None:
        st.write(f"**Wynik Gaussa-Czebyszewa ({liczba_wezlow_gaussa_ui} węzłów):** `{gc_wynik:.12f}`")
        if wartosc_rozwiazania_analitycznego is not None and not np.isnan(gc_wynik) and not np.isinf(gc_wynik):
            gc_blad = abs(gc_wynik - wartosc_rozwiazania_analitycznego)
            st.write(f"**Błąd Bezwzględny (względem analitycznego):** `{gc_blad:.3e}`")
    else:
        st.error("Obliczenia kwadraturą Gaussa-Czebyszewa nie powiodły się.")

    st.markdown("---")
    st.success("Obliczenia zakończone!")

else:
    st.info("Skonfiguruj parametry na panelu bocznym i kliknij 'Oblicz Całkę', aby rozpocząć.")