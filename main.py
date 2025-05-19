import streamlit as st
import numpy as np

# --- Dane dla Kwadratury Gaussa-Laguerre'a ---
# Dla całki od 0 do ∞ z e^(-x) * f(x) dx
# Węzły (x_i) i wagi (w_i)
# Źródło: np.polynomial.laguerre.laggauss z biblioteki Numpy dla spójności
# lub standardowe tablice, np. Abramowitz and Stegun.
GAUSS_LAGUERRE_DATA = {
    2: {
        'nodes': np.array([0.58578644, 3.41421356]),
        'weights': np.array([0.85355339, 0.14644661])
    },
    3: {
        'nodes': np.array([0.41577456, 2.29428036, 6.28994508]),
        'weights': np.array([0.71109301, 0.27851773, 0.01038926])
    },
    4: {
        'nodes': np.array([0.32254769, 1.7457611, 4.5366203, 9.39507091]),
        'weights': np.array([0.6031541, 0.35741869, 0.03888791, 0.000539295])
    },
    5: {
        'nodes': np.array([0.26356032, 1.41340306, 3.59642577, 7.08581001, 12.64080084]),
        'weights': np.array([0.52175561, 0.39866681, 0.07594245, 0.00361175, 0.00002337])
    }
}


# --- Funkcje testowe f(x) ---
# Całka ma postać ∫[0,∞) e^(-x) f(x) dx
def f_constant_one(x):
    # Obsługa wejścia tablicowego z węzłów Gaussa
    if isinstance(x, np.ndarray):
        return np.ones_like(x)
    return 1.0


def f_x(x):
    return x


def f_x_squared(x):
    return x ** 2


def f_x_cubed(x):
    return x ** 3


def f_sin_x(x):
    return np.sin(x)


def f_exp_neg_x(x):  # f(x) = e^-x, więc całkujemy e^(-x) * e^(-x) = e^(-2x)
    return np.exp(-x)


# Funkcje do wyboru w interfejsie Streamlit
# Struktura: { "Nazwa Wyświetlana": (obiekt_funkcji, lambda_dla_rozwiazania_analitycznego) }
FUNCTIONS = {
    "f(x) = 1": (f_constant_one, lambda: 1.0),
    "f(x) = x": (f_x, lambda: 1.0),  # Γ(2) = 1!
    "f(x) = x^2": (f_x_squared, lambda: 2.0),  # Γ(3) = 2!
    "f(x) = x^3": (f_x_cubed, lambda: 6.0),  # Γ(4) = 3!
    "f(x) = sin(x)": (f_sin_x, lambda: 0.5),  # ∫ e^(-x)sin(x)dx = 0.5
    "f(x) = exp(-x)": (f_exp_neg_x, lambda: 0.5),  # ∫ e^(-2x)dx = 0.5
}


# --- Metody Całkowania Numerycznego ---

# Podstawowa złożona metoda Simpsona dla całki oznaczonej ∫ func(x) dx od a do b z N podprzedziałami
def simpson_basic(func, a, b, N):
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N musi być dodatnią liczbą całkowitą dla metody Simpsona.")
    if N % 2 != 0:
        # Zgodnie z wymogiem metody Simpsona, N musi być parzyste.
        # Cicho dostosowujemy lub zgłaszamy błąd. Tutaj dostosowujemy.
        N += 1

    if a == b:
        return 0.0
    # Upewnij się, że a < b dla standardowej kolejności sumowania. Jeśli nie, zamień i zaneguj.
    if a > b:
        return -simpson_basic(func, b, a, N)

    h = (b - a) / N
    integral = func(a) + func(b)

    for i in range(1, N):  # Sumowanie po punktach wewnętrznych
        x = a + i * h
        if i % 2 == 1:  # Punkty o nieparzystych indeksach (od 1 do N-1) otrzymują wagę 4
            integral += 4 * func(x)
        else:  # Punkty o parzystych indeksach otrzymują wagę 2
            integral += 2 * func(x)

    integral *= h / 3.0
    return integral


# Adaptacyjna metoda Simpsona dla ∫ func(x) dx od a do b
# Iteracyjnie podwaja N (liczbę podprzedziałów) aż do zbieżności do `tol`
def adaptive_simpson(func, a, b, tol, N_start=10, max_iter_adaptive=15):
    if a == b:
        return 0.0
    if a > b:  # Obsługa odwróconego przedziału
        return -adaptive_simpson(func, b, a, tol, N_start, max_iter_adaptive)

    # Upewnij się, że N_start jest liczbą parzystą dla metody Simpsona
    current_N = N_start if N_start % 2 == 0 else N_start + 1

    val_prev = simpson_basic(func, a, b, current_N)

    for iteration in range(max_iter_adaptive):
        current_N *= 2
        val_curr = simpson_basic(func, a, b, current_N)

        if abs(val_curr - val_prev) < tol:
            return val_curr  # Zbieżność osiągnięta

        val_prev = val_curr

        # Przerwanie bezpieczeństwa, jeśli N stanie się zbyt duże, aby zapobiec problemom z wydajnością
        if current_N > 2 ** 20:  # Około 1 miliona podprzedziałów
            st.warning(
                f"Adaptacyjna metoda Simpsona: Osiągnięto maksymalne N ({current_N}) dla przedziału [{a:.2f},{b:.2f}]. Używam bieżącej wartości.")
            return val_curr

    st.warning(
        f"Adaptacyjna metoda Simpsona: Nie osiągnięto zbieżności do tolerancji {tol:.1e} dla przedziału [{a:.2f},{b:.2f}] po {max_iter_adaptive} iteracjach (N zakończyło się na {current_N}). Używam bieżącej wartości.")
    return val_curr


# Złożona kwadratura Newtona-Cotesa (Simpsona) dla Wariantu 3: ∫[0,∞) e^(-x)f_oryginal(x)dx
def newton_cotes_variant3(f_oryginal, epsilon_uzytkownika,
                          poczatkowy_limit_a=1.0, staly_delta_segmentu=1.0,
                          max_petli_zewnetrznych=100,
                          wspolczynnik_tol_adapt_simpsona_segmentu=0.001,
                          N_poczatkowe_adapt_simpsona=10):
    # Funkcja do całkowania metodą Simpsona to g(x) = e^(-x) * f_oryginal(x)
    g = lambda x_val: np.exp(-x_val) * f_oryginal(x_val)

    calka_calkowita = 0.0
    biezacy_A_gorny = 0.0  # Górny limit już scałkowanej części

    # --- Obliczanie całki na pierwszym segmencie [0, poczatkowy_limit_a] ---
    segment_a_pierwszy = biezacy_A_gorny
    segment_b_pierwszy = poczatkowy_limit_a

    if segment_b_pierwszy > segment_a_pierwszy:  # Całkuj tylko, jeśli przedział jest prawidłowy i ma dodatnią długość
        # Tolerancja dla adaptacyjnej metody Simpsona na tym segmencie:
        # Mały ułamek epsilon_uzytkownika, ale z absolutnym minimum dla precyzji.
        tol_dla_obliczen_segmentu = max(epsilon_uzytkownika * wspolczynnik_tol_adapt_simpsona_segmentu, 1e-13)

        calka_na_segmencie = adaptive_simpson(g, segment_a_pierwszy, segment_b_pierwszy,
                                              tol_dla_obliczen_segmentu,
                                              N_start=N_poczatkowe_adapt_simpsona)
        calka_calkowita += calka_na_segmencie

    biezacy_A_gorny = segment_b_pierwszy  # Aktualizacja bieżącego górnego limitu całkowania

    # --- Iteracyjne dodawanie segmentów [biezacy_A_gorny, biezacy_A_gorny + staly_delta_segmentu] ---
    # Ta pętla rozszerza całkowanie w kierunku +∞
    wartosc_calki_ostatniego_segmentu = float('inf')  # Inicjalizacja do pierwszej kontroli

    for i in range(max_petli_zewnetrznych):
        segment_a = biezacy_A_gorny
        segment_b = biezacy_A_gorny + staly_delta_segmentu

        tol_dla_obliczen_segmentu = max(epsilon_uzytkownika * wspolczynnik_tol_adapt_simpsona_segmentu, 1e-13)

        calka_na_segmencie = adaptive_simpson(g, segment_a, segment_b,
                                              tol_dla_obliczen_segmentu,
                                              N_start=N_poczatkowe_adapt_simpsona)

        calka_calkowita += calka_na_segmencie
        biezacy_A_gorny = segment_b
        wartosc_calki_ostatniego_segmentu = calka_na_segmencie

        # Sprawdzenie zbieżności ogona: jeśli całka na ostatnio dodanym segmencie jest wystarczająco mała
        if abs(calka_na_segmencie) < epsilon_uzytkownika:
            st.info(f"Newton-Cotes: Ogon zbieżny po {i + 1} segmencie(-ach) rozszerzającym(-ych).")
            st.write(f"  Ostatni oceniany segment: [{segment_a:.2f}, {segment_b:.2f}]")
            st.write(f"  Całka na tym ostatnim segmencie: {calka_na_segmencie:.3e}")
            st.write(f"  Całkowity zakres całkowania rozszerzony do A = {biezacy_A_gorny:.2f}")
            return calka_calkowita

        # Przerwanie bezpieczeństwa, jeśli A (górny limit) stanie się zbyt duże bez zbieżności
        if biezacy_A_gorny > 500:  # Heurystyczny limit dla A
            st.warning(
                f"Newton-Cotes: Limit całkowania A przekroczył 500 (obecnie {biezacy_A_gorny:.2f}). Zatrzymuję wcześnie.")
            break

    st.warning(
        f"Newton-Cotes: Osiągnięto maksymalną liczbę pętli zewnętrznych ({max_petli_zewnetrznych}) lub przekroczono limit A przed spełnieniem kryterium zbieżności ogona.")
    st.write(f"  Całkowanie zatrzymane na A = {biezacy_A_gorny:.2f}.")
    st.write(
        f"  Całka na ostatnim obliczonym segmencie: {wartosc_calki_ostatniego_segmentu:.3e} (Docelowy próg ogona: {epsilon_uzytkownika:.1e})")
    return calka_calkowita


# Kwadratura Gaussa-Laguerre'a dla Wariantu 3: ∫[0,∞) e^(-x)f_oryginal(x)dx
def gauss_laguerre(f_oryginal, liczba_wezlow):
    if liczba_wezlow not in GAUSS_LAGUERRE_DATA:
        st.error(f"Dane dla kwadratury Gaussa-Laguerre'a nie są dostępne dla {liczba_wezlow} węzłów.")
        return None  # Lub zgłoś błąd

    data = GAUSS_LAGUERRE_DATA[liczba_wezlow]
    nodes = data['nodes']  # wartości x_i
    weights = data['weights']  # wartości w_i

    # Wzór Gaussa-Laguerre'a: suma(w_i * f_oryginal(x_i))
    # Część e^(-x) jest uwzględniona w wagach i węzłach kwadratury Gaussa-Laguerre'a.
    wartosc_calki = np.sum(weights * f_oryginal(nodes))

    return wartosc_calki


# --- Interfejs Użytkownika Streamlit ---
st.set_page_config(layout="wide")
st.title("Całkowanie Numeryczne: Wariant 3")
st.markdown("Program oblicza całkę `∫[0, +∞) e^(-x) f(x) dx` używając dwóch metod:")
st.markdown("1. **Złożona kwadratura Newtona-Cotesa (metoda Simpsona)**: Iteracyjnie rozszerza przedział całkowania.")
st.markdown("2. **Kwadratura Gaussa-Laguerre'a**: Używa predefiniowanych węzłów i wag.")

st.sidebar.header("Konfiguracja Globalna")

# Wybór funkcji
nazwa_wybranej_funkcji = st.sidebar.selectbox(
    "Wybierz funkcję f(x):",
    list(FUNCTIONS.keys())
)
f_wybrana_funkcja_obj, f_rozwiazanie_analityczne_callable = FUNCTIONS[nazwa_wybranej_funkcji]

# Parametry Newtona-Cotesa
st.sidebar.subheader("Ustawienia Newtona-Cotesa (Simpsona)")
epsilon_uzytkownika_nc = st.sidebar.number_input(
    "Wymagana precyzja (ε) dla zbieżności ogona:",
    min_value=1e-12, max_value=1e-1, value=1e-6, format="%.1e",
    help="Metoda Newtona-Cotesa zatrzymuje rozszerzanie przedziału, gdy całka na ostatnio dodanym segmencie `[A, A+δ]` jest mniejsza od tej wartości."
)
poczatkowa_wartosc_a_nc = st.sidebar.slider(
    "Początkowy przedział `[0, a]` (wartość `a`):",
    min_value=0.1, max_value=20.0, value=5.0, step=0.1,
    help="Długość pierwszego segmentu całkowanego metodą Newtona-Cotesa."
)
delta_segmentu_nc = st.sidebar.slider(
    "Długość segmentu `δ` do rozszerzania przedziału:",
    min_value=0.1, max_value=10.0, value=1.0, step=0.1,
    help="Długość kolejnych segmentów `[A, A+δ]` dodawanych w celu rozszerzenia zakresu całkowania."
)
max_petli_nc_ui = st.sidebar.slider(
    "Maks. liczba segmentów rozszerzających dla Newtona-Cotesa:",
    min_value=10, max_value=500, value=100, step=10
)

# Parametry Gaussa-Laguerre'a
st.sidebar.subheader("Ustawienia Gaussa-Laguerre'a")
liczba_wezlow_gaussa_ui = st.sidebar.selectbox(
    "Liczba węzłów dla kwadratury Gaussa-Laguerre'a:",
    options=list(GAUSS_LAGUERRE_DATA.keys()),
    index=2  # Domyślnie 4 węzły dla ogólnie dobrego wyniku
)

if st.sidebar.button("Oblicz Całkę", type="primary"):
    st.header("Wyniki")

    # Rozwiązanie analityczne (jeśli dostępne)
    wartosc_rozwiazania_analitycznego = f_rozwiazanie_analityczne_callable()
    st.subheader(f"Funkcja Całkowana: `f(x) = {nazwa_wybranej_funkcji.split('=')[1].strip()}`")
    if wartosc_rozwiazania_analitycznego is not None:
        st.write(f"**Rozwiązanie Analityczne (Dokładne):** `{wartosc_rozwiazania_analitycznego:.12f}`")
    else:
        st.write("**Rozwiązanie Analityczne:** Nie podano dla tej funkcji.")
    st.markdown("---")

    # --- Obliczenia Newtona-Cotesa ---
    st.subheader("1. Złożona Kwadratura Newtona-Cotesa (Metoda Simpsona)")
    wsp_tol_segmentu_nc = 0.001
    N_pocz_dla_adapt_simpsona = 10

    with st.spinner("Obliczanie metodą Newtona-Cotesa (Simpsona)... To może chwilę potrwać."):
        nc_wynik = newton_cotes_variant3(
            f_wybrana_funkcja_obj,
            epsilon_uzytkownika_nc,
            poczatkowy_limit_a=poczatkowa_wartosc_a_nc,
            staly_delta_segmentu=delta_segmentu_nc,
            max_petli_zewnetrznych=max_petli_nc_ui,
            wspolczynnik_tol_adapt_simpsona_segmentu=wsp_tol_segmentu_nc,
            N_poczatkowe_adapt_simpsona=N_pocz_dla_adapt_simpsona
        )
    st.write(f"**Wynik Newtona-Cotesa:** `{nc_wynik:.12f}`")
    if wartosc_rozwiazania_analitycznego is not None:
        nc_blad = abs(nc_wynik - wartosc_rozwiazania_analitycznego)
        st.write(f"**Błąd Bezwzględny (względem analitycznego):** `{nc_blad:.3e}`")
    st.markdown("---")

    # --- Obliczenia Gaussa-Laguerre'a ---
    st.subheader("2. Kwadratura Gaussa-Laguerre'a")
    with st.spinner(f"Obliczanie kwadraturą Gaussa-Laguerre'a ({liczba_wezlow_gaussa_ui} węzłów)..."):
        gl_wynik = gauss_laguerre(f_wybrana_funkcja_obj, liczba_wezlow_gaussa_ui)

    if gl_wynik is not None:
        st.write(f"**Wynik Gaussa-Laguerre'a ({liczba_wezlow_gaussa_ui} węzłów):** `{gl_wynik:.12f}`")
        if wartosc_rozwiazania_analitycznego is not None:
            gl_blad = abs(gl_wynik - wartosc_rozwiazania_analitycznego)
            st.write(f"**Błąd Bezwzględny (względem analitycznego):** `{gl_blad:.3e}`")
    else:
        st.write("Obliczenia Gaussa-Laguerre'a nie powiodły się lub dane nie są dostępne dla wybranej liczby węzłów.")

    st.markdown("---")
    st.success("Obliczenia zakończone!")

else:
    st.info("Skonfiguruj parametry na panelu bocznym i kliknij 'Oblicz Całkę', aby rozpocząć.")