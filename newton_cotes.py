import numpy as np


def basic_simpson(func, a, b, n):

    # Walidacja danych

    if not isinstance(n, int) or n <= 0:
        raise ValueError("N musi być dodatnią liczbą całkowitą dla metody Simpsona.")
    if n % 2 != 0:
        raise ValueError("N musi być parzysta.")
    if a == b:
        return 0.0
    if a > b:
        return -basic_simpson(func, b, a, n) # Ponieważ całka f(x) od a do b to to samo co - całka f(x) od b do a

    h = (b - a) / n

    integral = func(a) + func(b) # Dla urposzczenia najpierw całka to krańce przedziałów

    for i in range(1, n):
        x = a + i * h
        if i % 2 == 1:
            integral += 4 * func(x)
        else:
            integral += 2 * func(x)

    integral *= h / 3.0 # Na koniec jeszcze całość razy to h/3

    return integral

def singularity_simpson(func, a, b, n_segment_per_piece = 32, tolerance = 0.001, max_steps = 128):

    # Walidacja danych

    if not isinstance(n_segment_per_piece, int) or n_segment_per_piece <= 0 or n_segment_per_piece % 2 == 1:
        raise ValueError("N_segment_per_piece musi być dodatnią parzystą liczbą całkowitą.")

    if a == b:
        return 0.0

    if a > b:
        return -singularity_simpson(func, a, b, n_segment_per_piece, tolerance, max_steps)

    total_integral = 0.0

    # Najpierw sprawdzamy lewy kraniec przedziału, czyli a. Zaczynamy podchodzić od środka przedziału
    mid_point = (a + b) / 2.0

    current_right_boundary = mid_point

    for _ in range(max_steps):
        current_left_boundary = (current_right_boundary + a) / 2.0

        segment_integral = basic_simpson(func, current_left_boundary, current_right_boundary, n_segment_per_piece)

        total_integral += segment_integral
        current_right_boundary = current_left_boundary

        if abs(segment_integral) < tolerance:
            break
    else: # Czyli jak nie doszło do break w pętli
        print(f"Ostrzeżenie: Osiągnięto max_steps ({max_steps}) dla lewego krańca na [{a},{b}]. Ostatni prawy kraniec: {current_right_boundary}")

    # Teraz prawy kraniec
    current_left_boundary = mid_point

    for i in range(max_steps):
        current_right_boundary = (current_left_boundary + b) / 2.0
        segment_integral = basic_simpson(func, current_left_boundary, current_right_boundary, n_segment_per_piece)
        total_integral += segment_integral

        current_left_boundary = current_right_boundary

        if abs(segment_integral) < tolerance:
            break
    else:
        print(f"Ostrzeżenie: Osiągnięto max_steps ({max_steps}) dla prawego krańca na [{a},{b}]. Ostatni lewy kraniec: {current_left_boundary}")

    return total_integral
