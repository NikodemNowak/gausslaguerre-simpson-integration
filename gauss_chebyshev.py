import numpy as np

def gauss_chebyshev_first_kind_integration(func_numerator, num_nodes):
    if not isinstance(num_nodes, int) or num_nodes <= 0:
        return None, ["Liczba węzłów (num_nodes) musi być dodatnią liczbą całkowitą."]

    messages = []

    try:
        nodes = np.array([np.cos((2 * k + 1) * np.pi / (2 * num_nodes)) for k in range(num_nodes)])
        function_values_at_nodes = np.array([func_numerator(node) for node in nodes])

        if np.any(np.isnan(function_values_at_nodes)) or np.any(np.isinf(function_values_at_nodes)):
            messages.append(
                "Ostrzeżenie (Gauss-Czebyszew): func_numerator zwróciła NaN lub Inf dla co najmniej jednego węzła.")

        sum_of_function_values = np.sum(function_values_at_nodes)
        integral_value = (np.pi / num_nodes) * sum_of_function_values

        if np.isnan(integral_value) or np.isinf(integral_value):
            messages.append("Ostrzeżenie (Gauss-Czebyszew): Wynikowa całka to NaN lub Inf.")

        return integral_value, messages

    except Exception as e:
        error_msg = f"Błąd w gauss_chebyshev_first_kind_integration: {e}"
        return None, [error_msg]