import math

def qho_average_energy(beta, hf, n_terms):
    """
    computing the average energy <E> for a quantum harmonic oscillator using n_terms.

    using energy levels:
    E_n = hf (n + 1/2)

    using Boltzmann weights:
    weight = exp(-beta * E_n)

    computing the partial sums (0 to n_terms-1) using closed-form geometric series
    to avoid slow loops for large n_terms.
    """

    # stopping if the number of terms is invalid
    if n_terms <= 0:
        print("n_terms must be a positive integer.")
        return None

    # stopping if beta is negative (this is not physical for equilibrium)
    if beta < 0:
        print("beta must be >= 0.")
        return None

    # handling the beta = 0 limit separately (all weights become 1)
    if beta == 0:
        # computing Z = sum_{n=0}^{N-1} 1 = N
        Z = float(n_terms)

        # computing sum_{n=0}^{N-1} (n + 1/2) = N(N-1)/2 + N/2 = N^2/2
        numerator = hf * (n_terms ** 2) / 2.0

        # computing the average energy
        return numerator / Z

    # computing x = beta * hf
    x = beta * hf

    # computing r = exp(-x)
    r = math.exp(-x)

    # computing (1 - r) in a stable way
    one_minus_r = -math.expm1(-x)

    # computing r^N safely (this will underflow to 0 for huge N, which is fine)
    rN = math.exp(-x * n_terms)

    # computing S0 = sum_{n=0}^{N-1} r^n = (1 - r^N) / (1 - r)
    S0 = (1.0 - rN) / one_minus_r

    # computing S1 = sum_{n=0}^{N-1} n r^n
    # using the closed-form: (r - N r^N + (N-1) r^{N+1}) / (1 - r)^2
    rN1 = rN * r
    denom_sq = one_minus_r ** 2
    S1 = (r - n_terms * rN + (n_terms - 1) * rN1) / denom_sq

    # computing <E> = hf * (S1/S0 + 1/2)
    avg_E = hf * (S1 / S0 + 0.5)

    return avg_E


# running the script interactively when executed directly
if __name__ == "__main__":
    # asking for hf with a default
    hf_text = input("enter hf (default 1): ").strip()
    hf = float(hf_text) if hf_text else 1.0

    # asking for beta with a default
    beta_text = input("enter beta (default 0.01): ").strip()
    beta = float(beta_text) if beta_text else 0.01

    # asking for term counts with a default
    terms_text = input("enter term counts as comma-separated integers (default 1000,1000000,1000000000): ").strip()
    if terms_text:
        terms_list = [int(s.strip()) for s in terms_text.split(",") if s.strip()]
    else:
        terms_list = [1000, 1000000, 1000000000]

    # printing a clean header
    print("\nquantum harmonic oscillator average energy")
    print(f"hf = {hf}, beta = {beta}\n")

    # computing and printing results for each term count
    for N in terms_list:
        E_avg = qho_average_energy(beta, hf, N)
        print(f"n_terms = {N}")
        print(f"<E> = {E_avg}\n")
