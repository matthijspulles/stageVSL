import numpy as np

data = {
    -10: {
        "RH_ref": np.array([
            9.984350445,
            24.59705078,
            48.99997082,
            73.46028597,
            93.03131563,
        ]),
        "RH_uut": np.array([
            9.013894834,
            22.8891449,
            46.00111246,
            68.20885656,
            85.2538294,
        ]),
    },
    1: {
        "RH_ref": np.array([
            5.199751919,
            24.93553167,
            49.35428633,
            74.00326668,
            93.65244946,
        ]),
        "RH_uut": np.array([
            5.055014335,
            25.45929362,
            50.69962928,
            75.22051539,
            94.27559854,
        ]),
    },
    21: {
        "RH_ref": np.array([
            5.022707519,
            24.84546239,
            49.56419791,
            74.27093063,
            93.91881755,
        ]),
        "RH_uut": np.array([
            4.751666667,
            25.03457695,
            50.26755109,
            74.7496871,
            94.18128512,
        ]),
    },
    23: {
        "RH_ref": np.array([
            5.075842939,
            24.88531929,
            49.63280229,
            74.34312007,
            94.08831243,
        ]),
        "RH_uut": np.array([
            4.79016129,
            25.05148946,
            50.31158367,
            74.9320284,
            94.29235359,
        ]),
    },
    50: {
        "RH_ref": np.array([
            5.111844816,
            25.02805329,
            49.90049156,
            74.43291146,
            93.69285781,
        ]),
        "RH_uut": np.array([
            4.680649634,
            24.83474094,
            49.70109464,
            74.27895476,
            92.64802448,
        ]),
    },
    85: {
        "RH_ref": np.array([
            5.048221469,
            25.03538028,
            50.09094379,
            75.18436543,
            95.35633009,
        ]),
        "RH_uut": np.array([
            4.292842548,
            24.87740183,
            49.78162374,
            75.51744357,
            94.43911111,
        ]),
    },
}


# Polynoom fitten per temperatuur
# Vorm: RH_uut = a0 + a1 * RH_ref + a2 * RH_ref^2

poly_order = 2
coeffs = {}

for T, d in data.items():
    x = d["RH_ref"]
    y = d["RH_uut"]

    a2, a1, a0 = np.polyfit(x, y, poly_order)

    coeffs[T] = np.array([a0, a1, a2])

print("Polynoomecoëfficiënten per temperatuur (RH_uut = a0 + a1*RH_ref + a2*RH_ref^2):")
for T in sorted(coeffs.keys()):
    a0, a1, a2 = coeffs[T]
    print(f"T = {T:>3} °C: a0 = {a0}, a1 = {a1}, a2 = {a2}")

def interpolate_coeffs(T_low, T_high, T_target, coeff_dict):
    """Lineaire interpolatie van polynoomecoëfficiënten in temperatuur."""
    c_low = coeff_dict[T_low]
    c_high = coeff_dict[T_high]
    alpha = (T_target - T_low) / (T_high - T_low)
    return c_low + alpha * (c_high - c_low)

def eval_poly(coeffs_T, rh_ref):
    """Bereken RH_uut uit RH_ref met coëfficiënten [a0, a1, a2]."""
    a0, a1, a2 = coeffs_T
    rh_ref = np.asarray(rh_ref)
    return a0 + a1 * rh_ref + a2 * rh_ref**2

rh_grid_all = np.array([12.0, 30.0, 50.0, 70.0, 95.0])

for T in sorted(coeffs.keys()):
    c_T = coeffs[T]
    rh_uut_grid = eval_poly(c_T, rh_grid_all)
    print(f"\nT = {T} °C:")
    for rh, uut in zip(rh_grid_all, rh_uut_grid):
        print(f"  RH_ref = {rh} %  ->  RH_uut(model) ≈ {uut} %")

targets = {
    -9: (-10, 1),  
    70: (50, 85),  
}

for T_target, (T_low, T_high) in targets.items():
    c_target = interpolate_coeffs(T_low, T_high, T_target, coeffs)

    rh_points = rh_grid_all.copy()
    rh_uut_pred = eval_poly(c_target, rh_points)
    a0, a1, a2 = c_target

    print(f"\nGeïnterpoleerde relatie bij T = {T_target} °C ")
    for rh, uut in zip(rh_points, rh_uut_pred):
        print(f"  RH_ref = {rh:>5.1f} %  ->  RH_uut(model) ≈ {uut:.3f} %")
