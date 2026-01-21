import pandas as pd
import numpy as np
import csv
import re
import math


bob_raw  = "21-BOB.log"
mbw_raw  = "21-MBW.log"
pt_raw   = "21-PT100.log"
baro_raw = "21-BARO.txt"  

BESTAND_UIT = "21_gecombineerd_RH_PT100_corr.xlsx"


# FUNCTIES OM LOGFILES IN TE LEZEN + OPSCHONEN

def load_raw_log(path):
    """Algemene loader voor .log files (BOB/MBW):
       - tab-gescheiden
       - haalt lege kolommen aan het einde weg
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        gen = (line.rstrip("\n").rstrip("\r").rstrip("\t") for line in f)
        reader = csv.reader(gen, delimiter="\t")
        header = next(reader)

        for r in reader:
            # lege kolommen aan het einde weghalen
            while r and r[-1] == "":
                r.pop()
            if len(r) == len(header):
                rows.append(r)

    return pd.DataFrame(rows, columns=header)


def normalize_datetime(df):
    """Zoek kolom met datum/tijd, hernoem naar 'DateTime' en maak er een
       pandas datetime van. Sorteer en verwijder dubbele tijden.
    """
    candidates = [c for c in df.columns if re.search("date|time", c, re.IGNORECASE)]
    if not candidates:
        raise ValueError("Geen DateTime kolom gevonden")

    dtcol = candidates[0]
    df = df.rename(columns={dtcol: "DateTime"})
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime")
    df = df[~df["DateTime"].duplicated(keep="first")]
    return df


def numeric_cleanup(df):
    """Maak alle kolommen behalve DateTime numeriek:
       - verwijder vreemde tekens
       - vervang komma's door punten indien nodig
    """
    def clean_float(x):
        if pd.isna(x):
            return np.nan
        x = re.sub(r"[^0-9\.,eE\-\+]", "", str(x))
        if "," in x and "." in x:
            x = x.replace(",", "")
        elif "," in x:
            x = x.replace(",", ".")
        return x

    for col in df.columns:
        if col != "DateTime":
            df[col] = df[col].map(clean_float)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# INLEZEN + OPSCHONEN (BOB, MBW, PT100)

bob = numeric_cleanup(normalize_datetime(load_raw_log(bob_raw)))
mbw = numeric_cleanup(normalize_datetime(load_raw_log(mbw_raw)))
pt  = numeric_cleanup(normalize_datetime(load_raw_log(pt_raw)))


# PT100 WEERSTAND → T90 (°C) MET b-POLYNOMEN

PT_B_COEFFS = {
    1: [-2.462119e2, 2.375126e0, 8.425819e-4, 4.294399e-7, 4.419203e-10],
    2: [-2.471460e2, 2.395072e0, 7.120757e-4, 5.036847e-7, 1.644349e-9],
    3: [-2.461255e2, 2.374504e0, 8.421668e-4, 4.297542e-7, 4.461535e-10],
    4: [-2.466664e2, 2.387776e0, 7.509422e-4, 4.774154e-7, 1.261988e-9],
    5: [-2.469900e2, 2.393200e0, 7.196400e-4, 4.970600e-7, 1.549200e-9],
    6: [-2.467400e2, 2.388700e0, 7.448500e-4, 4.852600e-7, 1.371100e-9],
}

def R_to_t90(R, ch):
    if pd.isna(R):
        return np.nan
    t = 0
    for i, b in enumerate(PT_B_COEFFS[ch]):
        t += b * (R ** i)
    return t


# Selecteer T1.OHMS t/m T6.OHMS
ohms_cols = [c for c in pt.columns if re.match(r"^T[1-6]\.OHMS", c)]

corr_temp_cols = []
for col in ohms_cols:
    ch = int(re.match(r"^T([1-6])", col).group(1))
    out_col = f"T{ch}_corr (°C)"
    pt[out_col] = pt[col].apply(lambda R, c=ch: R_to_t90(R, c))
    corr_temp_cols.append(out_col)

# Klein PT100-frame met alleen DateTime + gecorrigeerde T's
pt_small = pt[["DateTime"] + corr_temp_cols].copy()

# Gemiddelde gecorrigeerde temperatuur
pt_small["PT100_mean_T1_6 (°C)"] = pt_small[corr_temp_cols].mean(axis=1)

# Temperatuurgradiënt in de kamer (max - min van de 6 sensoren)
pt_small["PT100_gradient (K)"] = (
    pt_small[corr_temp_cols].max(axis=1)
    - pt_small[corr_temp_cols].min(axis=1)
)

# MERGEN (BOB + MBW + PT100)

comb = bob.merge(mbw, on="DateTime", how="inner", suffixes=("_BOB", "_MBW"))
comb = comb.merge(pt_small, on="DateTime", how="inner").sort_values("DateTime")

# BOB + MBW liepen 1 uur achter → alle tijden 1 uur naar voren
comb["DateTime"] = comb["DateTime"] + pd.Timedelta(hours=1)

# Keuze voor dew- of frostpoint voor mirror temp

def kies_mbw_temp(row):
    mirror = row.get("Mirror Temp (°C)")
    td     = row.get("Dew Point (°C)")
    tf     = row.get("Frost Point (°C)")

    # als iets ontbreekt: val terug op mirror
    if pd.isna(mirror) or pd.isna(tf) or pd.isna(td):
        return mirror

    d_dew   = abs(mirror - td)
    d_frost = abs(mirror - tf)

    # Als mirror dichter bij dewpoint ligt -> gebruik frostpoint
    if d_dew < d_frost:
        return tf

    return mirror

comb["MBW_input_raw (°C)"] = comb.apply(kies_mbw_temp, axis=1)


# BAROMETER INLEZEN EN DICHTSTBIJZIJNDE DRUK KOPPELEN

baro = pd.read_csv(
    baro_raw,
    header=None,
    names=["DateTime", "P_hPa"],   
)

baro["DateTime"] = pd.to_datetime(baro["DateTime"], dayfirst=True)
baro = baro.sort_values("DateTime")
comb = comb.sort_values("DateTime")

# dichtstbijzijnde barometerpunt per meetpunt
comb = pd.merge_asof(
    comb,
    baro,
    on="DateTime",
    direction="nearest"
)

comb["P_lab_Pa"] = comb["P_hPa"] * 100.0


# CMH-CORRECTIE 

def mbw_345_corrected_dp(t_uut):
    a0 = 8.2378e-3
    a1 = 9.9973e-1
    a2 = -2.7832e-5
    a3 = 3.3330e-7
    return a0 + a1*t_uut + a2*t_uut**2 + a3*t_uut**3

comb["MBW345_dp_corr (°C)"]   = comb["MBW_input_raw (°C)"].apply(mbw_345_corrected_dp)
comb["MBW345_dp_offset (°C)"] = comb["MBW345_dp_corr (°C)"] - comb["MBW_input_raw (°C)"]


# HARDY + ENHANCEMENT FACTOR + RH-BEREKENING

# Hardy-coëfficiënten water
g = [
    -2.8365744e3, -6.028076559e3, 1.954263612e1,
    -2.737830188e-2, 1.6261698e-5, 7.0229056e-10,
    -1.8680009e-13
]
g7 = 2.7150305

# Hardy-coëfficiënten ijs
k = [
    -5.8666426e3, 2.232870244e1, 1.39387003e-2,
    -3.4262402e-5, 2.7040955e-8
]
k5 = 6.7063522e-1

def es_pa(T_C):
    T = T_C + 273.15
    if T_C >= 0:
        s = sum(g[i] * T**(i-2) for i in range(7)) + g7 * math.log(T)
    else:
        s = sum(k[i] * T**(i-1) for i in range(5)) + k5 * math.log(T)
    return math.exp(s)

def enhancement_factor(T_C, P, e):
    T = T_C + 273.15
    if T_C >= 0:
        a = [-1.6302041e-1,1.8071570e-3,-6.7703064e-6,8.5813609e-9]
        b = [-5.9890467e1,3.4378043e-1,-7.7326396e-4,6.3405286e-7]
    else:
        a = [-7.1044201e-2,8.6786223e-4,-3.5912529e-6,5.0194210e-9]
        b = [-8.2308868e1,5.6519110e-1,-1.5304505e-3,1.5395086e-6]
    alpha = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3
    beta  = math.exp(b[0] + b[1]*T + b[2]*T**2 + b[3]*T**3)
    return math.exp((1 - e/P)*alpha + ((P/e) - 1)*beta)

def rh_from_frostpoint(Tf, T, P):
    e    = es_pa(Tf)
    es_T = es_pa(T)
    f_e   = enhancement_factor(Tf, P, e)
    f_esT = enhancement_factor(T,  P, es_T)
    return 100 * (e * f_e) / (es_T * f_esT)

# RH berekenen

T_PT_COL = "PT100_mean_T1_6 (°C)"

comb["RH_raw (%)"] = comb.apply(
    lambda r: rh_from_frostpoint(
        r["MBW_input_raw (°C)"], r[T_PT_COL], r["P_lab_Pa"]
    ),
    axis=1
)

comb["RH_corr (%)"] = comb.apply(
    lambda r: rh_from_frostpoint(
        r["MBW345_dp_corr (°C)"], r[T_PT_COL], r["P_lab_Pa"]
    ),
    axis=1
)

comb.to_excel(BESTAND_UIT, index=False)
