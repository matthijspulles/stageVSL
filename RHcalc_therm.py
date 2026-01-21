import pandas as pd
import numpy as np
import csv
import re
import math

bob_raw   = "50-BOB.log"     
mbw_raw   = "50-MBW.log"      
baro_raw  = "50-BARO.txt"     
pc_raw    = "50-PC.xlsx"      

BESTAND_UIT = "50_gecombineerd_RH_thermistors.xlsx"


# INLEZEN + OPSCHONEN BOB / MBW

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

# INLEZEN PC-BESTAND MET THERMISTORS


def load_pc_thermistors(path):
    """
    Leest het PC/thermistorbestand (XLSX) in:
      - zoekt een DateTime-kolom
      - maakt thermistorwaarden numeriek
      - berekent gemiddelde en gradiënt voor thermistor 1,2,5,6
    """
    df = pd.read_excel(path, sheet_name=0)
    df.columns = [str(c) for c in df.columns]

    # 1) Probeer datetime-kolom op basis van dtype
    dt_candidates = [c for c in df.columns
                     if np.issubdtype(df[c].dtype, np.datetime64)]

    # 2) Zo niet, dan op basis van naam ('time' of 'timestamp')
    if not dt_candidates:
        dt_candidates = [
            c for c in df.columns
            if re.search("time", c, re.IGNORECASE) or "timestamp" in c.lower()
        ]

    # 3) Zo niet, neem de eerste kolom als tijd
    if not dt_candidates:
        dtcol = df.columns[0]
    else:
        dtcol = dt_candidates[0]

    # Hernoem naar DateTime en naar datetime omzetten
    df = df.rename(columns={dtcol: "DateTime"})
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime")
    df = df[~df["DateTime"].duplicated(keep="first")]

    # Maak overige kolommen numeriek (komma’s → punten, etc.)
    df = numeric_cleanup(df)

    # Thermistor-kolommen: 'thermistor 1 (°C)' t/m 'thermistor 6 (°C)'
    # Gebruik alleen 1,2,5,6 (3 en 4 zijn kapot)
    therm_cols = []
    for col in df.columns:
        m = re.search(r"thermistor\s*([1-6])", col, re.IGNORECASE)
        if m and m.group(1) in ["1", "2", "5", "6"]:
            therm_cols.append(col)

    pc_small = df[["DateTime"] + therm_cols].copy()

    pc_small["Therm_mean_1_2_5_6 (°C)"] = pc_small[therm_cols].mean(axis=1)

    pc_small["Therm_gradient_1_2_5_6 (K)"] = (
        pc_small[therm_cols].max(axis=1) - pc_small[therm_cols].min(axis=1)
    )

    return pc_small

# INLEZEN BOB, MBW


bob = numeric_cleanup(normalize_datetime(load_raw_log(bob_raw)))
mbw = numeric_cleanup(normalize_datetime(load_raw_log(mbw_raw)))

# BOB + MBW samenvoegen op tijd
comb = bob.merge(mbw, on="DateTime", how="inner", suffixes=("_BOB", "_MBW"))
comb = comb.sort_values("DateTime")

# BOB + MBW liepen 1 uur achter -> alle tijden 1 uur naar voren
comb["DateTime"] = comb["DateTime"] + pd.Timedelta(hours=1)

# BAROMETER INLEZEN EN KOPPELEN

baro = pd.read_csv(
    baro_raw,
    header=None,
    names=["DateTime", "P_hPa"],
)

baro["DateTime"] = pd.to_datetime(baro["DateTime"], dayfirst=True)
baro = baro.sort_values("DateTime")
comb = comb.sort_values("DateTime")

comb = pd.merge_asof(
    comb,
    baro,
    on="DateTime",
    direction="nearest"
)

comb["P_lab_Pa"] = comb["P_hPa"] * 100.0

# THERMISTOR-PC AANKOPPELEN (nearest in time)

pc = load_pc_thermistors(pc_raw)

comb = comb.sort_values("DateTime")
pc   = pc.sort_values("DateTime")

comb = pd.merge_asof(
    comb,
    pc,
    on="DateTime",
    direction="nearest"
)

# Keuze voor dew- of frosspoint voor mirror temp

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

# MBW-INSTRUMENTCORRECTIE 

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


# RH BEREKENEN MET THERMISTOR-GEMIDDELDE

T_THERM_COL = "Therm_mean_1_2_5_6 (°C)"

comb["RH_raw (%)"] = comb.apply(
    lambda r: rh_from_frostpoint(
        r["MBW_input_raw (°C)"], r[T_THERM_COL], r["P_lab_Pa"]
    ) if pd.notna(r[T_THERM_COL]) and pd.notna(r["MBW_input_raw (°C)"]) else np.nan,
    axis=1
)

comb["RH_corr (%)"] = comb.apply(
    lambda r: rh_from_frostpoint(
        r["MBW345_dp_corr (°C)"], r[T_THERM_COL], r["P_lab_Pa"]
    ) if pd.notna(r[T_THERM_COL]) and pd.notna(r["MBW345_dp_corr (°C)"]) else np.nan,
    axis=1
)

comb.to_excel(BESTAND_UIT, index=False)

