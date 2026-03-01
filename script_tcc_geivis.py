import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuração de reprodutibilidade
np.random.seed(42)

# Período de simulação (6 meses)
start = pd.to_datetime("2025-01-01 06:00:00")
hours = 24 * 180 
timestamps = pd.date_range(start, periods=hours, freq="h")
machines = ["INJ-01", "INJ-02", "INJ-03", "INJ-04", "INJ-05"]
rows = []

def shift_label(hour):
    return "Manhã" if 6 <= hour < 14 else "Tarde" if 14 <= hour < 22 else "Noite"

def month_factor(ts):
    return 1.10 if ts.month == 3 else 1.00

event_state = {m: 0 for m in machines}

# --- Simulação de Dados ---
for ts in timestamps:
    hour = ts.hour
    shift = shift_label(hour)
    mf = month_factor(ts)

    for m in machines:
        prod_base = 1000
        if shift == "Manhã":
            shift_factor, s_noise, p_noise = 1.00, 1.0, 1.0
        elif shift == "Tarde":
            shift_factor, s_noise, p_noise = 0.95, 1.1, 1.05
        else:
            shift_factor, s_noise, p_noise = 0.92, 1.4, 1.15

        m_factor = 1.0 if m in ["INJ-01", "INJ-02"] else 0.96 if m == "INJ-03" else 0.93 if m == "INJ-04" else 0.90
        
        production = max(0, int(np.random.normal(prod_base * shift_factor * m_factor / mf, 75)))
        scrap_rate = np.random.uniform(0.02, 0.06) * s_noise * mf

        # Eventos intermitentes (falhas)
        if event_state[m] > 0:
            in_event = True
            event_state[m] -= 1
        else:
            in_event = False
            p_start = 0.020 if m == "INJ-05" else 0.010
            if np.random.rand() < (p_start * mf):
                event_state[m] = np.random.randint(6, 18)
                in_event = True

        if in_event:
            production = int(production * np.random.uniform(0.70, 0.90))
            scrap_rate *= np.random.uniform(1.8, 2.8)

        scrap = max(0, min(int(production * min(scrap_rate, 0.35)), production))
        good = production - scrap
        power = np.random.normal(22 * m_factor * p_noise * mf, 2.2)
        run_time = np.random.uniform(0.75, 1.0) * (np.random.uniform(0.75, 0.95) if in_event else 1)
        energy = max(0.0, power * run_time)

        rows.append([ts, m, shift, production, good, scrap, round(power, 2), round(run_time, 2), round(energy, 3)])

df = pd.DataFrame(rows, columns=["timestamp","machine_id","shift","production_pieces","good_pieces","scrap_pieces","power_kW","run_time_h","energy_kWh"])

# --- Cálculo de Desperdícios e Eficiências ---
baseline_kwh_per_piece = 0.022
good_safe = df["good_pieces"].replace(0, np.nan)
kwh_per_piece = df["energy_kWh"] / good_safe
scrap_rate_real = df["scrap_pieces"] / df["production_pieces"].replace(0, np.nan)

w_scrap = df["energy_kWh"] * scrap_rate_real.fillna(0) * 0.9
w_eff = ((kwh_per_piece - baseline_kwh_per_piece).clip(lower=0) * df["good_pieces"]).fillna(0)
w_lowprod = np.where((df["production_pieces"]/1000) < 0.75, df["energy_kWh"] * (0.75 - (df["production_pieces"]/1000)) * 0.6, 0)

df["Desperdicio_kWh"] = (w_scrap + w_eff + w_lowprod).fillna(0).clip(lower=0, upper=df["energy_kWh"]).round(3)
df["Ef_Energ"] = (1 - (df["Desperdicio_kWh"] / df["energy_kWh"].replace(0, np.nan))).clip(lower=0, upper=1)
df["Ef_Prod"] = (df["good_pieces"] / df["production_pieces"].replace(0, np.nan)).fillna(0)

# --- Clusterização (K-Means com StandardScaler) ---
features = ["Ef_Energ", "Ef_Prod", "scrap_pieces"]
X = df[features].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# --- Exportação Final ---
df.to_csv("simulacao_fabrica_TCC_FINAL.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
print("Processamento concluído e arquivo gerado.")