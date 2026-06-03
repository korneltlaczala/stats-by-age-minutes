import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patheffects as path_effects
from matplotlib.ticker import FixedLocator

# Folder na raporty
os.makedirs("reports_physical", exist_ok=True)

# Oficjalne barwy (zgodne z Twoimi preferencjami)
KSP_RED = "#e30613"
KSP_BLACK = "#1A171B" 

# Liczba odcinków na poziomach Beep Testu (na podstawie image_93dc8a.png)
BEEP_SHUTTLES = {
    1: 7, 2: 8, 3: 8, 4: 9, 5: 9, 6: 10, 7: 11, 8: 11, 9: 11, 10: 11,
    11: 12, 12: 12, 13: 13, 14: 13, 15: 13, 16: 14, 17: 14, 18: 15, 19: 15, 20: 16, 21: 16
}

def beep_to_continuous(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if not s or s == 'nan': return np.nan
    try:
        if '.' in s:
            parts = s.split('.')
            level = int(parts[0])
            shuttle = int(parts[1])
        else:
            level = int(float(s))
            shuttle = 0
        max_shuttles = BEEP_SHUTTLES.get(level, 12)
        shuttle = min(shuttle, max_shuttles)
        return level + (shuttle / (max_shuttles + 1))
    except: return np.nan

def continuous_to_beep(val):
    if pd.isna(val) or val <= 0: return ""
    level = int(val)
    fraction = val - level
    max_shuttles = BEEP_SHUTTLES.get(level, 12)
    shuttle = int(round(fraction * (max_shuttles + 1)))
    if shuttle >= max_shuttles + 1:
        level += 1
        shuttle = 0
    return f"{level}.{shuttle}"

# 1. Wczytanie danych z poprawnej ścieżki
df = pd.read_csv('processed/combined_data.csv', dtype={'Beep': str})
df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
df['DayOfYear'] = df['DOB'].dt.dayofyear
df['5m'] = pd.to_numeric(df['5m'], errors='coerce')
df['30m'] = pd.to_numeric(df['30m'], errors='coerce')
df['Beep_Continuous'] = df['Beep'].apply(beep_to_continuous)

def generate_physical_report(team_data, team_name):
    players = team_data.sort_values(by='DayOfYear').copy().reset_index(drop=True)
    n_players = len(players)
    if players.empty: return

    # --- ALGORYTM OSI CZASU Z MINIMALNYM ODSTĘPEM 0.1 ---
    days_diffs = []
    for i in range(n_players):
        if i == 0:
            days_diffs.append(players['DayOfYear'].iloc[0])
        else:
            days_diffs.append(players['DayOfYear'].iloc[i] - players['DayOfYear'].iloc[i-1])
            
    total_days_diff = sum(days_diffs) if sum(days_diffs) > 0 else 1
    
    bar_width = 0.8
    min_gap = 0.1  # Poprawione z 0.2 na 0.1
    total_space_available = n_players * 0.8 
    
    x_positions = []
    current_pos = 0.0
    
    for diff in days_diffs:
        time_gap = (diff / total_days_diff) * total_space_available
        current_pos += (time_gap + min_gap)
        x_positions.append(current_pos)
        current_pos += bar_width
        
    players['x_pos'] = x_positions

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(f"AKADEMIA POLONII WARSZAWA | RAPORT MOTORYCZNY: {team_name}", 
                 fontsize=22, fontweight='bold', color=KSP_BLACK, y=0.98)
    
    gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.22)
    
    tests = [
        ('5m', 'TEST 5m [s] (mniej = lepiej)', 'lower'),
        ('30m', 'TEST 30m [s] (mniej = lepiej)', 'lower'),
        ('Beep_Continuous', 'BEEP TEST (Level.Shuttle)', 'higher')
    ]
    
    for i, (col, title, direction) in enumerate(tests):
        # --- A. ŚREDNIE KWARTALNE (Ujednolicone kolory) ---
        ax_avg = fig.add_subplot(gs[i, 0])
        q_avg = players.groupby('Q_birth')[col].mean().reindex(['Q1', 'Q2', 'Q3', 'Q4']).fillna(0)
        
        # Kolory identyczne jak na wykresie po prawej
        avg_colors = [KSP_BLACK if q in ['Q1', 'Q3'] else KSP_RED for q in ['Q1', 'Q2', 'Q3', 'Q4']]
        avg_edges = [KSP_RED if q in ['Q1', 'Q3'] else KSP_BLACK for q in ['Q1', 'Q2', 'Q3', 'Q4']]
        
        bars_avg = ax_avg.bar(q_avg.index, q_avg.values, color=avg_colors, edgecolor=avg_edges, linewidth=1.2, width=0.5)
        ax_avg.set_title("Średnia na kwartał", fontweight='bold', fontsize=11, color=KSP_BLACK)
        
        for bar in bars_avg:
            h = bar.get_height()
            if h > 0:
                label = f"{h:.2f}" if col != 'Beep_Continuous' else continuous_to_beep(h)
                ax_avg.text(bar.get_x() + bar.get_width()/2, h, label, ha='center', va='bottom', fontweight='bold', fontsize=9)

        # --- B. WYNIKI INDYWIDUALNE ---
        ax_indiv = fig.add_subplot(gs[i, 1:])
        colors = [KSP_BLACK if q in ['Q1', 'Q3'] else KSP_RED for q in players['Q_birth']]
        edge_colors = [KSP_RED if q in ['Q1', 'Q3'] else KSP_BLACK for q in players['Q_birth']]
        
        bars_indiv = ax_indiv.bar(players['x_pos'], players[col].fillna(0), 
                                  color=colors, edgecolor=edge_colors, linewidth=1.0, width=bar_width)
        ax_indiv.set_title(title, fontweight='bold', fontsize=12, loc='left', color=KSP_BLACK)
        
        ax_indiv.xaxis.set_major_locator(FixedLocator(players['x_pos']))
        ax_indiv.set_xticklabels(players['Name'], rotation=30, fontsize=8, ha='right', color=KSP_BLACK)

        # Skalowanie osi Beepa
        if col == 'Beep_Continuous':
            y_vals = players[col].dropna()
            if not y_vals.empty:
                y_min, y_max = max(0, int(y_vals.min()) - 1), int(y_vals.max()) + 2
                ax_avg.set_ylim(y_min, y_max)
                ax_indiv.set_ylim(y_min, y_max)
                ticks = np.arange(y_min, y_max)
                ax_avg.yaxis.set_major_locator(FixedLocator(ticks))
                ax_indiv.yaxis.set_major_locator(FixedLocator(ticks))
                ax_avg.set_yticklabels([f"{int(t)}.0" for t in ticks])
                ax_indiv.set_yticklabels([f"{int(t)}.0" for t in ticks])

        y_bottom = ax_indiv.get_ylim()[0]
        y_range = ax_indiv.get_ylim()[1] - y_bottom
        
        for bar, (_, row) in zip(bars_indiv, players.iterrows()):
            val = row[col]
            if not pd.isna(val) and val > 0:
                label = f"{val:.2f}" if col != 'Beep_Continuous' else continuous_to_beep(val)
                ax_indiv.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label, 
                            ha='center', va='bottom', fontsize=7, fontweight='bold')
                
                # Daty na słupkach
                date_str = row['DOB'].strftime('%d.%m')
                t_pos = y_bottom + (y_range * 0.02)
                txt = ax_indiv.text(bar.get_x() + bar.get_width()/2, t_pos, date_str,
                                    rotation=90, color='white', fontsize=6, fontweight='bold', ha='center', va='bottom')
                txt.set_path_effects([path_effects.withStroke(linewidth=1.2, foreground=KSP_BLACK)])

        # Linie kwartałów
        month_days = [91, 182, 274]
        q_ticks = np.interp(month_days, players['DayOfYear'], players['x_pos'])
        for tick, lbl in zip(q_ticks, ['Q2', 'Q3', 'Q4']):
            ax_indiv.axvline(x=tick, color=KSP_RED, linestyle=':', alpha=0.3, lw=1.2)
            ax_indiv.text(tick, ax_indiv.get_ylim()[1], lbl, rotation=90, color=KSP_BLACK, 
                          fontsize=8, alpha=0.5, fontweight='bold', ha='right', va='top')
            
        ax_indiv.set_xlim(min(x_positions) - bar_width, max(x_positions) + bar_width * 2)

    plt.savefig(os.path.join("reports_physical", f"Physical_Report_{team_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Start generowania
for team in df['Age_group'].unique():
    generate_physical_report(df[df['Age_group'] == team], team)