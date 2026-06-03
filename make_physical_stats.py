import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Tworzymy folder na raporty motoryczne
os.makedirs("reports_physical", exist_ok=True)

# Kolory klubowe
KSP_RED = '#e30613'
KSP_BLACK = '#1A171B'

def convert_beep(val):
    """Konwertuje format Level.Shuttle na punkty do skalowania wykresu."""
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
        return level * 100 + shuttle
    except:
        return np.nan

def format_beep(score):
    """Konwertuje punkty z powrotem na czytelny format Level.Shuttle."""
    if pd.isna(score) or score == 0: return ""
    level = int(score // 100)
    shuttle = int(score % 100)
    return f"{level}.{shuttle}"

# 1. Wczytujemy dane (wymuszamy string dla Beep, żeby nie zgubić zer po kropce)
df = pd.read_csv('processed/combined_data.csv', dtype={'Beep': str})
df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
df['DayOfYear'] = df['DOB'].dt.dayofyear
df['5m'] = pd.to_numeric(df['5m'], errors='coerce')
df['30m'] = pd.to_numeric(df['30m'], errors='coerce')
df['Beep_Score'] = df['Beep'].apply(convert_beep)

def generate_physical_report(team_data, team_name):
    # Sortowanie po dacie urodzenia (dzień roku)
    team_data = team_data.sort_values('DayOfYear').copy()
    
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"RAPORT MOTORYCZNY: {team_name}", fontsize=22, fontweight='bold', color=KSP_BLACK)
    
    # Grid 3x3: 1 kolumna na średnie, 2 kolumny na wykresy indywidualne
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.25)
    
    tests = [
        ('5m', 'TEST 5m [s]', 'lower'),
        ('30m', 'TEST 30m [s]', 'lower'),
        ('Beep_Score', 'BEEP TEST (Level.Shuttle)', 'higher')
    ]
    
    for i, (col, title, direction) in enumerate(tests):
        # --- 1. WYKRES ŚREDNIEJ KWARTALNEJ ---
        ax_avg = fig.add_subplot(gs[i, 0])
        q_avg = team_data.groupby('Q_birth')[col].mean().reindex(['Q1', 'Q2', 'Q3', 'Q4']).fillna(0)
        
        bars_avg = ax_avg.bar(q_avg.index, q_avg.values, color=[KSP_BLACK, KSP_RED, KSP_BLACK, KSP_RED], alpha=0.8)
        ax_avg.set_title(f"Średnia na kwartał", fontweight='bold', fontsize=10)
        
        # Etykiety nad słupkami średniej
        for bar in bars_avg:
            h = bar.get_height()
            if h > 0:
                label = f"{h:.2f}" if col != 'Beep_Score' else format_beep(h)
                ax_avg.text(bar.get_x() + bar.get_width()/2, h, label, ha='center', va='bottom', fontweight='bold', fontsize=9)

        # --- 2. WYKRES INDYWIDUALNY ---
        ax_indiv = fig.add_subplot(gs[i, 1:])
        colors = [KSP_BLACK if q in ['Q1', 'Q3'] else KSP_RED for q in team_data['Q_birth']]
        
        # Wyświetlamy wyniki (NaN zamienione na 0 dla pozycji na osi, ale nie etykietujemy ich)
        bars_indiv = ax_indiv.bar(team_data['Name'], team_data[col].fillna(0), color=colors, alpha=0.7)
        ax_indiv.set_title(f"{title} - Wyniki indywidualne", fontweight='bold', fontsize=12)
        ax_indiv.set_xticklabels(team_data['Name'], rotation=90, fontsize=8)
        
        # Etykiety nad słupkami indywidualnymi
        for bar, val in zip(bars_indiv, team_data[col]):
            if not pd.isna(val) and val > 0:
                label = f"{val:.2f}" if col != 'Beep_Score' else format_beep(val)
                ax_indiv.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label, 
                            ha='center', va='bottom', fontsize=7)

        # Skalowanie osi Y dla Beep Testu, aby pokazywała format L.S
        if col == 'Beep_Score':
            y_vals = team_data[col].dropna()
            if not y_vals.empty:
                y_min, y_max = y_vals.min() * 0.95, y_vals.max() * 1.05
                ax_avg.set_ylim(y_min, y_max)
                ax_indiv.set_ylim(y_min, y_max)
                
                # Customowe ticki na osi Y
                ticks = np.linspace(y_min, y_max, 5)
                ax_avg.set_yticklabels([format_beep(t) for t in ax_avg.get_yticks()])
                ax_indiv.set_yticklabels([format_beep(t) for t in ax_indiv.get_yticks()])

        # Pionowe linie oddzielające kwartały (na podstawie DayOfYear)
        y_max_plot = ax_indiv.get_ylim()[1]
        for q_day, q_label in [(91, 'Q2'), (182, 'Q3'), (274, 'Q4')]:
            q_players = team_data[team_data['DayOfYear'] >= q_day]
            if not q_players.empty:
                first_player_name = q_players['Name'].iloc[0]
                idx = list(team_data['Name']).index(first_player_name)
                ax_indiv.axvline(x=idx - 0.5, color=KSP_RED, linestyle=':', alpha=0.4, lw=1.5)
                ax_indiv.text(idx - 0.5, y_max_plot, q_label, rotation=90, color=KSP_BLACK, 
                             fontsize=8, alpha=0.6, ha='right', va='top', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path = os.path.join("reports_physical", f"Physical_Report_{team_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Wygenerowano: {save_path}")

# Generowanie raportów dla wszystkich grup
for team in df['Age_group'].unique():
    generate_physical_report(df[df['Age_group'] == team], team)