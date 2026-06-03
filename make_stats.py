import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

# Tworzymy folder na raporty
os.makedirs("reports", exist_ok=True)

# 1. Wczytujemy dane
df = pd.read_csv('processed/combined_data.csv')

# Konwersja typów
df['Minutes'] = pd.to_numeric(df['Minutes'], errors='coerce').fillna(0)
df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
df['DayOfYear'] = df['DOB'].dt.dayofyear

def generate_full_report(data_subset, title_suffix, filename):
    """Funkcja generująca zestaw wykresów w układzie 2x3 z przywróconymi oryginalnymi podpisami."""
    
    # Przygotowanie statystyk kwartalnych
    q_stats = data_subset.groupby('Q_birth').agg(
        Suma_Minut=('Minutes', 'sum'),
        Liczba_Zawodnikow=('Name', 'count')
    ).reindex(['Q1', 'Q2', 'Q3', 'Q4']).fillna(0).reset_index()

    q_stats['Srednia_na_Zawodnika'] = q_stats['Suma_Minut'] / q_stats['Liczba_Zawodnikow'].replace(0, 1)

    # --- PALETA BARW POLONII ---
    KSP_RED = "#e30613"
    KSP_BLACK = "#1A171B"
    black_palette = ["#1A171B", "#2D292E", "#444045", "#5C585D"]
    red_palette = ["#B3050F", "#E30613", "#FF4D00", "#FF8000"]

    sns.set_theme(style="whitegrid")
    
    # Tworzymy figurę i siatkę 2 wiersze x 3 kolumny
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3)
    fig.suptitle(f'Raport: {title_suffix}', fontsize=22, fontweight='bold', color=KSP_BLACK)

    # Definicja osi na podstawie GridSpec
    ax_suma = fig.add_subplot(gs[0, 0])      # Góra lewo: Suma
    ax_srednia = fig.add_subplot(gs[0, 1])   # Góra środek: Średnia
    ax_heatmap = fig.add_subplot(gs[0, 2])   # Góra prawo: Heatmapa
    ax_trend = fig.add_subplot(gs[1, 0])     # Dół lewo (1/3): Gęstość
    ax_indiv = fig.add_subplot(gs[1, 1:])    # Dół prawo (2/3): Zawodnicy

    # --- GÓRNY WIERSZ ---
    
    # WYKRES 1: Suma minut na kwartał
    sns.barplot(data=q_stats, x='Q_birth', y='Suma_Minut', ax=ax_suma, palette=black_palette, edgecolor=KSP_RED, linewidth=1.5)
    ax_suma.set_title('Minuty rozegrane przez zawodników urodzonych w danym kwartale', fontsize=13, fontweight='bold', color=KSP_BLACK)
    ax_suma.set_xlabel('Kwartał urodzenia', fontweight='bold')
    ax_suma.set_ylabel('Suma minut', fontweight='bold')

    # WYKRES 2: Średnia na zawodnika
    sns.barplot(data=q_stats, x='Q_birth', y='Srednia_na_Zawodnika', ax=ax_srednia, palette=red_palette, edgecolor=KSP_BLACK, linewidth=1)
    ax_srednia.set_title('Średnia liczba minut na zawodnika', fontsize=13, fontweight='bold', color=KSP_BLACK)
    ax_srednia.set_xlabel('Kwartał urodzenia', fontweight='bold')
    ax_srednia.set_ylabel('Średnia minut', fontweight='bold')

    # WYKRES 4: Heatmapa (Rocznik vs Kwartał)
    pivot_table = data_subset.groupby(['Age_group', 'Q_birth']).size().unstack(fill_value=0)
    ksp_cmap = sns.light_palette(KSP_RED, as_cmap=True, input="hex")
    
    sns.heatmap(pivot_table, annot=True, cmap=ksp_cmap, ax=ax_heatmap, cbar=False, 
                fmt='g', vmin=0, linewidths=1.5, linecolor='#2D292E', annot_kws={"weight": "bold", "size": 22, "color": "white"})
    ax_heatmap.set_title('Rozkład zawodników wg. kwartału urodzenia', fontsize=13, fontweight='bold', color=KSP_BLACK)
    ax_heatmap.set_xlabel('Kwartał urodzenia', fontweight='bold')
    if "Cała Akademia" in title_suffix:
        ax_heatmap.set_ylabel('Grupa wiekowa', fontweight='bold')
    else:
        ax_heatmap.set_ylabel('')

    # --- DOLNY WIERSZ ---

    # WYKRES 3: Gęstość minutowa
    daily_minutes = data_subset.groupby('DayOfYear')['Minutes'].sum().reindex(range(1, 367), fill_value=0)
    smoothed = gaussian_filter1d(daily_minutes.values, sigma=10.0)
    ax_trend.plot(range(1, 367), smoothed, color=KSP_RED, lw=3.5)
    ax_trend.fill_between(range(1, 367), smoothed, alpha=0.25, color=KSP_RED)
    ax_trend.set_title('Trend minutowy w zależności od dnia urodzenia', fontsize=13, fontweight='bold', color=KSP_BLACK)
    ax_trend.set_xlabel('Dzień urodzenia', fontweight='bold')
    ax_trend.set_ylabel('Wygładzona intensywność minut', fontweight='bold')

    # NOWY WYKRES: Indywidualne minuty zawodników (z zachowaniem Twoich etykiet osi)
    players_sorted = data_subset.sort_values(by='DayOfYear')
    if not players_sorted.empty:
        x_indices = np.arange(len(players_sorted))
        bars = ax_indiv.bar(x_indices, players_sorted['Minutes'], color=KSP_BLACK, edgecolor=KSP_RED, width=0.7)
        
        # Podpisy na słupkach
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax_indiv.annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), 
                                  textcoords="offset points", ha='center', fontsize=8, fontweight='bold')

        ax_indiv.set_xticks(x_indices)
        ax_indiv.set_xticklabels(players_sorted['Name'], rotation=45, ha='right', fontsize=8, color=KSP_BLACK)
        ax_indiv.set_xlim(-0.5, len(players_sorted)-0.5)

    ax_indiv.set_title('Minuty poszczególnych zawodników (Chronologicznie wg urodzenia)', fontsize=13, fontweight='bold', color=KSP_BLACK)
    ax_indiv.set_xlabel('Zawodnik (Dzień roku urodzenia)', fontweight='bold')
    ax_indiv.set_ylabel('Minuty', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join("reports", filename)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"✅ Wygenerowano: {save_path}")

# --- GENEROWANIE RAPORTÓW ---

unique_groups = sorted(df['Age_group'].unique())

# 1. Raport zbiorczy
generate_full_report(df, "Cała Akademia - Zbiorczy", "akademia_zbiorczy.png")

# 2. Raporty rocznikowe
for group in unique_groups:
    group_data = df[df['Age_group'] == group]
    generate_full_report(group_data, f"Rocznik {group}", f"raport_{group}.png")

print("\n🚀 Wszystkie raporty gotowe, z Twoimi oryginalnymi podpisami!")