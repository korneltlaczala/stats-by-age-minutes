import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

# Tworzymy folder na raporty, żeby nie zaśmiecać głównego katalogu
os.makedirs("reports", exist_ok=True)

# 1. Wczytujemy dane
df = pd.read_csv('processed/combined_data.csv')

# Konwersja typów
df['Minutes'] = pd.to_numeric(df['Minutes'], errors='coerce').fillna(0)
df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
df['DayOfYear'] = df['DOB'].dt.dayofyear

def generate_full_report(data_subset, title_suffix, filename):
    """Funkcja generująca zestaw wykresów dla podanego zestawu danych."""
    
    # Przygotowanie statystyk kwartalnych
    q_stats = data_subset.groupby('Q_birth').agg(
        Suma_Minut=('Minutes', 'sum'),
        Liczba_Zawodnikow=('Name', 'count')
    ).reindex(['Q1', 'Q2', 'Q3', 'Q4']).fillna(0).reset_index()

    q_stats['Srednia_na_Zawodnika'] = q_stats['Suma_Minut'] / q_stats['Liczba_Zawodnikow'].replace(0, 1)

    # --- PALETA BARW POLONII WARSZAWA (KSP STYLE) ---
    KSP_RED = "#e30606"
    KSP_BLACK = "#1A171B"
    
    # Odcienie czerni/szarości dla zróżnicowania słupków w Wykresie 1
    black_palette = ["#1A171B", "#333034", "#4D4A4E", "#666467"]
    # Odcienie czerwieni dla zróżnicowania słupków w Wykresie 2
    red_palette = ["#e30613", "#eb3842", "#f26971", "#f99ba0"]

    # Ustawienie czystego tła z siatką zamiast domyślnego ggplot
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Raport: {title_suffix}', fontsize=22, fontweight='bold', color=KSP_BLACK)

    # WYKRES 1: Suma minut na kwartał (Odcienie czerni z czerwonym obramowaniem)
    sns.barplot(
        data=q_stats, x='Q_birth', y='Suma_Minut', ax=axes[0,0], 
        palette=black_palette, edgecolor=KSP_RED, linewidth=1.5
    )
    axes[0,0].set_title('Minuty rozegrane przez zawodników urodzonych w danym kwartale', fontsize=13, fontweight='bold', color=KSP_BLACK)
    axes[0,0].set_xlabel('Kwartał urodzenia', fontweight='bold')
    axes[0,0].set_ylabel('Suma minut', fontweight='bold')
    
    # WYKRES 2: Średnia na zawodnika (Odcienie czerwieni z czarnym obramowaniem)
    sns.barplot(
        data=q_stats, x='Q_birth', y='Srednia_na_Zawodnika', ax=axes[0,1], 
        palette=red_palette, edgecolor=KSP_BLACK, linewidth=1
    )
    axes[0,1].set_title('Średnia liczba minut na zawodnika', fontsize=13, fontweight='bold', color=KSP_BLACK)
    axes[0,1].set_xlabel('Kwartał urodzenia', fontweight='bold')
    axes[0,1].set_ylabel('Średnia minut', fontweight='bold')

    # WYKRES 3: Gęstość minutowa (Głęboka czerwień i delikatne czarno-czerwone akcenty)
    daily_minutes = data_subset.groupby('DayOfYear')['Minutes'].sum().reindex(range(1, 367), fill_value=0)
    smoothed = gaussian_filter1d(daily_minutes.values, sigma=10.0) # Zgodnie z plikiem stats-by-age-minutes[cite: 1]
    
    axes[1,0].plot(range(1, 367), smoothed, color=KSP_RED, lw=3.5, label='Trend')
    axes[1,0].fill_between(range(1, 367), smoothed, alpha=0.20, color=KSP_RED)
    axes[1,0].set_title('Trend minutowy w zależności od dnia urodzenia', fontsize=13, fontweight='bold', color=KSP_BLACK)
    axes[1,0].set_xlabel('Dzień urodzenia', fontweight='bold')
    axes[1,0].set_ylabel('Wygładzona intensywność minut', fontweight='bold')

    # WYKRES 4: Heatmapa (Przejście od czystej bieli do klubowej czerwieni)
    pivot_table = data_subset.groupby(['Age_group', 'Q_birth']).size().unstack(fill_value=0)
    
    # Tworzymy mapę kolorów zaczynającą się od czystej bieli (#FFFFFF)
    ksp_cmap = sns.light_palette(KSP_RED, as_cmap=True, input="hex")
    
    sns.heatmap(
        pivot_table, 
        annot=True, 
        cmap=ksp_cmap, 
        ax=axes[1,1], 
        cbar=False, 
        fmt='g', 
        vmin=0,                # Wymusza, aby 0 było początkiem skali (czysta biel)
        linewidths=1.5,        # Grubość obwódek kafelków
        linecolor='#2D292E',   # Kolor obwódek (czarnawy antracyt z Twojej palety)
        annot_kws={"weight": "bold", "size": 12}
    )
    axes[1,1].set_title('Rozkład zawodników wg. kwartału urodzenia', fontsize=13, fontweight='bold', color=KSP_BLACK)
    axes[1,1].set_xlabel('Kwartał urodzenia', fontweight='bold')
    if "Cała Akademia" in title_suffix:
        axes[1,1].set_ylabel('Grupa wiekowa', fontweight='bold')
    else:
        axes[1,1].set_ylabel('')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join("reports", filename)
    plt.savefig(save_path, dpi=120) # Delikatnie wyższa jakość wyjściowa plików PNG
    plt.close() # Zamykamy wykres, żeby nie zajmował pamięci w pętli
    print(f"✅ Wygenerowano: {save_path}")

# --- GENEROWANIE RAPORTÓW ---

# 1. Raport zbiorczy dla całej Akademii
generate_full_report(df, "Cała Akademia - Zbiorczy", "akademia_zbiorczy.png")

# 2. Raporty dla każdej grupy wiekowej z osobna
unique_groups = sorted(df['Age_group'].unique())

for group in unique_groups:
    group_data = df[df['Age_group'] == group]
    generate_full_report(group_data, f"Rocznik {group}", f"raport_{group}.png")

print("\n🚀 Wszystkie raporty gotowe w folderze 'reports/'!")