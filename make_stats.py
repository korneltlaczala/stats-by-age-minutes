import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

# Tworzymy folder na raporty, żeby nie zaśmiecać głównego katalogu
os.makedirs("reports", exist_ok=True)

# 1. Wczytujemy dane (trzymam się Twojej ścieżki)
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

    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Raport: {title_suffix}', fontsize=20)

    # WYKRES 1: Suma minut na kwartał
    sns.barplot(data=q_stats, x='Q_birth', y='Suma_Minut', ax=axes[0,0], palette='Blues_d')
    axes[0,0].set_title('Minuty rozegrane przez zawodników urodzonych w danym kwartale')
    axes[0,0].set_xlabel('Kwartał urodzenia')
    axes[0,0].set_ylabel('Suma minut')
    
    
    # WYKRES 2: Średnia na zawodnika
    sns.barplot(data=q_stats, x='Q_birth', y='Srednia_na_Zawodnika', ax=axes[0,1], palette='Oranges_d')
    axes[0,1].set_title('Średnia liczba minut na zawodnika')
    axes[0,1].set_xlabel('Kwartał urodzenia')
    axes[0,1].set_ylabel('Średnia minut')

    # WYKRES 3: Gęstość minutowa
    daily_minutes = data_subset.groupby('DayOfYear')['Minutes'].sum().reindex(range(1, 367), fill_value=0)
    # Dla małych grup (pojedyncze roczniki) zmniejszamy sigmę do 10, żeby lepiej widzieć różnice
    smoothed = gaussian_filter1d(daily_minutes.values, sigma=10)
    
    axes[1,0].plot(range(1, 367), smoothed, color='forestgreen', lw=3)
    axes[1,0].fill_between(range(1, 367), smoothed, alpha=0.3, color='green')
    axes[1,0].set_title('Trend minutowy w zależności od dnia urodzenia')
    axes[1,0].set_xlabel('Dzień roku')
    axes[1,0].set_ylabel('')

    # WYKRES 4: Heatmapa (Rocznik vs Kwartał) - dla pojedynczego rocznika pokaże po prostu jeden wiersz
    pivot_table = data_subset.groupby(['Age_group', 'Q_birth']).size().unstack(fill_value=0)
    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', ax=axes[1,1], cbar=False)
    axes[1,1].set_title('Rozkład zawodników wg. kwartału urodzenia')
    axes[1,1].set_xlabel('Kwartał urodzenia')
    if "Cała Akademia" in title_suffix:
        axes[1,1].set_ylabel('Grupa wiekowa')
    else:
        axes[1,1].set_ylabel('')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join("reports", filename)
    plt.savefig(save_path)
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