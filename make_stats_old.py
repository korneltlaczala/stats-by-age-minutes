import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter1d

df = pd.read_csv('processed/combined_data.csv')

# Konwersja typów dla pewności
df['Minutes'] = pd.to_numeric(df['Minutes'], errors='coerce').fillna(0)
df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')

# --- PRZYGOTOWANIE STATYSTYK KWARTALNYCH ---
# Agregujemy dane globalnie dla całej akademii
q_stats = df.groupby('Q_birth').agg(
    Suma_Minut=('Minutes', 'sum'),
    Liczba_Zawodnikow=('Name', 'count')
).reset_index()

q_stats['Srednia_na_Zawodnika'] = q_stats['Suma_Minut'] / q_stats['Liczba_Zawodnikow']

# --- WIZUALIZACJA ---
plt.style.use('ggplot') # Ładniejszy styl wykresów
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Analiza Minut w Akademii - Kornel Tłaczała Report', fontsize=20)

# WYKRES 1: Suma minut na kwartał (Słupkowy)
sns.barplot(data=q_stats, x='Q_birth', y='Suma_Minut', ax=axes[0,0], palette='Blues_d')
axes[0,0].set_title('Łączna liczba minut na kwartał')
axes[0,0].set_ylabel('Suma minut')

# WYKRES 2: Średnia minut na głowę (To pokaże "szansę" zawodnika)
sns.barplot(data=q_stats, x='Q_birth', y='Srednia_na_Zawodnika', ax=axes[0,1], palette='Oranges_d')
axes[0,1].set_title('Średnia liczba minut na JEDNEGO zawodnika')
axes[0,1].set_ylabel('Minuty / Zawodnik')

# WYKRES 3: Gęstość minutowa (Trend po dniach roku)
# Wyliczamy dzień roku (1-366)
df['DayOfYear'] = df['DOB'].dt.dayofyear
daily_minutes = df.groupby('DayOfYear')['Minutes'].sum().reindex(range(1, 367), fill_value=0)

# Nakładamy filtr Gaussa dla wygładzenia trendu (sigma=15 dni)
smoothed = gaussian_filter1d(daily_minutes.values, sigma=15)

axes[1,0].plot(range(1, 367), smoothed, color='forestgreen', lw=3)
axes[1,0].fill_between(range(1, 367), smoothed, alpha=0.3, color='green')
axes[1,0].set_title('Gęstość minut w zależności od dnia urodzenia (Trend)')
axes[1,0].set_xlabel('Dzień roku (od 1 stycznia)')
axes[1,0].set_ylabel('Wygładzona intensywność minut')

# WYKRES 4: Heatmapa: Rocznik vs Kwartał (Gdzie gra najwięcej osób?)
pivot_table = df.groupby(['Age_group', 'Q_birth']).size().unstack(fill_value=0)
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', ax=axes[1,1])
axes[1,1].set_title('Liczba zawodników w grupach')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('akademia_raport_wizualny.png')
plt.show()

print("✅ Wykresy zostały wygenerowane i zapisane jako 'akademia_raport_wizualny.png'")