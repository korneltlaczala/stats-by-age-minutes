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
    ax_srednia.set_title('Średnia liczba minut na zawodnika urodzonego w danym kwartale', fontsize=13, fontweight='bold', color=KSP_BLACK)
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
    ax_trend.set_xlim(-5, 371)
    
    # Dodanie linii kwartalnych (identycznie jak na dolnym wykresie)
    month_days = [1, 91, 182, 274, 366] # Styczeń, Kwiecień, Lipiec, Październik, koniec roku
    month_names = ['Styczeń', 'Kwiecień', 'Lipiec', 'Październik', 'Koniec roku']
    y_max_trend = ax_trend.get_ylim()[1]
    
    for day, name in zip(month_days, month_names):
        ax_trend.axvline(x=day, color='#e30613', linestyle=':', alpha=0.4, lw=1.5)
        # ax_trend.text(day, y_max_trend, name, rotation=90, color='#1A171B', 
        #               fontsize=8, alpha=0.7, fontweight='bold', ha='right', va='top')

    month_midpoints = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]
    month_labels_x = ['Styczeń', 'Luty', 'Marzec', 'Kwiecień', 'Maj', 'Czerwiec', 
                      'Lipiec', 'Sierpień', 'Wrzesień', 'Październik', 'Listopad', 'Grudzień']
    
    ax_trend.set_xticks(month_midpoints)
    ax_trend.set_xticklabels(month_labels_x, rotation=45, ha='right', va='top', 
                             rotation_mode="anchor", fontsize=9, color=KSP_BLACK)

    ax_trend.set_title('Trend minutowy w zależności od dnia urodzenia', fontsize=13, fontweight='bold', color=KSP_BLACK)
    ax_trend.set_xlabel('Dzień urodzenia', fontweight='bold')
    ax_trend.set_ylabel('Wygładzona intensywność minut', fontweight='bold')


    # --- DOLNY WIERSZ: SPROWADZENIE DO PROPORCJONALNEJ OSI CZASU ---
    BIG_LABELS_SIZE = 8
    SMALL_LABELS_SIZE = 3
    SMALL_LABEL_THRESHOLD = 50  # Próg liczby zawodników, poniżej którego stosujemy większe czcionki
    players = data_subset.sort_values(by='DayOfYear').copy()
    
    if not players.empty:
        n_players = len(players)
        
        # 1. Obliczamy rzeczywiste różnice w dniach między kolejnymi zawodnikami
        # Dla pierwszego zawodnika dystans od początku roku (lub 0)
        days_diffs = [players['DayOfYear'].iloc[0]]
        for i in range(1, n_players):
            diff = players['DayOfYear'].iloc[i] - players['DayOfYear'].iloc[i-1]
            days_diffs.append(diff)
            
        total_days_diff = sum(days_diffs) if sum(days_diffs) > 0 else 1
        
        # 2. Definiujemy proporcje wykresu: słupki mają zająć 50% szerokości
        # Przestrzeń jednego słupka przyjmujemy jako jednostkę 1.0
        bar_width = 1.0
        total_bars_space = n_players * bar_width  # Sumaryczna szerokość wszystkich słupków
        
        # Chcemy, żeby wolna przestrzeń była równa przestrzeni słupków (pół na pół)
        total_space_available = total_bars_space
        
        # 3. Rozdzielamy wolną przestrzeń proporcjonalnie do przerw w dniach
        allocated_spaces = []
        for diff in days_diffs:
            space = (diff / total_days_diff) * total_space_available
            allocated_spaces.append(space)
            
        # 4. Wyliczamy finalne pozycje X na wykresie
        x_positions = []
        current_pos = 0.0
        for space in allocated_spaces:
            current_pos += space
            x_positions.append(current_pos)
            current_pos += bar_width  # Przesuwamy o szerokość słupka dla kolejnego elementu
            
        players['x_pos'] = x_positions

        players["minutes_capped"] = players["Minutes"].clip(lower=5)
        # 5. Rysowanie słupków z odpowiednią, stałą szerokością
        bars = ax_indiv.bar(players['x_pos'], players['minutes_capped'], 
                            color=KSP_BLACK, edgecolor=KSP_RED, width=bar_width)
        
        # Podpisy wartości nad słupkami oraz dat na dole słupków
        import matplotlib.patheffects as path_effects
        import locale
        
        # Ustawiamy polskie nazwy miesięcy dla formatowania dat
        try:
            locale.setlocale(locale.LC_TIME, 'pl_PL.utf8')
        except Exception:
            try:
                locale.setlocale(locale.LC_TIME, 'pl_PL')
            except Exception:
                pass # W razie braku locale, użyje domyślnego formatu systemowego

        for bar, (_, player_row) in zip(bars, players.iterrows()):
            h = player_row['Minutes']
            if h >= 0:
                # Wartość minut nad słupkiem
                text_y_position = max(h, 360)
                minutes_fontsize = BIG_LABELS_SIZE if n_players < SMALL_LABEL_THRESHOLD else SMALL_LABELS_SIZE + 1
                ax_indiv.annotate(f'{int(h)}', 
                                  xy=(bar.get_x() + bar.get_width()/2, text_y_position),
                                  xytext=(0, 3), textcoords="offset points", 
                                  ha='center', fontsize=minutes_fontsize, fontweight='bold')
                
                # Formatowanie daty urodzenia (np. "1 stycznia 2009")
                # Zmiana na format numeryczny (np. 15.01.2007), żeby uniknąć problemów z czcionką w nazwach miesięcy
                date_str_numeric = player_row['DOB'].strftime('%d.%m.%Y')
                
                # Pionowy biały napis z datą na samym dole słupka
                if n_players < SMALL_LABEL_THRESHOLD: 
                    txt = ax_indiv.text(bar.get_x() + bar.get_width()/2, ax_indiv.get_ylim()[1] * 0.02, date_str_numeric,
                                        rotation=90, color='white', fontsize=7, fontweight='bold',
                                        ha='center', va='bottom')
                    
                    # Dodajemy subtelny czarny obrys wokół białych liter, żeby były czytelne przy niskich słupkach
                    txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='#1A171B')])

        # Przywrócenie skośnych nazwisk zawodników na wyliczonych pozycjach X
        ax_indiv.set_xticks(players['x_pos'])
        player_name_size = BIG_LABELS_SIZE if n_players < SMALL_LABEL_THRESHOLD else SMALL_LABELS_SIZE
        ax_indiv.set_xticklabels(players['Name'], rotation=45, ha='right', fontsize=player_name_size, color=KSP_BLACK, va="top")
        if n_players >= SMALL_LABEL_THRESHOLD:
            ax_indiv.tick_params(axis='x', pad=0)
        
        # Ustawienie ładnych granic osi X z małym marginesem
        ax_indiv.set_xlim(min(x_positions) - bar_width, max(x_positions) + bar_width * 2)
        
        # 6. Orientacyjne sekcje kwartałów
        month_days = [1, 91, 182, 274] # Styczeń, Kwiecień, Lipiec, Październik
        month_labels = ['Styczeń (Q1)', 'Kwiecień (Q2)', 'Lipiec (Q3)', 'Październik (Q4)']
        
        quarter_ticks = np.interp(month_days, players['DayOfYear'], players['x_pos'])
        
        # Pobieramy aktualny górny limit osi Y, żeby idealnie dopasować napisy do krawędzi ramki
        y_max = ax_indiv.get_ylim()[1]
        
        for q_tick, q_label in zip(quarter_ticks, month_labels):
            ax_indiv.axvline(x=q_tick, color='#e30613', linestyle=':', alpha=0.4, lw=1.5)
            
            # Napisy przesunięte w dół – punkt zakotwiczenia na samej górnej krawędzi wykresu (y_max),
            # wyrównanie pionowe ustawione na 'top' (prawa krawędź napisu po obrocie licuje się idealnie z linią ramki)
            ax_indiv.text(q_tick, y_max, q_label, 
                          rotation=90, color='#1A171B', fontsize=8, alpha=0.7, 
                          fontweight='bold', ha='right', va='top')

    ax_indiv.set_title('Minuty poszczególnych zawodników w zależności od dnia urodzenia', fontsize=13, fontweight='bold', color=KSP_BLACK)
    ax_indiv.set_xlabel('Zawodnik', fontweight='bold')
    ax_indiv.set_ylabel('Minuty', fontweight='bold')


    # DODAJEMY MARGINESY I ZAPISUJEMY RAPORT

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join("reports", filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Wygenerowano: {save_path}")
    print(f"   - Liczba zawodników w raporcie: {len(data_subset)}")

# --- GENEROWANIE RAPORTÓW ---

unique_groups = sorted(df['Age_group'].unique())

# 1. Raport zbiorczy
generate_full_report(df, "Cała Akademia - Zbiorczy", "akademia_zbiorczy.png")

# 2. Raporty rocznikowe
for group in unique_groups:
    group_data = df[df['Age_group'] == group]
    generate_full_report(group_data, f"Rocznik {group}", f"raport_{group}.png")

print("\n🚀 Wszystkie raporty gotowe, z Twoimi oryginalnymi podpisami!")