import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import random
from typing import Tuple

###############################################################################
# 0. CONFIGURATION GLOBALE
###############################################################################

# On paramètre ici les frais (0.1%)
FEE_RATE = 0.001

# Pour éviter les avertissements SettingWithCopyWarning
pd.options.mode.chained_assignment = None  

###############################################################################
# 1. RÉCUPÉRATION DES DONNÉES DE BINANCE AVEC CCXT
###############################################################################

def get_binance_data(symbol: str,
                     timeframe: str,
                     days: int = 30) -> pd.DataFrame:
    """
    Récupère l'historique OHLCV depuis Binance sur 'days' jours et un 'timeframe' donné.
    """
    binance = ccxt.binance()
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_ohlcv = []
    limit = 500
    batch_since = since

    while True:
        ohlcv_batch = binance.fetch_ohlcv(symbol, timeframe, since=batch_since, limit=limit)
        if not ohlcv_batch:
            break
        all_ohlcv += ohlcv_batch
        batch_since = ohlcv_batch[-1][0] + 1
        if len(ohlcv_batch) < limit:
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df

###############################################################################
# 2. FONCTIONS D’INDICATEURS ET DE SIGNAUX
###############################################################################

def add_ema_features(df: pd.DataFrame, open_period: int, close_period: int) -> pd.DataFrame:
    """
    Calcule les EMA sur 'open' et 'close'.
    """
    df = df.copy()
    df['ema_open'] = df['open'].ewm(span=open_period, adjust=False).mean()
    df['ema_close'] = df['close'].ewm(span=close_period, adjust=False).mean()
    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère des signaux (1 / -1) en fonction des EMA 
    (close > open => 1, close < open => -1).
    """
    df = df.copy()
    df['signal'] = 0
    df.loc[df['ema_close'] > df['ema_open'], 'signal'] = 1
    df.loc[df['ema_close'] < df['ema_open'], 'signal'] = -1
    return df

###############################################################################
# 3. BACKTEST DE LA STRATÉGIE
###############################################################################

def backtest_strategy(df: pd.DataFrame,
                      initial_balance: float = 1000.0,
                      fee_rate: float = FEE_RATE) -> pd.DataFrame:
    """
    Backtest sur signaux (1 = achat, -1 = vente).
    On sort complètement avant de racheter, etc.
    """
    df = df.copy()
    balance = initial_balance
    position = 0.0

    portfolio_values = np.zeros(len(df))

    for i in range(len(df)):
        sig = df['signal'].iloc[i]
        price = df['close'].iloc[i]

        if i == 0:
            portfolio_values[i] = balance
            continue

        prev_sig = df['signal'].iloc[i - 1]

        # Passage à 1 => on achète
        if sig == 1 and prev_sig != 1:
            # Si déjà position, on la vend d'abord
            if position > 0:
                balance = position * price * (1 - fee_rate)
                position = 0.0
            # Achat
            position = (balance * (1 - fee_rate)) / price
            balance = 0.0

        # Passage à -1 => on vend
        elif sig == -1 and prev_sig != -1:
            if position > 0:
                balance = position * price * (1 - fee_rate)
                position = 0.0

        # Valeur du portefeuille
        if position > 0:
            portfolio_values[i] = position * price
        else:
            portfolio_values[i] = balance

    df['portfolio_value'] = portfolio_values
    return df

###############################################################################
# 4. MESURE DE PERFORMANCE
###############################################################################

def compute_sharpe_ratio(df: pd.DataFrame, annual_factor: float) -> float:
    """
    Calcule le Sharpe Ratio annualisé (annual_factor ~ nombre de points par an).
    """
    df = df.copy()
    df['returns'] = df['portfolio_value'].pct_change().fillna(0)
    mean_ret = df['returns'].mean() * annual_factor
    std_ret = df['returns'].std() * np.sqrt(annual_factor)
    return 0 if std_ret == 0 else mean_ret / std_ret

def compute_sortino_ratio(df: pd.DataFrame, annual_factor: float) -> float:
    """
    Calcule le Sortino Ratio annualisé.
    """
    df = df.copy()
    df['returns'] = df['portfolio_value'].pct_change().fillna(0)
    mean_ret = df['returns'].mean() * annual_factor
    negative_returns = df['returns'][df['returns'] < 0]
    dd = negative_returns.std() * np.sqrt(annual_factor)
    return 0 if dd == 0 else mean_ret / dd

def compute_max_drawdown(df: pd.DataFrame) -> float:
    """
    Calcule le Max Drawdown (en %) de la colonne portfolio_value.
    """
    roll_max = df['portfolio_value'].cummax()
    drawdown = (df['portfolio_value'] - roll_max) / roll_max
    return drawdown.min()  # valeur négative

def compute_calmar_ratio(df: pd.DataFrame, annual_factor: float) -> float:
    """
    Calcule le Calmar Ratio: (Annualized Return) / (Max Drawdown absolu).
    """
    df = df.copy()
    df['returns'] = df['portfolio_value'].pct_change().fillna(0)
    total_ret = (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0]) - 1
    nb_periods = len(df)
    cagr = (1 + total_ret) ** (annual_factor / nb_periods) - 1 if nb_periods>0 else 0
    max_dd = compute_max_drawdown(df)
    if max_dd == 0:
        return 0
    return cagr / abs(max_dd)

def buy_and_hold(df: pd.DataFrame, initial_balance: float = 1000.0) -> pd.DataFrame:
    """
    Calcule la valeur du portefeuille en Buy & Hold sur la même période.
    """
    df = df.copy()
    first_price = df['close'].iloc[0]
    quantity = initial_balance / first_price
    df['buy_hold_value'] = df['close'] * quantity
    return df

###############################################################################
# 5. ALGORITHME GÉNÉTIQUE
###############################################################################

def genetic_optimize(df: pd.DataFrame,
                     population_size: int = 20,
                     generations: int = 10,
                     ema_min: int = 5,
                     ema_max: int = 50,
                     annual_factor: float = 35040) -> Tuple[int, int]:
    """
    Algorithme génétique pour trouver (open_ma, close_ma) maxant le Sharpe Ratio
    sur 'df'.
    """
    # Génération aléatoire
    population = []
    for _ in range(population_size):
        open_ma = random.randint(ema_min, ema_max)
        close_ma = random.randint(ema_min, ema_max)
        population.append((open_ma, close_ma))

    def fitness(open_ma: int, close_ma: int) -> float:
        temp = df.copy()
        temp = add_ema_features(temp, open_ma, close_ma)
        temp = generate_signals(temp)
        temp = backtest_strategy(temp, fee_rate=FEE_RATE)
        return compute_sharpe_ratio(temp, annual_factor=annual_factor)

    for gen in range(generations):
        scores = []
        for indiv in population:
            s = fitness(indiv[0], indiv[1])
            scores.append((s, indiv))
        scores.sort(key=lambda x: x[0], reverse=True)
        survivors = scores[:population_size // 2]

        new_population = []
        # On garde l'élite
        for s in survivors:
            new_population.append(s[1])

        # Reproduction
        while len(new_population) < population_size:
            parent1 = random.choice(survivors)[1]
            parent2 = random.choice(survivors)[1]
            child_open = parent1[0] if random.random() < 0.5 else parent2[0]
            child_close = parent1[1] if random.random() < 0.5 else parent2[1]
            # Mutation
            if random.random() < 0.1:
                child_open = random.randint(ema_min, ema_max)
            if random.random() < 0.1:
                child_close = random.randint(ema_min, ema_max)
            new_population.append((child_open, child_close))

        population = new_population
        best_score, best_indiv = scores[0]
        print(f"Génération {gen+1}/{generations} | Meilleur score: {best_score:.4f} | "
              f"Individu: {best_indiv}")

    # Score final
    final_scores = [(fitness(ind[0], ind[1]), ind) for ind in population]
    final_scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_indiv = final_scores[0]
    print(f"Meilleur individu final: {best_indiv} avec un score de {best_score:.4f}")
    return best_indiv

###############################################################################
# 6. K-FOLD VALIDATION (TIME SERIES SPLIT) 
###############################################################################

def time_series_kfold_split(df: pd.DataFrame, k: int = 6):
    """
    Découpe le DataFrame en k segments (folds) successifs.
    On renvoie des tuples (train_df, test_df) pour chaque fold.
    """
    df = df.copy()
    n = len(df)
    fold_size = n // (k+1) if (k+1) > 0 else n

    segments = []
    for i in range(k+1):
        seg_start = i * fold_size
        seg_end = (i+1)* fold_size if (i+1)*fold_size < n else n
        segment_df = df.iloc[seg_start:seg_end]
        segments.append(segment_df)

    folds = []
    # On a k folds => index 1..k
    for i in range(1, k+1):
        train = pd.concat(segments[:i])
        test  = segments[i]
        folds.append((train, test))
    return folds

###############################################################################
# 7. META-BACKTEST (EXÉCUTION) + GRAPHIQUES (CLOSE + EMAs + BUY/SELL + PORTFOLIO)
###############################################################################

if __name__ == "__main__":
    symbol = "BTC/USDT"
    timeframes = ["1h", "30m", "15m"]

    # =========================================
    # Choix de la période personnalisée
    # =========================================
    days_to_fetch = 365  
    start_date_str = "2025-01-01"
    end_date_str   = "2025-02-26"

    start_date = pd.to_datetime(start_date_str)
    end_date   = pd.to_datetime(end_date_str)

    initial_balance = 1000.0
    k_folds = 6

    for tf in timeframes:
        print("\n" + "="*60)
        print(f"=== TIMEFRAME: {tf} ===")
        print("="*60)

        # 1) Récupération des données sur X jours (365)
        df_full = get_binance_data(symbol=symbol, timeframe=tf, days=days_to_fetch)
        if len(df_full) < 50:
            print(f"Pas assez de données pour {tf}")
            continue
        
        # 2) Filtrage sur la plage souhaitée
        df_raw = df_full.loc[(df_full.index >= start_date) & (df_full.index <= end_date)]
        if df_raw.empty:
            print(f"Aucune donnée dans la plage {start_date_str} -> {end_date_str} pour {tf}.")
            continue

        # 3) K-Fold Validation
        folds = time_series_kfold_split(df_raw, k=k_folds)
        if not folds:
            print(f"Impossible de créer des folds pour {tf}")
            continue

        all_fold_results = []

        # Facteur d'annualisation 
        if tf == "1h":
            annual_factor = 8760
        elif tf == "30m":
            annual_factor = 17520
        elif tf == "15m":
            annual_factor = 35040
        else:
            annual_factor = 35040  # fallback

        fold_index = 0
        for train_df, test_df in folds:
            fold_index += 1
            print(f"\n--- Fold {fold_index}/{k_folds} ---")

            # a) OPTIMISATION sur la partie train
            best_open_ma, best_close_ma = genetic_optimize(
                train_df,
                population_size=10,
                generations=3,
                ema_min=5,
                ema_max=50,
                annual_factor=annual_factor
            )

            # b) APPLICATION sur la partie test
            test_copy = test_df.copy()
            test_copy = add_ema_features(test_copy, best_open_ma, best_close_ma)
            test_copy = generate_signals(test_copy)
            test_copy = backtest_strategy(test_copy, fee_rate=FEE_RATE)

            # c) Performance
            sharpe = compute_sharpe_ratio(test_copy, annual_factor=annual_factor)
            sortino = compute_sortino_ratio(test_copy, annual_factor=annual_factor)
            max_dd = compute_max_drawdown(test_copy)
            calmar = compute_calmar_ratio(test_copy, annual_factor=annual_factor)

            # Valeur finale & Return
            final_val = test_copy['portfolio_value'].iloc[-1]
            ret_pct = (final_val / test_copy['portfolio_value'].iloc[0] - 1) * 100

            # d) Buy & Hold
            test_copy = buy_and_hold(test_copy, initial_balance=test_copy['portfolio_value'].iloc[0])
            bh_final = test_copy['buy_hold_value'].iloc[-1]
            bh_ret_pct = (bh_final / test_copy['buy_hold_value'].iloc[0] - 1) * 100

            # e) Sauvegarde des résultats
            results = {
                'fold': fold_index,
                'best_open_ma': best_open_ma,
                'best_close_ma': best_close_ma,
                'sharpe': sharpe,
                'sortino': sortino,
                'max_drawdown': max_dd,
                'calmar': calmar,
                'final_value': final_val,
                'strategy_return_%': ret_pct,
                'buy_hold_return_%': bh_ret_pct
            }
            all_fold_results.append(results)

            # f) Affichage console
            print(f"Résultats Fold {fold_index} (test set) :")
            print(f"  EMA (open={best_open_ma}, close={best_close_ma})")
            print(f"  Valeur Finale: {final_val:.2f} USDT")
            print(f"  Return: {ret_pct:.2f}% vs. Buy&Hold: {bh_ret_pct:.2f}%")
            print(f"  Sharpe: {sharpe:.4f}, Sortino: {sortino:.4f}, Calmar: {calmar:.4f}, "
                  f"MaxDD: {max_dd:.2%}")

        # 4) Analyse globale sur k folds
        if all_fold_results:
            df_res = pd.DataFrame(all_fold_results)
            avg_sharpe = df_res['sharpe'].mean()
            avg_sortino = df_res['sortino'].mean()
            avg_calmar = df_res['calmar'].mean()
            avg_mdd = df_res['max_drawdown'].mean()
            avg_ret = df_res['strategy_return_%'].mean()
            avg_bh = df_res['buy_hold_return_%'].mean()

            print("\n=== META-BACKTEST (agrégation des folds) ===")
            print(f"Timeframe = {tf}")
            print(f"Nombre de folds = {k_folds}")
            print(f"Moyenne Sharpe: {avg_sharpe:.4f}")
            print(f"Moyenne Sortino: {avg_sortino:.4f}")
            print(f"Moyenne Calmar: {avg_calmar:.4f}")
            print(f"Moyenne Max Drawdown: {avg_mdd:.2%}")
            print(f"Moyenne Return: {avg_ret:.2f}% vs. Buy&Hold: {avg_bh:.2f}%")

            # === PLOT FINAL ===
            best_index = df_res['sharpe'].idxmax()
            best_open = df_res.loc[best_index, 'best_open_ma']
            best_close = df_res.loc[best_index, 'best_close_ma']

            # On applique ces params "gagnants" sur tout df_raw
            df_best = df_raw.copy()
            df_best = add_ema_features(df_best, best_open, best_close)
            df_best = generate_signals(df_best)
            df_best = backtest_strategy(df_best, fee_rate=FEE_RATE,
                                        initial_balance=initial_balance)
            df_best = buy_and_hold(df_best, initial_balance=initial_balance)

            # === IDENTIFIER LES POINTS BUY/SELL ===
            buy_signals = df_best[
                (df_best['signal'] == 1) &
                (df_best['signal'].shift(1) != 1)
            ]
            sell_signals = df_best[
                (df_best['signal'] == -1) &
                (df_best['signal'].shift(1) != -1)
            ]

            # === TRACE SUR 2 SUBPLOTS ===
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # -- SUBPLOT 1: Prix + EMAs + Buy/Sell signals --
            ax1.plot(df_best.index, df_best['close'], label='Close Price', color='black')
            ax1.plot(df_best.index, df_best['ema_open'], label=f'EMA (open={best_open})', color='orange')
            ax1.plot(df_best.index, df_best['ema_close'], label=f'EMA (close={best_close})', color='blue')

            # On place les signaux Buy/Sell au prix 'close'
            ax1.scatter(buy_signals.index, buy_signals['close'], 
                        marker='^', color='green', s=100, label='Buy Signal')
            ax1.scatter(sell_signals.index, sell_signals['close'], 
                        marker='v', color='red', s=100, label='Sell Signal')

            ax1.set_ylabel('Price (USDT)')
            ax1.legend(loc='best')
            ax1.grid(True)

            # -- SUBPLOT 2: Valeur du portefeuille + Buy & Hold --
            ax2.plot(df_best.index, df_best['portfolio_value'], 
                     label='Stratégie Optimale', color='blue')
            ax2.plot(df_best.index, df_best['buy_hold_value'], 
                     label='Buy & Hold', color='orange', alpha=0.7)
            ax2.set_ylabel('Portfolio Value (USDT)')
            ax2.set_xlabel('Date')
            ax2.legend(loc='best')
            ax2.grid(True)

            plt.suptitle(
                f"Best Strategy vs Buy & Hold - {tf}\n"
                f"EMA(open={best_open}, close={best_close}) - "
                f"Période: {start_date_str} -> {end_date_str}"
            )
            plt.tight_layout()
            plt.show()

        else:
            print("Aucun fold n'a pu être traité.")

    print("\n=== Fin du script ===")
