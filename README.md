# Projet de Backtest et Optimisation d'une Stratégie de Trading par Algorithme Génétique

Ce projet propose un pipeline complet pour :
1. **Récupérer** des données de marché à partir de **Binance** (via la librairie [ccxt](https://github.com/ccxt/ccxt)),
2. **Construire** une stratégie basée sur des **moyennes mobiles exponentielles (EMA)**,
3. **Optimiser** les paramètres de la stratégie (via un **algorithme génétique**),
4. **Évaluer** la robustesse des résultats grâce à une **validation croisée** spécifique aux séries temporelles,
5. **Comparer** la performance à un simple **Buy & Hold**,
6. **Visualiser** le tout avec **matplotlib** (prix, signaux d’entrée/sortie, courbe de valeur du portefeuille, etc.).

Ce script s’appelle `levraibon.py` et se veut un exemple de code pédagogique pour la recherche et l’analyse de stratégies de trading. Il couvre les aspects essentiels : backtest, mesures de performance et optimisation.

---

## Table des matières

- [Fonctionnalités principales](#fonctionnalités-principales)
- [Installation et dépendances](#installation-et-dépendances)
- [Utilisation](#utilisation)
- [Paramètres personnalisables](#paramètres-personnalisables)
- [Structure du code](#structure-du-code)
- [Exemple de fonctionnement](#exemple-de-fonctionnement)
- [Avertissement](#avertissement)

---

## Fonctionnalités principales

1. **Récupération des données** :  
   Le script utilise la bibliothèque `ccxt` pour se connecter à Binance et extraire un historique de chandeliers (OHLCV). Il est possible de spécifier l’intervalle de temps (timeframe) et la période souhaitée en jours.

2. **Calcul d'indicateurs** :  
   - **EMA sur le prix d’ouverture** (open) et **EMA sur le prix de clôture** (close).  
   - Un signal d’achat/vente est généré en comparant l’EMA du close à celle du open.

3. **Backtest** :  
   - Simulation de trading en suivant les signaux (achat = 1, vente = -1).  
   - Gestion du capital et des frais (paramétrables) pour chaque transaction.  
   - Comparaison avec la méthode **Buy & Hold** sur la même période.

4. **Mesures de performance** :  
   - **Sharpe Ratio**, **Sortino Ratio**, **Max Drawdown**, **Calmar Ratio**.  
   - Retour total en pourcentage par rapport à l’investissement initial.

5. **Algorithme génétique** :  
   - Recherche des périodes optimales des EMA (de `ema_min` à `ema_max`).  
   - Évaluation de la performance de chaque individu via le Sharpe Ratio.  
   - Sélection, cross-over, mutation et itérations sur plusieurs générations.

6. **Validation croisée (Time Series Split)** :  
   - Découpe des données en `k` segments successifs (folds).  
   - Optimisation sur les premières parties, puis test sur la partie suivante pour évaluer la robustesse.  
   - Agrégation finale des résultats obtenus sur chaque fold.

7. **Visualisation** :  
   - Tracé du prix, des EMAs, des signaux d’achat/vente, et de la valeur du portefeuille par rapport au Buy & Hold.  

---

## Installation et dépendances

Avant de lancer le script, assurez-vous d’installer les dépendances Python suivantes :

```bash
pip install ccxt pandas numpy matplotlib tqdm
```

- **ccxt** : Pour récupérer les données de marché sur Binance.
- **pandas** / **numpy** : Pour la gestion et l’analyse des données.
- **matplotlib** : Pour la visualisation.
- **tqdm** : Pour afficher une barre de progression.

Le script est compatible avec **Python 3.9+**.

---

## Utilisation

1. **Cloner le dépôt** ou télécharger le script `levraibon.py`.
2. **Installer les dépendances** comme mentionné ci-dessus.
3. **Exécuter** le script en ligne de commande :

```bash
python levraibon.py
```

---

## Paramètres personnalisables

- **Frais de transaction (`FEE_RATE`)** : Par défaut `0.001` (soit 0,1 %).
- **Symbol (`symbol`)** : Par défaut `"BTC/USDT"`.
- **Timeframes (`timeframes`)** : `"1h", "30m", "15m"`.
- **days_to_fetch** : Nombre de jours d’historique à récupérer (`365` par défaut).
- **start_date_str** et **end_date_str** : Période d’analyse (`"2025-01-01"` à `"2025-02-26"`).
- **initial_balance** : Capital initial (`1000.0`).
- **k_folds** : Nombre de segments pour la validation croisée (`6`).
- **Algorithme génétique** :
  - `population_size = 20`
  - `generations = 10`
  - `ema_min = 5`, `ema_max = 50`
  - `annual_factor = 35040` (dépend du timeframe)

---

## Structure du code

Le script `levraibon.py` est divisé en sections claires :

1. **Configuration globale**
2. **Récupération des données (ccxt)**
3. **Calcul des indicateurs et signaux**
4. **Backtest**
5. **Mesures de performance**
6. **Algorithme génétique**
7. **Validation croisée (K-fold)**
8. **Exécution et visualisation**

---

## Exemple de fonctionnement

Une fois lancé, le script effectue :

1. **Téléchargement** des chandeliers pour plusieurs timeframes.
2. **Filtrage** sur la période définie.
3. **K-fold Validation** : Optimisation des EMAs sur train, test sur validation.
4. **Calcul des performances** : Sharpe, Sortino, Max Drawdown, etc.
5. **Affichage des graphiques** : Comparaison Stratégie Optimale vs Buy & Hold.

---

## Avertissement ⚠️

Ce projet est fourni à des **fins d’expérimentation et de démonstration**. Il n’est en **aucun cas** une recommandation d’investissement. Les performances passées ne présagent pas des performances futures. **Faites toujours vos propres recherches** et soyez conscient des risques avant de prendre toute décision de trading.

---

