# Comyco Federated Deep Learning

Cette version du projet Comyco implémente mot à mot la feuille de route partagée :

1. **Architecture FDL multi-clients** – mise en place d'une simulation avec un
   serveur fédéré et des clients qui effectuent un apprentissage local avant
   d'envoyer leurs poids. Les clients possèdent des jeux de données distincts,
   générés à partir de scénarios réseau différents (maison, campagne, voiture).
2. **Simulation de débit** – génération de traces de bande passante spécifiques
   à chaque client, sans courbes identiques, conformément aux retours
   d'encadrement.
3. **Apprentissage local** – chaque client entraîne un modèle léger de qualité
   d'expérience vidéo sur son sous-ensemble (train/test) en ignorant la métrique
   `wait_ms` jugée trop instable.
4. **Agrégation fédérée** – le serveur applique FedAvg à fréquence configurable
   (chaque round, ou toutes les *n* itérations) et renvoie le modèle global aux
   clients.
5. **Expérimentations** – scripts dédiés pour comparer l'apprentissage
   centralisé et fédéré, pour étudier l'impact du nombre de clients, de la
   fréquence des synchronisations et des scénarios réseau.

## Structure du projet

```
src/
  comyco/
    data/              # Génération des scénarios de débit
    federated/         # Dataset, modèles, clients et serveur fédéré
    experiments/       # Scripts CLI centralisé vs fédéré
    reporting/         # Génération de résumés d'expériences
notebooks/
  legacy_comyco.ipynb  # Notebook d'origine conservé pour référence
```

## Pré-requis

```bash
pip install -r requirements.txt
```

Les dépendances principales sont : `numpy`, `pandas`, `scikit-learn`, `torch` et
`matplotlib` pour l'analyse éventuelle.

## Lancer une simulation fédérée

```bash
python -m comyco.experiments.run_federated --clients 3 --rounds 5 --update-frequency 2
```

Ce script génère un fichier `federated_run.json` contenant un résumé des
agrégations, ainsi que les statistiques par client (débit moyen, interruptions,
matrices de corrélation sans `wait_ms`).

## Lancer le baseline centralisé

```bash
python -m comyco.experiments.run_centralized --clients 3 --epochs 10
```

Un fichier `centralised_run.json` est produit pour comparer la convergence à la
version fédérée.
