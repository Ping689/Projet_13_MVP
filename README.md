# Projet 13 - MVP RAG Puls-Events

Ce projet est la version **MVP** locale et robuste du chatbot RAG de Puls-Events, conçu pour recommander des événements culturels. Suite au POC (Projet 11) en apportant d'importantes évolutions d'infrastructure, d'interface et d'algorithmes.

---

## Fonctionnalités clés du MVP

1.  **Base Vectorielle Relationnelle (PostgreSQL + `pgvector`)** : Remplacement de l'index FAISS local par une base de données PostgreSQL gérant nativement la similarité cosinus sur des vecteurs à 1024 dimensions (Mistral Embeddings).
2.  **Géolocalisation & Filtrage Spatial** : Calcul de distance sphérique en kilomètres via la formule SQL native de **Haversine** pour restreindre ou trier les événements recommandés selon la position GPS et un rayon paramétrable.
3.  **Interface Utilisateur Moderne (Chainlit)** : Chatbot interactif fluide, affichant les détails de latence du pipeline RAG pour le monitoring et déportant les sources documentaires dans un tiroir latéral propre.
4.  **Persistance de Discussion & Historique UI** : Sauvegarde des messages en base de données SQL. L'interface propose un écran de connexion persistant pour retrouver tout son historique de sessions dans la barre latérale de gauche.
5.  **Recherche Web de Repli (smolagents)** : Intégration d'un agent autonome utilisant Hugging Face `smolagents` pour effectuer des recherches en direct sur le web si les données de la base locale sont absentes ou peu pertinentes.

---

## Installation et préparation de l'environnement

### Étape 1 - Cloner / Préparer le dossier du projet
Assurer d'avoir Docker installé et actif sur notre machine.

### Étape 2 - Créer et activer l'environnement virtuel
```bash
# Créer l'environnement virtuel
python -m venv .venv

# Activer l'environnement (PowerShell)
.\.venv\Scripts\Activate.ps1
```

### Étape 3 - Installer les dépendances
```bash
# Installation des paquets requis
python -m pip install -r requirements.txt
```

### Étape 4 - Configuration des variables d'environnement (`.env`)
Créer ou modifier le fichier `.env` à la racine du projet avec les clés suivantes :
```env
MISTRAL_API_KEY=votre_cle_api_mistral
OPENAGENDA_API_KEY=votre_cle_api_openagenda
OPENAGENDA_AGENDA_UIDS=61665301,37836092,8697104,...

DB_HOST=localhost
DB_PORT=5433
DB_NAME=puls_events_mvp
DB_USER=postgres
DB_PASSWORD=password

CHAINLIT_AUTH_SECRET=puls_events_mvp_secret_key_aleatoire
```

---

## Lancement de l'infrastructure Docker

Lancer le conteneur PostgreSQL contenant l'extension vectorielle en arrière-plan :
```bash
docker-compose up -d
```

---

## Ingestion et vectorisation des données

Pour prétraiter les données OpenAgenda collectées et insérer automatiquement les 1236 chunks avec leurs embeddings dans PostgreSQL :
```bash
.\.venv\Scripts\python scripts/build_faiss_index.py
```

---

## Démarrer le chatbot Chainlit

Lancer l'interface conversationnelle locale :
```bash
.\.venv\Scripts\chainlit run app/chainlit_app.py
```
L'interface web est alors accessible sur :
**[http://localhost:8000](http://localhost:8000)**


## Lancer la suite de tests

Une suite de tests d'intégration valide le bon fonctionnement de la base PostgreSQL locale, du filtrage spatial Haversine, de l'enregistrement de l'historique SQL et des feedbacks utilisateurs :
```bash
.\.venv\Scripts\python -m pytest tests/test_mvp_features.py
```
---

## Administration BDD avec pgAdmin

Le contenu stocké (événements, logs de discussion, feedbacks de l'interface) dans **pgAdmin 4** 
