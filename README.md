# Test technique - Tech / Interface chat avec RAG

## **1. Contexte**

Emilia Parenti dirige un **cabinet d’avocats en droit des affaires**, situé à Paris.

Son équipe traite quotidiennement des documents confidentiels : contrats, litiges, notes internes, jurisprudences, etc. Emilia souhaite mettre en place un **chatbot interne sécurisé** pour faciliter l’accès à l'information juridique tout en garantissant la confidentialité.

Pour cette **preuve de concept (PoC)**, les documents utilisés sont **anonymisés** avec de faux noms, et le modèle de langage devra être **appelé via une API** sécurisée.

---

## **2. Objectif fonctionnel**

Le but du test est de concevoir une **application Streamlit** intégrant un système de **RAG (Retrieval-Augmented Generation)** basé sur des documents juridiques uploadés manuellement. L’objectif est de tester :

- ta capacité à **intégrer un LLM à une interface personnalisée**
- ta rigueur dans le **pré-traitement et vectorisation des documents**
- la qualité de ton **architecture logicielle**

### **2.1 Page 1 – Interface Chatbot**

Cette page permet à un collaborateur de :

- Poser une question à l’IA via une interface de chat
- Recevoir une réponse basée exclusivement sur les documents internes
- Créer une nouvelle conversation (💬 bonus : gestion d’un historique de conversations)

Toutes les réponses doivent être générées à partir des **documents vectorisés** (pas de génération hors corpus).

### **2.2 Page 2 – Gestion des documents**

Cette page permet à l’utilisateur de :

- **Uploader** des documents (`.txt`, `.csv`, `.html`)
- **Supprimer** des documents existants
- Automatiquement :
    - **Nettoyer les fichiers**
    - **Vectoriser** le contenu pour la base RAG

L’ensemble des documents doit être indexé pour que le modèle puisse s’y référer via un moteur vectoriel (type FAISS, Chroma, etc.).

---

## **3. Livrables & Environnement de Test**

### **3.1 Setup minimal**

Avant de commencer :

- Créer un environnement Python dédié
- Installer les dépendances nécessaires (ex : `streamlit`, `langchain`, `openai`, `chromadb`, etc.)
- Utiliser un modèle LLM disponible via API (`OpenAI (clef fournit)`, `Mistral`, `Claude`, etc.)
- Créer un dossier local ou une base vectorielle pour stocker les embeddings

### **3.2 Livrables attendus**

| Élément | Détail attendu |
| --- | --- |
| 💻 Application | Interface Streamlit fonctionnelle avec deux pages |
| 📦 Gestion de fichiers | Upload / delete + vectorisation automatisée |
| 🔗 Intégration LLM | API propre, sécurisé, réponse contrôlée via RAG |
| 🧹 Nettoyage des données | Pipeline de preprocessing simple et efficace |
| 📜 Historique (bonus) | Gestion conversationnelle avec suivi des échanges |
| 📁 README | Instructions claires pour exécuter le projet en local |
| 🔗 GitHub | Repo : https://github.com/AI-Sisters/test_technique |

---

## **4. Évaluation**

| Critère | Éléments attendus | Points |
| --- | --- | --- |
| ⚙️ Fonctionnalité | Upload, RAG, interface chat, vectorisation | 150pt |
| 🧱 Architecture | Structure du projet claire, code modulaire | 100pt |
| 🤖 Intégration IA | API LLM bien utilisée, réponses cohérentes | 75pt |
| 🧼 Données | Pipeline de nettoyage fiable et simple | 50pt |
| 🧪 Robustesse | Gestion des erreurs, logs, stabilité | 50pt |
| 🎯 UX | Interface fluide, logique d’usage claire | 50pt |
| 🎁 Bonus | Historique, logs, sécurité, documentation | +10 à +50pt |
| **Total** |  |  |

> 🧠 Tu peux utiliser tous les outils d’IA à disposition (ChatGPT, Copilot, etc.), mais la rigueur et la qualité de ton code primeront.
> 

---

## **5. Conclusion**

Ce test a pour but de valider :

- Ta capacité à **prototyper un outil complet en autonomie**
- Ton aisance avec les concepts de **RAG, vectorisation, et intégration LLM**
- Ta **rigueur technique** (structure, propreté du code, gestion des erreurs)
- Ton **agilité** : apprendre vite, aller à l’essentiel, mais proprement

---

## 🚀 Démarrage rapide

1) **Environnement**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) **Configurer les secrets**  
Copiez `.env.example` en `.env` et renseignez `OPENAI_API_KEY` (les autres valeurs ont des défauts sûrs).

3) **Lancer l’app**
```bash
OPENAI_API_KEY=... streamlit run main.py
```
Ouvrez l’URL affichée par Streamlit (par défaut http://localhost:8501).

4) **Tests**
```bash
source .venv/bin/activate
pytest
```

## 🗺️ Architecture rapide
- `main.py` : entrée Streamlit (multi-pages).
- `pages/1_Chat.py` : interface chat + historique.
- `pages/2_Documents.py` : upload/suppression + indexing.
- `rag/` : logique RAG (preprocess, chunking, registry SQLite, vector store Chroma, hybrid retrieval, QA pipeline, sécurité).
- `data/` : exemples anonymisés et stockage persistant (`uploads/`, `chroma/`, `registry.sqlite3`, `conversations.sqlite3`).

## 🔧 Variables de configuration
Définissables dans `.env` (voir `.env.example`) :

| Variable | Description | Défaut |
| --- | --- | --- |
| `OPENAI_API_KEY` | Clé API OpenAI (obligatoire pour l’exécution) | – |
| `OPENAI_MODEL` | Modèle de génération | `gpt-4o-mini` |
| `OPENAI_EMBEDDINGS` | Modèle d’embed | `text-embedding-3-small` |
| `TOP_K` | Passages retournés par la fusion | `4` |
| `HYBRID_K` | Candidates récupérés par dense/BM25 avant fusion | `8` |
| `LEXICAL_WEIGHT` | Pondération BM25 dans la fusion | `0.4` |
| `CHUNK_SIZE` | Taille des chunks | `1000` |
| `CHUNK_OVERLAP` | Recouvrement entre chunks | `100` |
| `USE_TIKTOKEN` | Découpage tiktoken si `true` | `true` |
| `DOC_PREVIEW_CHARS` | Taille max de l’aperçu fichier en UI | `400` |
| `MAX_INPUT_LENGTH` | Longueur max question | `4000` |
| `HISTORY_MAX_MESSAGES` | Nb messages max dans le résumé | `12` |
| `HISTORY_MAX_CHARS` | Taille max du résumé | `1200` |
| `REWRITE_MAX_MESSAGES` | Nb messages pour la réécriture | `6` |
| `ANONYMIZED_TELEMETRY` | Telemetry Chroma (désactivée) | `false` |

## 🧭 Usage
- Page **Documents** : uploader `.txt/.csv/.html`, voir/supprimer les documents indexés (chunks, métadonnées et vecteurs en Chroma).
- Page **Chat** : poser des questions, citations auto `[n]` et sources listées ; historique persistant et suppression possible.

## 🧱 Notes techniques
- Index vectoriel : Chroma persistant sous `data/chroma`.
- Registry & conversations : SQLite dans `data/registry.sqlite3` et `data/conversations.sqlite3`.
- Sécurité : sanitization d’entrée basique (taille, caractères non imprimables, motifs d’injection courants), réponses limitées au corpus via RAG.***

Tu es libre dans tes choix techniques tant que tu **justifies ton raisonnement**, que ton code est **complet et maintenable**, et que le prototype **fonctionne avec fluidité**.
