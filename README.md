# RAG Chat PoC — Cabinet Emilia Parenti

Chat et gestion documentaire pour un cabinet d’avocats : upload de fichiers (.txt/.csv/.html), nettoyage + chunking, indexation Chroma, et chat RAG avec citations obligatoires. Interface Streamlit en deux pages (Chat, Documents) avec historique persistant et bouton de réinitialisation d’index. Config via `.env`, data persistées sous `data/` (uploads, vecteurs, registres SQLite). Tests Pytest fournis pour chunking, registry, hybrid retrieval, QA citations, et pipeline sécurité.

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

