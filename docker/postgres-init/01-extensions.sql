-- Exécuté une seule fois, à la création du volume Postgres.
-- L'image pgvector fournit l'extension mais ne l'active pas.
CREATE EXTENSION IF NOT EXISTS vector;
