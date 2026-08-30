-- Schéma de l'index. Le service ne stocke que ce qu'il faut pour chercher :
-- des passages, leurs vecteurs, et le périmètre qui les délimite.
-- Il ne stocke ni élèves, ni droits, ni cours publiés.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Identifiant du document côté plateforme. C'est lui qui fait foi :
    -- réindexer le même document remplace ses passages.
    external_id         text NOT NULL,
    title               text NOT NULL,
    source_reference    text,

    -- Périmètre, transmis par la plateforme à chaque appel.
    country             text NOT NULL,
    subject             text NOT NULL,
    grade               text NOT NULL,
    curriculum_version  text NOT NULL,
    language            text NOT NULL DEFAULT 'fr',

    -- Le modèle d'embedding utilisé. Deux modèles produisent des vecteurs de
    -- dimensions différentes : les comparer fait échouer la requête.
    embedding_model     text NOT NULL,
    embedding_dimension integer NOT NULL,

    characters          integer NOT NULL DEFAULT 0,
    chunk_count         integer NOT NULL DEFAULT 0,
    indexed_at          timestamptz NOT NULL DEFAULT now(),
    created_at          timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT documents_external_id_unique UNIQUE (external_id)
);

CREATE INDEX IF NOT EXISTS documents_scope_idx
    ON documents (country, subject, grade, curriculum_version, language);

CREATE TABLE IF NOT EXISTS chunks (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ordinal      integer NOT NULL,
    -- Repère lisible : "Maths > Algèbre" ou "p. 12". Sert aux citations.
    locator      text NOT NULL,
    content      text NOT NULL,
    token_count  integer NOT NULL,
    -- La dimension est celle de bge-m3, notre modèle d'embedding. pgvector
    -- l'exige pour bâtir l'index HNSW — sans elle, la migration échoue en
    -- « column does not have dimensions ». Changer de modèle = nouvelle
    -- migration ; le service vérifie déjà la dimension à chaque indexation.
    embedding    vector(1024) NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT chunks_document_ordinal_unique UNIQUE (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);

-- Index de similarité cosinus. HNSW est plus rapide en lecture qu'IVFFlat et
-- ne demande pas de réentraînement quand le volume grandit.
-- Note : il se crée par dimension. À recréer si le modèle d'embedding change.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
