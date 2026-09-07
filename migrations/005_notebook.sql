-- LA BASE CAHIERS — physiquement séparée de la base des professeurs.
--
-- Décision fondatrice du 01/09/2026, confirmée le 07/09 : deux bases de
-- connaissance qui ne se mélangent JAMAIS.
--
--   documents / chunks                  la base PROF : contenu validé,
--                                       partagé par périmètre, il nourrit
--                                       la génération de tous les cours ;
--   notebook_documents / notebook_chunks la base CAHIERS : ce qu'un élève a
--                                       photographié dans SON cahier.
--
-- Pourquoi des TABLES séparées et pas une colonne `role` de plus : un filtre
-- s'oublie. Il suffit d'une requête écrite trop vite, d'un `role` absent
-- d'une clause WHERE, et le cahier d'un élève — son écriture, ses erreurs,
-- parfois son nom — remonte dans le cours d'un autre ou dans la génération
-- d'un professeur. Avec deux tables, l'oubli ne rend pas le mauvais
-- résultat : il ne compile pas. La séparation est portée par le schéma,
-- pas par la discipline de celui qui écrit la requête.
--
-- `student_account_id` est NOT NULL et présent dans chaque index : un
-- passage de cahier sans propriétaire ne peut pas exister.

CREATE TABLE IF NOT EXISTS notebook_documents (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Identifiant côté plateforme. Réindexer le même remplace ses passages.
    external_id         text NOT NULL,
    -- LE propriétaire. Rien ici n'est partagé, jamais.
    student_account_id  text NOT NULL,
    title               text NOT NULL,

    -- Le périmètre vient du COMPTE de l'élève, jamais d'un champ client.
    country             text NOT NULL,
    subject             text NOT NULL,
    grade               text NOT NULL,
    track               text NOT NULL DEFAULT '',
    curriculum_version  text NOT NULL,
    language            text NOT NULL DEFAULT 'fr',
    -- Le chapitre qu'Awa a choisi dans le planning officiel : c'est lui qui
    -- dit où chercher la preuve quand un passage est mal lu.
    chapter             text NOT NULL DEFAULT '',

    embedding_model     text NOT NULL,
    embedding_dimension integer NOT NULL,

    characters          integer NOT NULL DEFAULT 0,
    chunk_count         integer NOT NULL DEFAULT 0,
    indexed_at          timestamptz NOT NULL DEFAULT now(),
    created_at          timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT notebook_documents_external_id_unique UNIQUE (external_id)
);

-- L'élève d'abord dans l'index : toute lecture part de LUI.
CREATE INDEX IF NOT EXISTS notebook_documents_student_idx
    ON notebook_documents (student_account_id, subject, grade, chapter);

CREATE TABLE IF NOT EXISTS notebook_chunks (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  uuid NOT NULL
                 REFERENCES notebook_documents (id) ON DELETE CASCADE,
    -- Recopié depuis le document : une recherche cloisonnée ne doit pas
    -- dépendre d'une jointure qu'on pourrait oublier d'écrire.
    student_account_id text NOT NULL,
    ordinal      integer NOT NULL,
    locator      text NOT NULL,
    content      text NOT NULL,
    token_count  integer NOT NULL,
    embedding    vector(1024) NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT notebook_chunks_document_ordinal_unique UNIQUE (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS notebook_chunks_student_idx
    ON notebook_chunks (student_account_id);

-- Index de similarité, propre à cette base. Le même modèle d'embedding que
-- la base prof (bge-m3, 1024) : c'est ce qui permet de comparer un passage
-- de cahier à un passage validé pour la réparation par preuve.
CREATE INDEX IF NOT EXISTS notebook_chunks_embedding_idx
    ON notebook_chunks USING hnsw (embedding vector_cosine_ops);
