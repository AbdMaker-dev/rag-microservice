-- Chaque document indexé appartient à un cours, créé côté plateforme avant
-- tout dépôt : le professeur crée le cours (matière, niveau, série, classe),
-- puis y ajoute ses documents un à un. C'est cette référence qui permet à la
-- génération de retrouver exactement les documents d'UN cours, et pas
-- seulement ceux d'un périmètre.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS course_id text NOT NULL DEFAULT '',
    -- Le cycle, dans les appellations du pays : primaire, CEM, lycée au
    -- Sénégal. Le service stocke ce que la plateforme nomme, sans l'interpréter.
    ADD COLUMN IF NOT EXISTS level text NOT NULL DEFAULT '',
    -- La série (L, S, S1…). Vide pour le primaire et le CEM, qui n'en ont pas.
    ADD COLUMN IF NOT EXISTS track text NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS documents_course_idx ON documents (course_id);
