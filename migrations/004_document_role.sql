-- Deux natures de documents dans la base de connaissance :
--
--   support-cours       déposé par un professeur, rattaché à SON cours ;
--   programme-officiel  déposé par un administrateur, rattaché au périmètre
--                       seul (matière, niveau, série, classe, version) — il
--                       fait référence pour TOUS les cours de ce périmètre.
--
-- La génération puisera dans les deux : les supports du prof pour le contenu,
-- le programme officiel pour le cadre et les compétences exigibles. La
-- conformité au programme devient structurelle, pas déclarative.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS role text NOT NULL DEFAULT 'support-cours';

CREATE INDEX IF NOT EXISTS documents_role_idx ON documents (role);
