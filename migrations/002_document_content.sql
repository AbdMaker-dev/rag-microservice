-- Le texte original entier, tel que le prof l'a relu et validé.
-- Les passages de `chunks` sont découpés et se recouvrent : ils servent à la
-- recherche, pas à l'affichage. Un prof doit pouvoir relire son document.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content text NOT NULL DEFAULT '';
ALTER TABLE documents ALTER COLUMN content DROP DEFAULT;
