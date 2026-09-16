"""Banc de Lawal : les mêmes vraies questions d'élèves, avant et après chaque
changement (consigne, modèle local, moteur en ligne, GPU).

Né le 15/09/2026 pour mesurer les corrections de Lawal, complété le
16/09/2026 avec les cas de la note d'évaluation (calcul, coordonnées,
erreur à contester, incompréhension).

À lancer DANS le conteneur rag, contre le service local — aucune
plateforme, aucun crédit élève :

    docker compose cp scripts/banc_lawal.py api:/tmp/banc_lawal.py
    docker compose exec -T api python /tmp/banc_lawal.py <nom> [cle ...]

Il écrit /tmp/banc-<nom>.json et affiche, par question, le temps et les
contrôles automatiques. Ces contrôles ne sont qu'un repère : ils attrapent
une erreur connue (« (1, 3) » pour 1 + i√3), ils ne jugent pas une
explication. Les réponses se LISENT.

Les cours visés sont ceux du serveur d'essai (Terminale S2, maths) : un
autre serveur demande d'autres identifiants.

Un moteur EN LIGNE (16/09/2026) : BANC_PROVIDER=gemini BANC_MODEL=… et la
clé sur l'entrée standard — jamais en argument ni en variable affichée,
pour qu'elle n'apparaisse ni dans `ps` ni dans l'historique du shell :

    <déchiffrement côté management> | docker compose exec -T \
        -e BANC_PROVIDER=gemini -e BANC_MODEL=gemini-3.8-flash \
        api python /tmp/banc_lawal.py gemini
"""

import json
import os
import re
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
TOKEN = os.environ["SERVICE_SHARED_SECRET"]
SCOPE = {"country": "SN", "subject": "maths", "level": "secondaire", "track": "S2",
         "grade": "terminale", "curriculumVersion": "2006"}
SIMI = "8abbc64a-4338-481c-b5fc-d532c5e9d16a"
SUITES = "6d236b60-7f2a-42ee-8145-1141efe5c18a"
CPLX = "058d09e2-db91-4003-be2f-853f5b0d9d4b"

CENTRE_Q = "Qu'est-ce que le centre d'une similitude ?"
CENTRE_R = ("Le centre d'une similitude est un point invariant. Si \\( a \\neq 1 \\), "
            "il est donné par \\[ \\omega = \\frac{b}{1-a} \\]")
GEO_Q = "C'est quoi une suite géométrique ?"
GEO_R = ("Une suite géométrique est une suite où l'on passe d'un terme au suivant en "
         "multipliant toujours par le même nombre q, la raison : \\( u_{n+1} = q \\, u_n \\).")
FAUSSE_R = ("On a \\( 2e^{i\\pi/3} = 2\\left(\\frac{1}{2} + i\\frac{\\sqrt{3}}{2}\\right) = 1 + i\\sqrt{3} \\). "
            "Les coordonnées du nouveau point sont donc (1, 3).")

RACINE_3 = r"\\sqrt\s*\{?\s*3|√\s*3|sqrt\(3\)"
UN_TROIS = r"\(\s*1\s*[,;]\s*3\s*\)"

# (clé, cours, question, historique, contrôles)
# contrôles : {"doit": [regex…], "jamais": [regex…]} — insensibles à la casse.
QUESTIONS = [
    ("rapport", SIMI, "Pourquoi le rapport d'une similitude est-il le module de a ?", [], {}),
    ("reconnaitre-apres-centre", SIMI,
     "Comment reconnaît-on une similitude directe à son écriture complexe ?",
     [{"role": "eleve", "content": CENTRE_Q}, {"role": "lawal", "content": CENTRE_R}],
     {"doit": [r"a\s*\\neq\s*0|a\s*≠\s*0|non nul|\\mathbb\{C\}\^\*"]}),
    ("geometrique", SUITES, GEO_Q, [], {}),
    ("croissante", SUITES, "Comment montrer qu'une suite est croissante ?", [],
     {"doit": [r"u_\{?n\s*\+\s*1\}?\s*-\s*u_\{?n"]}),
    ("i-carre", CPLX, "Pourquoi i² = -1 ?", [], {}),
    ("module-3+4i", CPLX, "Comment calculer le module de 3+4i ?", [], {"doit": [r"\b5\b"]}),
    ("arithmetique-apres-geo", SUITES, "Et une suite arithmétique, c'est quoi la différence ?",
     [{"role": "eleve", "content": GEO_Q}, {"role": "lawal", "content": GEO_R}],
     {"doit": [r"ajout|addition|\+\s*r|\+\s*d"]}),
    ("hors-cours-pythagore", SIMI, "C'est quoi le théorème de Pythagore ?", [],
     {"doit": [r"pas dans ton cours"]}),
    # ── La note d'évaluation du 16/09/2026 ──────────────────────────────
    ("note-1-calcul", CPLX, "Calcule 2e^{iπ/3} sous forme algébrique.", [],
     {"doit": [RACINE_3], "jamais": [UN_TROIS]}),
    ("note-2-coordonnees", CPLX, "Quelles sont les coordonnées du point associé à 1 + i√3 ?", [],
     {"doit": [RACINE_3], "jamais": [UN_TROIS]}),
    ("note-3-detecter", CPLX,
     "Un camarade dit que 1 + i√3 correspond au point (1, 3). Est-ce correct ?", [],
     {"doit": [RACINE_3, r"\bnon\b|pas correct|incorrect|faux|erreur"]}),
    ("note-4-se-corriger", CPLX, "Tu as écrit (1, 3). Vérifie et corrige ta réponse.",
     [{"role": "eleve", "content": "Calcule 2e^{iπ/3} et donne le point associé."},
      {"role": "lawal", "content": FAUSSE_R}],
     {"doit": [RACINE_3, r"erreur|tromp"]}),
    ("note-5-comprends-pas", CPLX,
     "Je ne comprends pas pourquoi e^{iθ} représente une rotation. Explique-le avec z = 1.",
     [{"role": "eleve", "content": "Pourquoi l'argument de a est l'angle ?"},
      {"role": "lawal", "content": "L'argument de a est l'angle de la similitude, car a = r e^{iθ}."}],
     {"doit": [r"\\cos|cos\s*θ|cos\\theta"]}),
]


def call(path, body=None):
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type": "application/json", "X-Service-Token": TOKEN},
        method="POST" if body else "GET",
    )
    return json.loads(urllib.request.urlopen(request, timeout=60).read())


def controles(text, checks):
    text = text or ""
    ratés = [f"manque /{p}/" for p in checks.get("doit", []) if not re.search(p, text, re.I)]
    ratés += [f"contient /{p}/" for p in checks.get("jamais", []) if re.search(p, text, re.I)]
    return ratés


def engine():
    provider = os.environ.get("BANC_PROVIDER", "")
    if not provider or provider == "local":
        return None
    key = sys.stdin.readline().strip()
    if len(key) < 10:
        raise SystemExit("clé absente de l'entrée standard")
    return {"provider": provider, "model": os.environ["BANC_MODEL"], "apiKey": key}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "banc"
    only = set(sys.argv[2:])
    moteur = engine()
    results = []
    for key, course, question, history, checks in QUESTIONS:
        if only and key not in only:
            continue
        started = time.time()
        job = call("/answer", {"requestId": "banc-" + key, "courseId": course,
                               "question": question, "scope": SCOPE,
                               "history": history,
                               **({"engine": moteur} if moteur else {})})["jobId"]
        while True:
            status = call(f"/answer/{job}")
            if status["status"] in ("done", "failed"):
                break
            time.sleep(3)
        text = (status.get("answer") or "") + "\n" + (status.get("check") or "")
        failed = controles(text, checks) if status["status"] == "done" else ["échec"]
        results.append({
            "key": key, "question": question, "seconds": round(time.time() - started),
            "status": status["status"], "answer": status.get("answer"),
            "check": status.get("check"), "warnings": status.get("warnings"),
            "engine": status.get("engine"), "error": status.get("error"),
            "controles": failed,
        })
        mark = "ok " if not failed else "KO "
        print(f"{mark} {key:28} {round(time.time() - started):4} s  {'; '.join(failed)}", flush=True)
    path = f"/tmp/banc-{name}.json"
    with open(path, "w") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=1)
    passed = sum(1 for r in results if not r["controles"])
    print(f"\n{passed}/{len(results)} contrôles automatiques passés — {path}")


if __name__ == "__main__":
    main()
