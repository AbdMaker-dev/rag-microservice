"""Deux files, un seul travail à la fois — et l'élève passe devant.

Mesuré pendant l'E2E du 04/09/2026 : une section prenait 78 s seule et
240 s quand des traitements se chevauchaient. Un seul exécutant rend les
temps prévisibles ; la priorité aux élèves évite qu'une question de 30 s
attende derrière un cours de 4 minutes.
"""

import asyncio

from app.core.jobs import JobStore


def test_un_seul_travail_a_la_fois():
    """Trois demandes simultanées ne se marchent pas dessus."""

    async def scenario():
        store = JobStore()
        ensemble, maximum = [], 0

        def work(nom):
            async def run():
                nonlocal maximum
                ensemble.append(nom)
                maximum = max(maximum, len(ensemble))
                await asyncio.sleep(0.05)
                ensemble.remove(nom)
                return nom
            return run

        jobs = [store.submit(work(f"j{i}")) for i in range(3)]
        for _ in range(100):
            await asyncio.sleep(0.02)
            if all(j.status == "done" for j in jobs):
                break
        assert maximum == 1, "deux travaux se sont exécutés en même temps"
        assert [j.result for j in jobs] == ["j0", "j1", "j2"], "ordre d'arrivée non tenu"

    asyncio.run(scenario())


def test_l_eleve_passe_devant_les_professeurs():
    async def scenario():
        store = JobStore()
        servis = []

        def work(nom):
            async def run():
                servis.append(nom)
                await asyncio.sleep(0.02)
                return nom
            return run

        # Un professeur occupe la machine, deux autres attendent, puis un
        # élève arrive : il doit passer avant les professeurs en attente.
        store.submit(work("prof-1"), lane="prof")
        store.submit(work("prof-2"), lane="prof")
        await asyncio.sleep(0.01)
        store.submit(work("eleve"), lane="eleve")
        store.submit(work("prof-3"), lane="prof")

        for _ in range(100):
            await asyncio.sleep(0.02)
            if len(servis) == 4:
                break
        assert servis[0] == "prof-1", "le premier arrivé garde sa place"
        assert servis[1] == "eleve", "l'élève doit passer devant les professeurs en attente"

    asyncio.run(scenario())


def test_la_place_et_l_attente_sont_annoncees():
    """Le deuxième professeur voit « 2e — environ N secondes », jamais un
    « en cours » muet."""

    async def scenario():
        store = JobStore()

        async def long():
            await asyncio.sleep(0.3)
            return "fini"

        premier = store.submit(lambda: long(), lane="prof")
        second = store.submit(lambda: long(), lane="prof")
        await asyncio.sleep(0.05)

        assert premier.status == "running"
        assert store.position(premier) == 0, "un travail en cours n'attend plus"
        assert store.wait_seconds(premier) is None

        assert second.status == "queued"
        assert store.position(second) == 1, "un seul devant lui : le premier est en cours"
        attente = store.wait_seconds(second)
        assert attente is not None and attente > 0, "une attente doit être annoncée"

    asyncio.run(scenario())


def test_un_eleve_qui_arrive_voit_sa_place_parmi_les_eleves():
    async def scenario():
        store = JobStore()

        async def long():
            await asyncio.sleep(0.3)
            return "fini"

        store.submit(lambda: long(), lane="prof")   # occupe la machine
        store.submit(lambda: long(), lane="prof")   # attend
        await asyncio.sleep(0.02)
        eleve = store.submit(lambda: long(), lane="eleve")

        # L'élève ne compte pas les professeurs devant lui : il les double.
        assert store.position(eleve) == 1

    asyncio.run(scenario())


def test_une_tache_qui_echoue_ne_bloque_pas_la_file():
    async def scenario():
        store = JobStore()

        async def casse():
            raise RuntimeError("le modèle n'a pas répondu")

        async def bonne():
            return "ok"

        rate = store.submit(lambda: casse())
        suivante = store.submit(lambda: bonne())
        for _ in range(100):
            await asyncio.sleep(0.02)
            if suivante.status in ("done", "failed"):
                break
        assert rate.status == "failed"
        assert "n'a pas répondu" in (rate.error or "")
        assert suivante.status == "done", "la file doit continuer après un échec"

    asyncio.run(scenario())


def test_l_attente_annoncee_suit_les_durees_reelles():
    """Le jour où le modèle change, l'estimation suit sans toucher au code."""

    async def scenario():
        store = JobStore()

        async def court():
            await asyncio.sleep(0.05)
            return "ok"

        for _ in range(3):
            job = store.submit(lambda: court(), lane="prof")
            for _ in range(50):
                await asyncio.sleep(0.02)
                if job.status == "done":
                    break
        # Après trois travaux de 0,05 s, l'estimation n'est plus la valeur
        # de départ (150 s) mais la moyenne observée.
        assert store._average("prof") < 1.0

    asyncio.run(scenario())
