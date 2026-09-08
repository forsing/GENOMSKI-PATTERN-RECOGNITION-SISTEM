"""
LOTO 7/39 — GENOMSKI PATTERN RECOGNITION SISTEM

Sistem obrađuje dva potpuno odvojena CSV fajla:

1. Loto;
2. Loto Plus.

Tok obrade:

CSV referentni genom
→ k-mer/minimizer indeks
→ sekvencijalno poravnanje
→ slični istorijski regioni
→ De Bruijn graf nastavaka
→ Hidden Markov režimi
→ Bajesov variant calling
→ sužavanje prostora na relevantne kandidate
→ sastavljanje i ocenjivanje NEXT sedmorke
→ stroga hronološka walk-forward validacija
→ zamrznuta holdout provera

Prvi red svakog CSV fajla je najstariji.
Poslednji red svakog CSV fajla je najnoviji.
"""

from __future__ import annotations

import itertools
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import beta as beta_raspodela

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as greska:
    raise SystemExit(
        "\nNedostaje paket hmmlearn.\n"
        "Instalacija:\n\n"
        "pip install hmmlearn\n"
    ) from greska


# =============================================================================
# PODEŠAVANJA
# =============================================================================

SEED = 39

BROJ_KUGLICA = 39
BROJEVA_U_KOMBINACIJI = 7

OSNOVNA_STOPA = BROJEVA_U_KOMBINACIJI / BROJ_KUGLICA
SLUCAJNO_OCEKIVANJE = BROJEVA_U_KOMBINACIJI**2 / BROJ_KUGLICA

LOTO_CSV = Path(
    "/Users/4c/Desktop/GHQ/data/loto7_4680_k71_loto_2962.csv"
)

LOTO_PLUS_CSV = Path(
    "/Users/4c/Desktop/GHQ/data/loto7_4680_k71_loto_plus_1718.csv"
)

# Dužina vremenskog k-mera izražena brojem uzastopnih izvlačenja.
KMER_DUZINA = 8

# MinHash podešavanja.
BROJ_MINHASH_FUNKCIJA = 32
BROJ_MINHASH_GRUPA = 8
MINHASH_PO_GRUPI = BROJ_MINHASH_FUNKCIJA // BROJ_MINHASH_GRUPA
MINHASH_PROST_BROJ = 2_147_483_647

# Broj kandidata koji se zatim precizno poravnavaju.
MINHASH_KANDIDATI = 240

# Broj najbolje poravnatih istorijskih regiona.
BROJ_SLICNIH_REGIONA = 60

# De Bruijn graf koristi diskretizovane vremenske simbole.
DE_BRUIJN_RED = 3

# Hidden Markov model.
BROJ_HMM_REZIMA = 3
HMM_ITERACIJE = 80
HMM_ISTORIJA = 800

# Bajesovo sužavanje prostora.
BROJ_RELEVANTNIH_BROJEVA = 18
BAJES_PRIOR_SNAGA = 20.0
KAZNA_NEIZVESNOSTI = 0.20

# Validacija.
MINIMUM_ISTORIJE = 350
BROJ_VALIDACIONIH_KORAKA = 120
BROJ_HOLDOUT_KORAKA = 160

# Završno ocenjivanje kombinacija.
TEZINA_PARNOG_SKORA = 0.18

# Bootstrap provera.
BROJ_BOOTSTRAP_SIMULACIJA = 20_000
BOOTSTRAP_BLOK = 10

EPS = 1e-9

NAZIVI_MODELA = (
    "Genomski k-mer nastavci",
    "De Bruijn graf",
    "Hidden Markov režimi",
    "Bajesov variant calling",
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# OPŠTE FUNKCIJE
# =============================================================================

def naslov(tekst: str, znak: str = "=") -> None:
    print()
    print(znak * 78)
    print(tekst)
    print(znak * 78)


def formatiraj_kombinaciju(brojevi) -> str:
    return ", ".join(
        f"{int(broj):02d}"
        for broj in sorted(brojevi)
    )


def standardizuj(vrednosti: np.ndarray) -> np.ndarray:
    vrednosti = np.asarray(vrednosti, dtype=float)

    sredina = float(np.mean(vrednosti))
    odstupanje = float(np.std(vrednosti))

    if not np.isfinite(odstupanje) or odstupanje < EPS:
        return np.zeros_like(vrednosti)

    rezultat = (vrednosti - sredina) / odstupanje

    return np.nan_to_num(
        rezultat,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def skor_u_verovatnoce(skor: np.ndarray) -> np.ndarray:
    """
    Pretvara kontinuirani skor u 39 verovatnoća uključivanja
    čiji je zbir tačno sedam.
    """

    z = standardizuj(skor)

    donja = -30.0
    gornja = 30.0
    nagib = 0.75

    for _ in range(80):
        sredina = (donja + gornja) / 2.0
        zbir = float(
            np.sum(expit(nagib * z + sredina))
        )

        if zbir > BROJEVA_U_KOMBINACIJI:
            gornja = sredina
        else:
            donja = sredina

    pomeraj = (donja + gornja) / 2.0
    verovatnoce = expit(nagib * z + pomeraj)

    return np.clip(
        verovatnoce,
        EPS,
        1.0 - EPS,
    )


def ucitaj_csv(
    putanja: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if not putanja.exists():
        raise FileNotFoundError(
            f"CSV fajl ne postoji: {putanja}"
        )

    okvir = pd.read_csv(
        putanja,
        header=None,
    )

    okvir = okvir.dropna(
        axis=1,
        how="all",
    )

    if okvir.shape[1] != BROJEVA_U_KOMBINACIJI:
        raise ValueError(
            f"{putanja.name}: očekivano je tačno 7 kolona, "
            f"a pronađeno je {okvir.shape[1]}."
        )

    okvir = okvir.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if okvir.isna().any().any():
        redovi = (
            np.where(okvir.isna().any(axis=1))[0] + 1
        )

        raise ValueError(
            f"{putanja.name}: neispravne vrednosti u redovima "
            f"{redovi[:20].tolist()}."
        )

    izvlacenja = okvir.to_numpy(dtype=int)

    for indeks, red in enumerate(
        izvlacenja,
        start=1,
    ):
        if np.any(red < 1) or np.any(red > BROJ_KUGLICA):
            raise ValueError(
                f"{putanja.name}, red {indeks}: "
                "brojevi moraju biti između 1 i 39."
            )

        if len(np.unique(red)) != BROJEVA_U_KOMBINACIJI:
            raise ValueError(
                f"{putanja.name}, red {indeks}: "
                "svih sedam brojeva mora biti različito."
            )

    matrica = np.zeros(
        (len(izvlacenja), BROJ_KUGLICA),
        dtype=np.float64,
    )

    for i, red in enumerate(izvlacenja):
        matrica[i, red - 1] = 1.0

    return izvlacenja, matrica


# =============================================================================
# MINHASH FUNKCIJE
# =============================================================================

_rng_hash = np.random.default_rng(SEED)

MINHASH_A = _rng_hash.integers(
    1,
    MINHASH_PROST_BROJ - 1,
    size=BROJ_MINHASH_FUNKCIJA,
    dtype=np.int64,
)

MINHASH_B = _rng_hash.integers(
    0,
    MINHASH_PROST_BROJ - 1,
    size=BROJ_MINHASH_FUNKCIJA,
    dtype=np.int64,
)


def prozor_u_tokene(prozor: np.ndarray) -> np.ndarray:
    """
    Vremenski prozor pretvara u skup tokena.

    Token sadrži:
    - relativnu poziciju izvlačenja;
    - broj koji je izvučen;
    - prelaz između brojeva iz dva uzastopna izvlačenja.
    """

    tokeni = []

    for pozicija, red in enumerate(prozor):
        brojevi = np.flatnonzero(red > 0.5) + 1

        for broj in brojevi:
            tokeni.append(
                1 + pozicija * 40 + int(broj)
            )

        if pozicija > 0:
            prethodni = (
                np.flatnonzero(
                    prozor[pozicija - 1] > 0.5
                ) + 1
            )

            for prethodni_broj in prethodni:
                for trenutni_broj in brojevi:
                    tokeni.append(
                        10_000
                        + pozicija * 2_000
                        + int(prethodni_broj) * 40
                        + int(trenutni_broj)
                    )

    return np.unique(
        np.asarray(tokeni, dtype=np.int64)
    )


def minhash_potpis(tokeni: np.ndarray) -> np.ndarray:
    if len(tokeni) == 0:
        return np.full(
            BROJ_MINHASH_FUNKCIJA,
            MINHASH_PROST_BROJ,
            dtype=np.int64,
        )

    vrednosti = (
        MINHASH_A[:, None] * tokeni[None, :]
        + MINHASH_B[:, None]
    ) % MINHASH_PROST_BROJ

    return np.min(vrednosti, axis=1)


def minhash_slicnost(
    prvi: np.ndarray,
    drugi: np.ndarray,
) -> float:
    return float(np.mean(prvi == drugi))


def napravi_minhash_indeks(
    potpisi: list[np.ndarray],
) -> list[dict[tuple[int, ...], list[int]]]:
    indeksi = [
        {}
        for _ in range(BROJ_MINHASH_GRUPA)
    ]

    for indeks_prozora, potpis in enumerate(potpisi):
        for grupa in range(BROJ_MINHASH_GRUPA):
            pocetak = grupa * MINHASH_PO_GRUPI
            kraj = pocetak + MINHASH_PO_GRUPI

            kljuc = tuple(
                int(x)
                for x in potpis[pocetak:kraj]
            )

            indeksi[grupa].setdefault(
                kljuc,
                [],
            ).append(indeks_prozora)

    return indeksi


def kandidati_iz_indeksa(
    upit_potpis: np.ndarray,
    indeksi: list[dict],
    broj_prozora: int,
) -> np.ndarray:
    glasovi: dict[int, int] = {}

    for grupa in range(BROJ_MINHASH_GRUPA):
        pocetak = grupa * MINHASH_PO_GRUPI
        kraj = pocetak + MINHASH_PO_GRUPI

        kljuc = tuple(
            int(x)
            for x in upit_potpis[pocetak:kraj]
        )

        for kandidat in indeksi[grupa].get(
            kljuc,
            [],
        ):
            glasovi[kandidat] = (
                glasovi.get(kandidat, 0) + 1
            )

    if glasovi:
        sortirani = sorted(
            glasovi,
            key=lambda i: (
                -glasovi[i],
                -i,
            ),
        )

        return np.asarray(
            sortirani[:MINHASH_KANDIDATI],
            dtype=int,
        )

    # Ako nijedna grupa nema potpuno poklapanje, koristi se
    # vremenski raspoređen rezervni skup.
    broj = min(
        broj_prozora,
        MINHASH_KANDIDATI,
    )

    return np.linspace(
        0,
        broj_prozora - 1,
        broj,
        dtype=int,
    )


# =============================================================================
# SEKVENCIJALNO PORAVNANJE
# =============================================================================

def jaccard_redova(
    prvi: np.ndarray,
    drugi: np.ndarray,
) -> float:
    presek = float(
        np.sum((prvi > 0.5) & (drugi > 0.5))
    )

    unija = float(
        np.sum((prvi > 0.5) | (drugi > 0.5))
    )

    if unija < EPS:
        return 0.0

    return presek / unija


def sekvencijalno_poravnanje(
    upit: np.ndarray,
    kandidat: np.ndarray,
) -> float:
    """
    Gapless lokalno poravnanje vremenskih k-merova.

    Novija izvlačenja unutar k-mera imaju veću težinu.
    """

    duzina = len(upit)

    tezine = np.linspace(
        0.50,
        1.50,
        duzina,
    )

    tezine /= tezine.sum()

    lokalni = np.asarray(
        [
            jaccard_redova(upit[i], kandidat[i])
            for i in range(duzina)
        ],
        dtype=float,
    )

    osnovni_skor = float(
        np.dot(tezine, lokalni)
    )

    # Dodatna sličnost promena između uzastopnih izvlačenja.
    if duzina > 1:
        upit_promene = np.abs(
            np.diff(upit, axis=0)
        )

        kandidat_promene = np.abs(
            np.diff(kandidat, axis=0)
        )

        promena_skor = 1.0 - float(
            np.mean(
                np.abs(
                    upit_promene
                    - kandidat_promene
                )
            )
        )
    else:
        promena_skor = 0.0

    return (
        0.80 * osnovni_skor
        + 0.20 * promena_skor
    )


# =============================================================================
# K-MER/MINIMIZER PRETRAGA REFERENTNOG CSV GENOMA
# =============================================================================

def pronadji_slicne_regione(
    matrica: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(matrica)

    if n < KMER_DUZINA + 2:
        return (
            np.empty(0, dtype=int),
            np.empty(0, dtype=float),
        )

    upit = matrica[-KMER_DUZINA:]
    upit_tokeni = prozor_u_tokene(upit)
    upit_potpis = minhash_potpis(upit_tokeni)

    # Kraj istorijskog prozora mora imati poznato sledeće izvlačenje.
    krajevi = np.arange(
        KMER_DUZINA,
        n,
        dtype=int,
    )

    prozori = [
        matrica[
            kraj - KMER_DUZINA:kraj
        ]
        for kraj in krajevi
    ]

    potpisi = [
        minhash_potpis(
            prozor_u_tokene(prozor)
        )
        for prozor in prozori
    ]

    indeks = napravi_minhash_indeks(
        potpisi
    )

    kandidati = kandidati_iz_indeksa(
        upit_potpis,
        indeks,
        len(prozori),
    )

    # MinHash rezervno rangiranje dodaje najbolje približne potpise.
    minhash_skorovi = np.asarray(
        [
            minhash_slicnost(
                upit_potpis,
                potpisi[i],
            )
            for i in range(len(potpisi))
        ]
    )

    dodatni = np.argsort(
        minhash_skorovi,
        kind="stable",
    )[-MINHASH_KANDIDATI:]

    kandidati = np.unique(
        np.concatenate(
            [kandidati, dodatni]
        )
    )

    poravnanja = []

    for indeks_prozora in kandidati:
        skor = sekvencijalno_poravnanje(
            upit,
            prozori[indeks_prozora],
        )

        poravnanja.append(
            (
                int(krajevi[indeks_prozora]),
                float(skor),
            )
        )

    poravnanja.sort(
        key=lambda x: (
            -x[1],
            -x[0],
        )
    )

    najbolji = poravnanja[
        :BROJ_SLICNIH_REGIONA
    ]

    if not najbolji:
        return (
            np.empty(0, dtype=int),
            np.empty(0, dtype=float),
        )

    najbolji_krajevi = np.asarray(
        [x[0] for x in najbolji],
        dtype=int,
    )

    najbolji_skorovi = np.asarray(
        [x[1] for x in najbolji],
        dtype=float,
    )

    # Sličnost se pretvara u pozitivne težine.
    najbolji_skorovi -= najbolji_skorovi.max()

    tezine = np.exp(
        8.0 * najbolji_skorovi
    )

    # Noviji istorijski regioni dobijaju blagu dodatnu važnost.
    vremenska_tezina = np.exp(
        -(
            n - najbolji_krajevi
        ) / max(200.0, n / 3.0)
    )

    tezine *= vremenska_tezina
    tezine /= max(float(tezine.sum()), EPS)

    return najbolji_krajevi, tezine


def kmer_nastavci_model(
    matrica: np.ndarray,
    krajevi: np.ndarray,
    tezine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(krajevi) == 0:
        neutralno = np.full(
            BROJ_KUGLICA,
            OSNOVNA_STOPA,
        )

        return (
            np.zeros(BROJ_KUGLICA),
            np.zeros(
                (BROJ_KUGLICA, BROJ_KUGLICA)
            ),
        )

    nastavci = matrica[krajevi]

    marginalno = (
        tezine @ nastavci
    )

    parovi = np.zeros(
        (BROJ_KUGLICA, BROJ_KUGLICA),
        dtype=float,
    )

    for red, tezina in zip(
        nastavci,
        tezine,
    ):
        aktivni = np.flatnonzero(red > 0.5)

        for i, prvi in enumerate(aktivni):
            for drugi in aktivni[i + 1:]:
                parovi[prvi, drugi] += tezina
                parovi[drugi, prvi] += tezina

    ocekivani_par = (
        BROJEVA_U_KOMBINACIJI
        * (BROJEVA_U_KOMBINACIJI - 1)
        / (
            BROJ_KUGLICA
            * (BROJ_KUGLICA - 1)
        )
    )

    parovi = np.log(
        np.clip(
            (
                parovi + ocekivani_par
            ) / (
                2.0 * ocekivani_par
            ),
            0.25,
            4.0,
        )
    )

    return (
        standardizuj(
            marginalno - OSNOVNA_STOPA
        ),
        parovi,
    )


# =============================================================================
# DE BRUIJN GRAF
# =============================================================================

def diskretni_simbol(
    red: np.ndarray,
) -> tuple[int, int, int, int]:
    brojevi = np.flatnonzero(red > 0.5) + 1

    zbir = int(np.sum(brojevi))
    neparni = int(np.sum(brojevi % 2))
    niski = int(np.sum(brojevi <= 19))
    raspon = int(np.max(brojevi) - np.min(brojevi))

    return (
        zbir // 15,
        neparni,
        niski,
        raspon // 6,
    )


def de_bruijn_model(
    matrica: np.ndarray,
) -> np.ndarray:
    n = len(matrica)

    if n < 100:
        return np.zeros(BROJ_KUGLICA)

    simboli = [
        diskretni_simbol(red)
        for red in matrica
    ]

    maksimalni_red = min(
        DE_BRUIJN_RED,
        n - 2,
    )

    brojanja = np.zeros(
        BROJ_KUGLICA,
        dtype=float,
    )

    ukupna_tezina = 0.0

    # Backoff: prvo puni red grafa, zatim kraći sufiksi.
    for red_grafa in range(
        maksimalni_red,
        0,
        -1,
    ):
        upit = tuple(
            simboli[-red_grafa:]
        )

        lokalna_brojanja = np.zeros(
            BROJ_KUGLICA,
            dtype=float,
        )

        lokalna_tezina = 0.0

        for kraj in range(
            red_grafa,
            n,
        ):
            stanje = tuple(
                simboli[
                    kraj - red_grafa:kraj
                ]
            )

            if stanje != upit:
                continue

            starost = n - kraj

            tezina = (
                red_grafa**2
                * math.exp(
                    -starost / max(200.0, n / 3.0)
                )
            )

            lokalna_brojanja += (
                tezina * matrica[kraj]
            )

            lokalna_tezina += tezina

        if lokalna_tezina > 0:
            brojanja += lokalna_brojanja
            ukupna_tezina += lokalna_tezina

            # Najduži pronađeni kontekst ima prednost.
            if red_grafa == maksimalni_red:
                break

    if ukupna_tezina < EPS:
        return np.zeros(BROJ_KUGLICA)

    prior = 10.0

    procena = (
        brojanja
        + prior * OSNOVNA_STOPA
    ) / (
        ukupna_tezina + prior
    )

    return standardizuj(
        procena - OSNOVNA_STOPA
    )


# =============================================================================
# HIDDEN MARKOV REŽIMI
# =============================================================================

def osobine_izvlacenja(
    matrica: np.ndarray,
) -> np.ndarray:
    brojevi = np.arange(
        1,
        BROJ_KUGLICA + 1,
        dtype=float,
    )

    zbir = matrica @ brojevi
    neparni = matrica @ (brojevi % 2)
    niski = matrica[:, :19].sum(axis=1)

    sredina = zbir / BROJEVA_U_KOMBINACIJI

    centrirani = (
        brojevi[None, :]
        - sredina[:, None]
    ) * matrica

    rasipanje = np.sqrt(
        np.sum(
            centrirani**2,
            axis=1,
        ) / BROJEVA_U_KOMBINACIJI
    )

    osobine = np.column_stack(
        [
            zbir,
            neparni,
            niski,
            rasipanje,
        ]
    )

    prosek = osobine.mean(axis=0)
    std = osobine.std(axis=0)
    std[std < EPS] = 1.0

    return (
        osobine - prosek
    ) / std


def hmm_model(
    matrica: np.ndarray,
) -> np.ndarray:
    if len(matrica) < 150:
        return np.zeros(BROJ_KUGLICA)

    istorija = matrica[
        -min(len(matrica), HMM_ISTORIJA):
    ]

    osobine = osobine_izvlacenja(
        istorija
    )

    model = GaussianHMM(
        n_components=BROJ_HMM_REZIMA,
        covariance_type="diag",
        n_iter=HMM_ITERACIJE,
        tol=1e-3,
        random_state=SEED,
        min_covar=1e-4,
    )

    try:
        model.fit(osobine)
        stanja = model.predict(osobine)
    except Exception:
        return np.zeros(BROJ_KUGLICA)

    poslednje_stanje = int(
        stanja[-1]
    )

    sledece_tezine = model.transmat_[
        poslednje_stanje
    ]

    rezultat = np.zeros(
        BROJ_KUGLICA,
        dtype=float,
    )

    for stanje in range(BROJ_HMM_REZIMA):
        maska = stanja == stanje
        broj_redova = int(np.sum(maska))

        if broj_redova == 0:
            stopa = np.full(
                BROJ_KUGLICA,
                OSNOVNA_STOPA,
            )
        else:
            brojanja = istorija[
                maska
            ].sum(axis=0)

            prior = max(
                20.0,
                broj_redova * 0.20,
            )

            stopa = (
                brojanja
                + prior * OSNOVNA_STOPA
            ) / (
                broj_redova + prior
            )

        rezultat += (
            sledece_tezine[stanje]
            * stopa
        )

    return standardizuj(
        rezultat - OSNOVNA_STOPA
    )


# =============================================================================
# BAJESOV VARIANT CALLING
# =============================================================================

def bajesov_variant_calling(
    matrica: np.ndarray,
    krajevi: np.ndarray,
    tezine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Referentna stopa 7/39 predstavlja referentni genom.

    Nastavci sličnih istorijskih regiona predstavljaju očitavanja.
    Bajesov posterior određuje da li odstupanje ima dovoljno podrške.
    """

    if len(krajevi) == 0:
        sredina = np.full(
            BROJ_KUGLICA,
            OSNOVNA_STOPA,
        )

        sirina = np.ones(
            BROJ_KUGLICA,
        )

        return (
            np.zeros(BROJ_KUGLICA),
            sredina,
            sirina,
        )

    nastavci = matrica[krajevi]

    # Težine se skaliraju na efektivni broj regiona.
    efektivne_tezine = (
        tezine * len(tezine)
    )

    uspesi = (
        efektivne_tezine @ nastavci
    )

    efektivni_n = float(
        efektivne_tezine.sum()
    )

    alfa_prior = (
        BAJES_PRIOR_SNAGA
        * OSNOVNA_STOPA
    )

    beta_prior = (
        BAJES_PRIOR_SNAGA
        * (1.0 - OSNOVNA_STOPA)
    )

    alfa = alfa_prior + uspesi

    beta_parametar = (
        beta_prior
        + efektivni_n
        - uspesi
    )

    sredina = alfa / (
        alfa + beta_parametar
    )

    donja = beta_raspodela.ppf(
        0.025,
        alfa,
        beta_parametar,
    )

    gornja = beta_raspodela.ppf(
        0.975,
        alfa,
        beta_parametar,
    )

    sirina = gornja - donja

    skor = (
        sredina
        - OSNOVNA_STOPA
        - KAZNA_NEIZVESNOSTI * sirina
    )

    return (
        standardizuj(skor),
        sredina,
        sirina,
    )


# =============================================================================
# KOMPLETAN MODEL
# =============================================================================

def izracunaj_modele(
    matrica: np.ndarray,
) -> tuple[np.ndarray, dict]:
    krajevi, slicnost_tezine = pronadji_slicne_regione(
        matrica
    )

    kmer_skor, parni_skor = kmer_nastavci_model(
        matrica,
        krajevi,
        slicnost_tezine,
    )

    de_bruijn_skor = de_bruijn_model(
        matrica
    )

    hmm_skor = hmm_model(
        matrica
    )

    (
        bajes_skor,
        bajes_sredina,
        bajes_sirina,
    ) = bajesov_variant_calling(
        matrica,
        krajevi,
        slicnost_tezine,
    )

    skorovi = np.vstack(
        [
            kmer_skor,
            de_bruijn_skor,
            hmm_skor,
            bajes_skor,
        ]
    )

    verovatnoce = np.vstack(
        [
            skor_u_verovatnoce(skor)
            for skor in skorovi
        ]
    )

    dodatno = {
        "krajevi": krajevi,
        "broj_regiona": len(krajevi),
        "parni_skor": parni_skor,
        "bajes_sredina": bajes_sredina,
        "bajes_sirina": bajes_sirina,
    }

    return verovatnoce, dodatno


def kombinuj_modele(
    modeli: np.ndarray,
    tezine: np.ndarray,
) -> np.ndarray:
    rezultat = np.average(
        modeli,
        axis=0,
        weights=tezine,
    )

    rezultat *= (
        BROJEVA_U_KOMBINACIJI
        / max(float(rezultat.sum()), EPS)
    )

    return np.clip(
        rezultat,
        EPS,
        1.0 - EPS,
    )


# =============================================================================
# SASTAVLJANJE NEXT SEDMORKE
# =============================================================================

def skor_kombinacije(
    kombinacija: tuple[int, ...],
    verovatnoce: np.ndarray,
    parni_skor: np.ndarray,
) -> float:
    indeksi = np.asarray(
        kombinacija,
        dtype=int,
    )

    marginalni = float(
        np.sum(
            np.log(
                np.clip(
                    verovatnoce[indeksi],
                    EPS,
                    1.0,
                )
            )
        )
    )

    parni = 0.0
    broj_parova = 0

    for i, prvi in enumerate(indeksi):
        for drugi in indeksi[i + 1:]:
            parni += parni_skor[prvi, drugi]
            broj_parova += 1

    if broj_parova:
        parni /= broj_parova

    return (
        marginalni
        + TEZINA_PARNOG_SKORA * parni
    )


def izaberi_sedam(
    verovatnoce: np.ndarray,
    parni_skor: np.ndarray,
    iscrpno: bool,
) -> np.ndarray:
    """
    Tokom validacije koristi determinističku lokalnu optimizaciju.

    Za završnu NEXT predikciju iscrpno proverava sve sedmorke
    sastavljene od 18 Bajesovo najrelevantnijih brojeva.
    """

    relevantni = np.argsort(
        verovatnoce,
        kind="stable",
    )[-BROJ_RELEVANTNIH_BROJEVA:]

    if iscrpno:
        najbolja = None
        najbolji_skor = -np.inf

        for kombinacija in itertools.combinations(
            relevantni,
            BROJEVA_U_KOMBINACIJI,
        ):
            skor = skor_kombinacije(
                kombinacija,
                verovatnoce,
                parni_skor,
            )

            if skor > najbolji_skor:
                najbolji_skor = skor
                najbolja = kombinacija

        return np.sort(
            np.asarray(najbolja, dtype=int) + 1
        )

    trenutno = set(
        np.argsort(
            verovatnoce,
            kind="stable",
        )[-BROJEVA_U_KOMBINACIJI:].tolist()
    )

    poboljsanje = True

    while poboljsanje:
        poboljsanje = False

        trenutna_tuple = tuple(
            sorted(trenutno)
        )

        trenutni_skor = skor_kombinacije(
            trenutna_tuple,
            verovatnoce,
            parni_skor,
        )

        najbolji_skup = trenutno
        najbolji_skor = trenutni_skor

        van = [
            int(x)
            for x in relevantni
            if int(x) not in trenutno
        ]

        for izbaceni in sorted(trenutno):
            for ubaceni in van:
                kandidat = set(trenutno)
                kandidat.remove(izbaceni)
                kandidat.add(ubaceni)

                kandidat_skor = skor_kombinacije(
                    tuple(sorted(kandidat)),
                    verovatnoce,
                    parni_skor,
                )

                if kandidat_skor > najbolji_skor + 1e-12:
                    najbolji_skor = kandidat_skor
                    najbolji_skup = kandidat
                    poboljsanje = True

        trenutno = najbolji_skup

    return np.sort(
        np.asarray(
            sorted(trenutno),
            dtype=int,
        ) + 1
    )


# =============================================================================
# HRONOLOŠKA VALIDACIJA
# =============================================================================

def napravi_walk_forward(
    matrica: np.ndarray,
    pocetak: int,
    kraj: int,
    naziv: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predikcije = []
    parni_skorovi = []
    ishodi = []

    ukupno = kraj - pocetak

    for redni, granica in enumerate(
        range(pocetak, kraj),
        start=1,
    ):
        modeli, dodatno = izracunaj_modele(
            matrica[:granica]
        )

        predikcije.append(modeli)
        parni_skorovi.append(
            dodatno["parni_skor"]
        )
        ishodi.append(matrica[granica])

        if (
            redni == 1
            or redni == ukupno
            or redni % 10 == 0
        ):
            print(
                f"  {naziv}: {redni}/{ukupno}"
            )

    return (
        np.asarray(predikcije, dtype=float),
        np.asarray(parni_skorovi, dtype=float),
        np.asarray(ishodi, dtype=float),
    )


def funkcija_gubitka(
    tezine: np.ndarray,
    predikcije: np.ndarray,
    ishodi: np.ndarray,
) -> float:
    kombinovano = np.einsum(
        "tmn,m->tn",
        predikcije,
        tezine,
    )

    brier = float(
        np.mean(
            (kombinovano - ishodi) ** 2
        )
    )

    ravnomerno = np.full(
        len(tezine),
        1.0 / len(tezine),
    )

    regularizacija = 0.002 * float(
        np.sum(
            (tezine - ravnomerno) ** 2
        )
    )

    return brier + regularizacija


def odredi_tezine(
    predikcije: np.ndarray,
    ishodi: np.ndarray,
) -> np.ndarray:
    broj_modela = predikcije.shape[1]

    pocetne = np.full(
        broj_modela,
        1.0 / broj_modela,
    )

    rezultat = minimize(
        funkcija_gubitka,
        pocetne,
        args=(predikcije, ishodi),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * broj_modela,
        constraints={
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
        },
        options={
            "maxiter": 500,
            "ftol": 1e-12,
        },
    )

    if not rezultat.success:
        return pocetne

    tezine = np.clip(
        rezultat.x,
        0.0,
        1.0,
    )

    if tezine.sum() < EPS:
        return pocetne

    return tezine / tezine.sum()


def oceni_holdout(
    predikcije: np.ndarray,
    parni_skorovi: np.ndarray,
    ishodi: np.ndarray,
    tezine: np.ndarray,
) -> dict:
    pogodci = []
    brier = []

    for modeli, parni_skor, stvarno in zip(
        predikcije,
        parni_skorovi,
        ishodi,
    ):
        verovatnoce = kombinuj_modele(
            modeli,
            tezine,
        )

        izbor = izaberi_sedam(
            verovatnoce,
            parni_skor,
            iscrpno=False,
        )

        pogodci.append(
            int(
                np.sum(
                    stvarno[izbor - 1]
                )
            )
        )

        brier.append(
            float(
                np.mean(
                    (verovatnoce - stvarno) ** 2
                )
            )
        )

    pogodci = np.asarray(
        pogodci,
        dtype=int,
    )

    return {
        "pogodci": pogodci,
        "prosek": float(np.mean(pogodci)),
        "brier": float(np.mean(brier)),
        "najmanje_3": float(
            np.mean(pogodci >= 3)
        ),
        "najmanje_4": float(
            np.mean(pogodci >= 4)
        ),
        "maksimum": int(np.max(pogodci)),
    }


# =============================================================================
# BLOK-BOOTSTRAP
# =============================================================================

def blok_bootstrap(
    pogodci: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    pogodci = np.asarray(
        pogodci,
        dtype=float,
    )

    n = len(pogodci)

    broj_blokova = math.ceil(
        n / BOOTSTRAP_BLOK
    )

    najveci_pocetak = max(
        1,
        n - BOOTSTRAP_BLOK + 1,
    )

    proseci = np.empty(
        BROJ_BOOTSTRAP_SIMULACIJA,
        dtype=float,
    )

    for simulacija in range(
        BROJ_BOOTSTRAP_SIMULACIJA
    ):
        delovi = []

        for _ in range(broj_blokova):
            pocetak = int(
                rng.integers(
                    0,
                    najveci_pocetak,
                )
            )

            delovi.append(
                pogodci[
                    pocetak:
                    pocetak + BOOTSTRAP_BLOK
                ]
            )

        uzorak = np.concatenate(delovi)[:n]
        proseci[simulacija] = np.mean(uzorak)

    donja, gornja = np.quantile(
        proseci,
        [0.025, 0.975],
    )

    return float(donja), float(gornja)


# =============================================================================
# OBRADA JEDNE IGRE
# =============================================================================

def obradi_igru(
    naziv: str,
    csv_putanja: Path,
    seed_pomeraj: int,
) -> dict:
    _, matrica = ucitaj_csv(csv_putanja)
    n = len(matrica)

    naslov(f"OBRADA: {naziv}")

    print(f"CSV: {csv_putanja}")
    print(f"Broj redova: {n}")
    print("Prvi red se tretira kao najstariji.")
    print("Poslednji red se tretira kao najnoviji.")

    potrebno = (
        MINIMUM_ISTORIJE
        + BROJ_VALIDACIONIH_KORAKA
        + BROJ_HOLDOUT_KORAKA
    )

    if n < potrebno:
        raise RuntimeError(
            f"{naziv}: potrebno je najmanje {potrebno} redova, "
            f"a pronađeno je {n}."
        )

    holdout_pocetak = (
        n - BROJ_HOLDOUT_KORAKA
    )

    validacija_pocetak = (
        holdout_pocetak
        - BROJ_VALIDACIONIH_KORAKA
    )

    print()
    print("Razvojna walk-forward validacija...")

    (
        razvoj_predikcije,
        _,
        razvoj_ishodi,
    ) = napravi_walk_forward(
        matrica=matrica,
        pocetak=validacija_pocetak,
        kraj=holdout_pocetak,
        naziv="Razvoj",
    )

    tezine = odredi_tezine(
        razvoj_predikcije,
        razvoj_ishodi,
    )

    print()
    print("Zamrznuta holdout provera...")

    (
        holdout_predikcije,
        holdout_parni,
        holdout_ishodi,
    ) = napravi_walk_forward(
        matrica=matrica,
        pocetak=holdout_pocetak,
        kraj=n,
        naziv="Holdout",
    )

    holdout = oceni_holdout(
        holdout_predikcije,
        holdout_parni,
        holdout_ishodi,
        tezine,
    )

    rng = np.random.default_rng(
        SEED + seed_pomeraj
    )

    bootstrap_donja, bootstrap_gornja = blok_bootstrap(
        holdout["pogodci"],
        rng,
    )

    zavrsni_modeli, dodatno = izracunaj_modele(
        matrica
    )

    zavrsne_verovatnoce = kombinuj_modele(
        zavrsni_modeli,
        tezine,
    )

    next_kombinacija = izaberi_sedam(
        zavrsne_verovatnoce,
        dodatno["parni_skor"],
        iscrpno=True,
    )

    statisticki_pouzdan = (
        holdout["prosek"] > SLUCAJNO_OCEKIVANJE
        and bootstrap_donja > SLUCAJNO_OCEKIVANJE
    )

    return {
        "naziv": naziv,
        "csv_putanja": csv_putanja,
        "broj_redova": n,
        "next": next_kombinacija,
        "tezine": tezine,
        "holdout": holdout,
        "bootstrap_donja": bootstrap_donja,
        "bootstrap_gornja": bootstrap_gornja,
        "statisticki_pouzdan": statisticki_pouzdan,
        "verovatnoce": zavrsne_verovatnoce,
        "broj_regiona": dodatno["broj_regiona"],
        "bajes_sredina": dodatno["bajes_sredina"],
        "bajes_sirina": dodatno["bajes_sirina"],
    }


# =============================================================================
# ISPIS
# =============================================================================

def ispisi_rezultat(
    rezultat: dict,
) -> None:
    naslov(
        rezultat["naziv"],
        znak="#",
    )

    print(
        f"NEXT: "
        f"{formatiraj_kombinaciju(rezultat['next'])}"
    )

    print(
        f"CSV redova: {rezultat['broj_redova']}"
    )

    print(
        f"Sličnih istorijskih regiona: "
        f"{rezultat['broj_regiona']}"
    )

    print()
    print("Težine modela:")

    for naziv, tezina in zip(
        NAZIVI_MODELA,
        rezultat["tezine"],
    ):
        print(
            f"  {naziv:<31} {tezina:>8.2%}"
        )

    print()
    print("Izabrani brojevi i Bajesova procena:")

    for broj in rezultat["next"]:
        indeks = int(broj) - 1

        print(
            f"  Broj {broj:02d}"
            f" — ensemble {rezultat['verovatnoce'][indeks]:.6f}"
            f" — Bajes {rezultat['bajes_sredina'][indeks]:.6f}"
            f" — širina 95% intervala "
            f"{rezultat['bajes_sirina'][indeks]:.6f}"
        )

    holdout = rezultat["holdout"]

    print()
    print("Zamrznuta holdout provera:")

    print(
        f"  Broj koraka:                  "
        f"{len(holdout['pogodci'])}"
    )

    print(
        f"  Prosečan broj pogodaka:       "
        f"{holdout['prosek']:.6f}"
    )

    print(
        f"  Slučajno očekivanje:          "
        f"{SLUCAJNO_OCEKIVANJE:.6f}"
    )

    print(
        f"  Brierov skor:                 "
        f"{holdout['brier']:.6f}"
    )

    print(
        f"  Najmanje tri pogotka:         "
        f"{holdout['najmanje_3']:.2%}"
    )

    print(
        f"  Najmanje četiri pogotka:      "
        f"{holdout['najmanje_4']:.2%}"
    )

    print(
        f"  Najveći broj pogodaka:        "
        f"{holdout['maksimum']}"
    )

    print(
        f"  Blok-bootstrap 95% interval:  "
        f"[{rezultat['bootstrap_donja']:.6f}, "
        f"{rezultat['bootstrap_gornja']:.6f}]"
    )

    print()
    print("GLAVNI ODGOVOR:")

    if rezultat["statisticki_pouzdan"]:
        print(
            "  DA — donja granica 95% hronološkog bootstrap "
            "intervala prelazi slučajno očekivanje."
        )
    else:
        print(
            "  NE — na zamrznutom holdoutu nije potvrđena "
            "statistički pouzdana prediktivna prednost"
        )

        print(
            f"  iznad slučajnog očekivanja od "
            f"{SLUCAJNO_OCEKIVANJE:.6f} pogodaka."
        )


# =============================================================================
# GLAVNI PROGRAM
# =============================================================================

def main() -> None:
    np.random.seed(SEED)

    naslov(
        "LOTO 7/39 — GENOMSKI PATTERN RECOGNITION SISTEM"
    )

    print(f"Seed: {SEED}")
    print(
        f"Teorijska stopa broja: "
        f"{OSNOVNA_STOPA:.9f}"
    )
    print(
        f"Teorijsko očekivanje pogodaka: "
        f"{SLUCAJNO_OCEKIVANJE:.9f}"
    )
    print(
        f"Ukupno mogućih kombinacija: "
        f"{math.comb(BROJ_KUGLICA, BROJEVA_U_KOMBINACIJI):,}"
    )

    print()
    print(
        "CSV referentni genom → k-mer/minimizer indeks i pretraga "
        "→ sekvencijalno poravnanje"
    )

    print(
        "→ slični istorijski regioni → De Bruijn graf nastavaka "
        "→ Hidden Markov režimi"
    )

    print(
        "→ Bajesov variant calling → relevantne NEXT sedmorke "
        "→ walk-forward → holdout"
    )

    loto = obradi_igru(
        naziv="Loto",
        csv_putanja=LOTO_CSV,
        seed_pomeraj=0,
    )

    loto_plus = obradi_igru(
        naziv="Loto Plus",
        csv_putanja=LOTO_PLUS_CSV,
        seed_pomeraj=1,
    )

    naslov(
        "KONAČNE NEXT PREDIKCIJE",
        znak="#",
    )

    print(
        f"Loto:      "
        f"{formatiraj_kombinaciju(loto['next'])}"
    )

    print(
        f"Loto Plus: "
        f"{formatiraj_kombinaciju(loto_plus['next'])}"
    )

    ispisi_rezultat(loto)
    ispisi_rezultat(loto_plus)

    naslov(
        "ZAVRŠNI ZAKLJUČAK",
        znak="#",
    )

    print("Loto:")

    if loto["statisticki_pouzdan"]:
        print(
            "  Potvrđena je statistički pouzdana prediktivna "
            "prednost na zamrznutom holdoutu."
        )
    else:
        print(
            "  Nije potvrđena statistički pouzdana prediktivna "
            "prednost na zamrznutom holdoutu."
        )

    print()
    print("Loto Plus:")

    if loto_plus["statisticki_pouzdan"]:
        print(
            "  Potvrđena je statistički pouzdana prediktivna "
            "prednost na zamrznutom holdoutu."
        )
    else:
        print(
            "  Nije potvrđena statistički pouzdana prediktivna "
            "prednost na zamrznutom holdoutu."
        )


if __name__ == "__main__":
    main()



"""
==============================================================================
LOTO 7/39 — GENOMSKI PATTERN RECOGNITION SISTEM
==============================================================================
Seed: 39
Teorijska stopa broja: 0.179487179
Teorijsko očekivanje pogodaka: 1.256410256
Ukupno mogućih kombinacija: 15,380,937

CSV referentni genom → k-mer/minimizer indeks i pretraga → sekvencijalno poravnanje
→ slični istorijski regioni → De Bruijn graf nastavaka → Hidden Markov režimi
→ Bajesov variant calling → relevantne NEXT sedmorke → walk-forward → holdout

==============================================================================
OBRADA: Loto
==============================================================================
CSV: /Users/4c/Desktop/GHQ/data/loto7_4680_k71_loto_2962.csv
Broj redova: 2962
Prvi red se tretira kao najstariji.
Poslednji red se tretira kao najnoviji.

Razvojna walk-forward validacija...
  Razvoj: 1/120
  Razvoj: 10/120
  Razvoj: 20/120
  Razvoj: 30/120
  Razvoj: 40/120
  Razvoj: 50/120
  Razvoj: 60/120
  Razvoj: 70/120
  Razvoj: 80/120
  Razvoj: 90/120
  Razvoj: 100/120
  Razvoj: 110/120
  Razvoj: 120/120

Zamrznuta holdout provera...
  Holdout: 1/160
  Holdout: 10/160
  Holdout: 20/160
  Holdout: 30/160
  Holdout: 40/160
  Holdout: 50/160
  Holdout: 60/160
  Holdout: 70/160
  Holdout: 80/160
  Holdout: 90/160
  Holdout: 100/160
  Holdout: 110/160
  Holdout: 120/160
  Holdout: 130/160
  Holdout: 140/160
  Holdout: 150/160
  Holdout: 160/160

==============================================================================
OBRADA: Loto Plus
==============================================================================
CSV: /Users/4c/Desktop/GHQ/data/loto7_4680_k71_loto_plus_1718.csv
Broj redova: 1718
Prvi red se tretira kao najstariji.
Poslednji red se tretira kao najnoviji.

Razvojna walk-forward validacija...
  Razvoj: 1/120
  Razvoj: 10/120
  Razvoj: 20/120
  Razvoj: 30/120
  Razvoj: 40/120
  Razvoj: 50/120
  Razvoj: 60/120
  Razvoj: 70/120
  Razvoj: 80/120
  Razvoj: 90/120
  Razvoj: 100/120
  Razvoj: 110/120
  Razvoj: 120/120

Zamrznuta holdout provera...
  Holdout: 1/160
  Holdout: 10/160
  Holdout: 20/160
  Holdout: 30/160
  Holdout: 40/160
  Holdout: 50/160
  Holdout: 60/160
  Holdout: 70/160
  Holdout: 80/160
  Holdout: 90/160
  Holdout: 100/160
  Holdout: 110/160
  Holdout: 120/160
  Holdout: 130/160
  Holdout: 140/160
  Holdout: 150/160
  Holdout: 160/160

##############################################################################
KONAČNE NEXT PREDIKCIJE
##############################################################################
Loto:      19, 22, 23, 24, 25, 26, 35
Loto Plus: 08, 22, 23, 27, 31, 33, 34

##############################################################################
Loto
##############################################################################
NEXT: 19, 22, 23, 24, 25, 26, 35
CSV redova: 2962
Sličnih istorijskih regiona: 60

Težine modela:
  Genomski k-mer nastavci           18.26%
  De Bruijn graf                    37.57%
  Hidden Markov režimi              27.43%
  Bajesov variant calling           16.74%

Izabrani brojevi i Bajesova procena:
  Broj 19 — ensemble 0.309009 — Bajes 0.254798 — širina 95% intervala 0.189019
  Broj 22 — ensemble 0.277077 — Bajes 0.127074 — širina 95% intervala 0.143902
  Broj 23 — ensemble 0.248432 — Bajes 0.212435 — širina 95% intervala 0.177296
  Broj 24 — ensemble 0.278963 — Bajes 0.272779 — širina 95% intervala 0.193248
  Broj 25 — ensemble 0.243321 — Bajes 0.232091 — širina 95% intervala 0.183061
  Broj 26 — ensemble 0.303035 — Bajes 0.191166 — širina 95% intervala 0.170354
  Broj 35 — ensemble 0.241860 — Bajes 0.155029 — širina 95% intervala 0.156606

Zamrznuta holdout provera:
  Broj koraka:                  160
  Prosečan broj pogodaka:       1.293750
  Slučajno očekivanje:          1.256410
  Brierov skor:                 0.152474
  Najmanje tri pogotka:         8.75%
  Najmanje četiri pogotka:      0.62%
  Najveći broj pogodaka:        4
  Blok-bootstrap 95% interval:  [1.162500, 1.400000]

GLAVNI ODGOVOR:
  NE — na zamrznutom holdoutu nije potvrđena statistički pouzdana prediktivna prednost
  iznad slučajnog očekivanja od 1.256410 pogodaka.

##############################################################################
Loto Plus
##############################################################################
NEXT: 08, 22, 23, 27, 31, 33, 34
CSV redova: 1718
Sličnih istorijskih regiona: 60

Težine modela:
  Genomski k-mer nastavci           20.03%
  De Bruijn graf                    30.99%
  Hidden Markov režimi              30.38%
  Bajesov variant calling           18.60%

Izabrani brojevi i Bajesova procena:
  Broj 08 — ensemble 0.251318 — Bajes 0.207507 — širina 95% intervala 0.175756
  Broj 22 — ensemble 0.248489 — Bajes 0.207897 — širina 95% intervala 0.175879
  Broj 23 — ensemble 0.385405 — Bajes 0.235559 — širina 95% intervala 0.184018
  Broj 27 — ensemble 0.249374 — Bajes 0.184128 — širina 95% intervala 0.167881
  Broj 31 — ensemble 0.347245 — Bajes 0.303560 — širina 95% intervala 0.199565
  Broj 33 — ensemble 0.269252 — Bajes 0.264137 — širina 95% intervala 0.191267
  Broj 34 — ensemble 0.425316 — Bajes 0.202158 — širina 95% intervala 0.174038

Zamrznuta holdout provera:
  Broj koraka:                  160
  Prosečan broj pogodaka:       1.312500
  Slučajno očekivanje:          1.256410
  Brierov skor:                 0.151580
  Najmanje tri pogotka:         11.88%
  Najmanje četiri pogotka:      1.25%
  Najveći broj pogodaka:        4
  Blok-bootstrap 95% interval:  [1.150000, 1.431250]

GLAVNI ODGOVOR:
  NE — na zamrznutom holdoutu nije potvrđena statistički pouzdana prediktivna prednost
  iznad slučajnog očekivanja od 1.256410 pogodaka.

##############################################################################
ZAVRŠNI ZAKLJUČAK
##############################################################################
Loto:
  Nije potvrđena statistički pouzdana prediktivna prednost na zamrznutom holdoutu.

Loto Plus:
  Nije potvrđena statistički pouzdana prediktivna prednost na zamrznutom holdoutu.
"""



"""
Ljudski genom ima približno 3,2 milijarde parova DNK baza. Na svakom mestu moguća su četiri slova: A, C, G i T.
Ako zamišljamo sve moguće DNK nizove dužine 3,2 milijarde, broj kombinacija je 
broj sa oko 1,93 milijarde cifara.
Redosled približno 20.000 ljudskih gena, broj mogućih poredaka bio bi: 20000!
Međutim, geni se u stvarnom genomu ne kombinuju proizvoljno: većina DNK je zajednička ljudima, a razlike nastaju kroz milione varijanti, strukturne promene i njihove kombinacije. Zato ne postoji jedan jedini praktičan broj „svih mogućih ljudskih genoma“.

Ljudski genom sadrži približno 3,2 milijarde pozicija, a na svakoj mogu biti četiri DNK baze: A, C, G ili T.
Zato je teorijski broj mogućih ljudskih DNK sekvenci broj sa približno 1,93 milijarde cifara. 


Naučnici uglavnom ne pregledaju sve moguće ljudske genome. To bi bilo fizički nemoguće. Oni ogromni prostor svode na mali deo koji stvarno postoji ili je relevantan:
- referentni genom predstavlja osnovu, pa se čuvaju samo razlike;
- analiziraju se stvarno izmerene sekvence, ne svih \(4^{3,2\text{ milijarde}}\) mogućnosti;
- genom se deli na regione i obrađuje paralelno;
- koriste se indeksi, heširanje, grafovi i kompresovane strukture podataka;
- modeli pretražuju samo verovatne kandidate;
- aproksimacije i statist se koriste umesto potpunog nabrajanja.


Pravo ograničenje je sledeće: čak i kada svih 15 miliona kombinacija precizno rangiramo, potreban je pouzdan prediktivni signal koji bi pravilnu kombinaciju stavio bliže vrhu. Ako su izvlačenja nezavisna i poštena, svaka kombinacija ima istu verovatnoću:
1/15,380,937
Dakle, računarska obrada 15 miliona kandidata nije glavni problem; problem je što istorijski podaci možda ne sadrže informaciju koja razlikuje budućeg dobitnika od ostalih kombinacija.


CSV referentni genom
→ k-mer/minimizer indeks
→ sekvencijalno poravnanje
→ slični istorijski regioni
→ De Bruijn graf nastavaka
→ Hidden Markov režimi
→ Bajesov variant calling
→ sužavanje prostora na relevantne kandidate
→ sastavljanje i ocenjivanje NEXT sedmorke
→ stroga hronološka walk-forward validacija
→ zamrznuta holdout provera
"""
