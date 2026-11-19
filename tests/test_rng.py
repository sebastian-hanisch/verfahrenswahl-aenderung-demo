"""Zufallsquelle: die Stapel-Fassung (vwa_normal) ist Bit für Bit gleich dem SplitMix64 der Vorlage; die Normalzahlen gelten auf 1e-15, sonst als Band."""

import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

import vwa_normal as N
import vwa_rng


@pytest.mark.parametrize("seed", [0, 1, 12345, 2 ** 63 + 5, 2 ** 64 - 1])
def test_stapel_gleich_klasse_bitgleich(seed):
    r = vwa_rng.SplitMix64(seed)
    ref = [r.next_u64() for _ in range(700)]
    blk = N.u64_block(seed, 0, 700)
    assert [int(v) for v in blk] == ref


def test_stapel_mit_versatz_setzt_fort():
    r = vwa_rng.SplitMix64(77)
    ref = [r.next_u64() for _ in range(300)]
    assert [int(v) for v in N.u64_block(77, 120, 100)] == ref[120:220]


def test_uniform_gleich_klasse_bis_auf_halbes_bit():
    """Klasse: u = (k) / 2^53; Stapel: (k + 0,5) / 2^53 (liegt in (0, 1)). Beide stammen aus denselben oberen 53 Bit."""
    r = vwa_rng.SplitMix64(5)
    ref = np.array([r.random() for _ in range(500)])
    u = N.Stream(5).uniform(500)
    assert np.all(np.abs(u - ref) <= 2.5e-16)   # halbes Bit (5,6e-17) plus Rundung von (k + 0,5) auf double
    assert u.min() > 0 and u.max() < 1


def test_strom_ist_fortlaufend_und_reproduzierbar():
    a = N.Stream(9)
    x1, x2 = a.normal(50), a.normal(50)
    b = N.Stream(9)
    both = b.normal(100)
    assert np.array_equal(np.concatenate([x1, x2]), both)
    assert np.array_equal(N.Stream(9).normal(100), both)
    assert not np.array_equal(N.Stream(10).normal(100), both)


def test_normalzahlen_gleich_inv_cdf_auf_1e_9():
    u = N.Stream(3).uniform(1000)
    nd = NormalDist()
    ref = np.array([nd.inv_cdf(float(v)) for v in u])
    z = N.Stream(3).normal(1000)
    assert np.max(np.abs(z - ref)) < 1e-9


def test_normalzahlen_im_band():
    """200 000 Zahlen: Mittel, Standardabweichung, Anteil |x| > 3 (0,0026998) und Schiefe liegen im Band."""
    z = N.Stream(7).normal(200_000)
    assert abs(z.mean()) < 4 / math.sqrt(z.size)
    assert abs(z.std() - 1) < 0.01
    p3 = (np.abs(z) > 3).mean()
    assert abs(p3 - 0.0026998) < 4 * math.sqrt(0.0027 / z.size)
    assert abs(((z - z.mean()) ** 3).mean()) < 0.03


def test_rng_datei_ist_die_vorlage_unveraendert():
    vorlage = Path(__file__).resolve().parents[2] / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
    if not vorlage.exists():
        pytest.skip("Vorlage liegt nicht neben der Demo")
    assert (Path(vwa_rng.__file__).read_bytes() == vorlage.read_bytes())
