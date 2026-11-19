"""Presets: Vollständigkeit, gültige Werte, Voreinstellung = erstes Preset, jedes Preset einmalig; Permalink (Begrenzen, Rundlauf, ungültige Werte)."""

import pytest

import vwa_constants as C
import vwa_presets as P
import vwa_wegweiser as W


def test_jedes_preset_hat_hilfetext_und_nur_bekannte_regler():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and all(C.PRESET_HELP[n] for n in C.PRESETS)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.SETTING_SPECS), name


def test_werte_der_presets_liegen_im_gueltigen_bereich():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            assert not P._bad(spec, value), (name, key, value)
            if spec.lo is not None:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_erstes_preset_ist_die_voreinstellung_und_jedes_preset_ist_einmalig():
    first = next(iter(C.PRESETS))
    assert C.PRESETS[first] == {}
    full = {n: tuple(sorted(((k, p.get(k, s.default)) for k, s in P.SETTING_SPECS.items()), key=lambda kv: kv[0])) for n, p in C.PRESETS.items()}
    assert len(set(full.values())) == len(full)          # kein Preset gleicht einem anderen: die Hervorhebung ist eindeutig


def test_jedes_preset_ausser_dem_ersten_aendert_mindestens_einen_regler():
    for name, preset in list(C.PRESETS.items())[1:]:
        assert any(v != P.SETTING_SPECS[k].default for k, v in preset.items()), name


def test_regler_kennungen_der_adresszeile_sind_eindeutig_und_jeder_regler_hat_eine_gruppe():
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))
    assert all(s.group in P.GROUP_HEADERS and s.label for s in P.SETTING_SPECS.values())
    assert set(P.GROUP_HEADERS) == {"exp1", "exp2", "exp3", "exp4", "exp5"}


def test_optionen_des_wegweisers_stimmen_mit_den_reglern_ueberein():
    assert P.SETTING_SPECS["w_abh"].options == W.ABHAENGIGKEIT and P.SETTING_SPECS["w_kanaele"].options == W.KANAELE and P.SETTING_SPECS["w_sprung"].options == W.SPRUENGE
    assert P.SETTING_SPECS["w_m"].options == W.PHASE1_M and P.SETTING_SPECS["m_phase1"].options == C.PHASE1_M


@pytest.mark.parametrize("key,raw,expected", [("delta", "2", 2.0), ("mass", "Pollak", "Pollak"), ("p", "7", 7), ("phi", "0.35", 0.35), ("cand_combo", "1", True), ("cand_shewhart", "0", False), ("m_phase1", "300", 300)])
def test_permalink_rundlauf_und_umwandlung(key, raw, expected):
    spec = P.SETTING_SPECS[key]
    assert P._cast(spec, raw) == expected
    assert P._cast(spec, P._write(spec, expected)) == expected


def test_permalink_ungueltige_werte_werfen_fehler_der_nur_abgefangen_wird():
    for key, raw in (("delta", "9"), ("mass", "irgendwas"), ("p", "abc")):
        with pytest.raises(ValueError):
            P._cast(P.SETTING_SPECS[key], raw)
    assert P._bad(P.SETTING_SPECS["delta"], 9.0) and not P._bad(P.SETTING_SPECS["delta"], 0.25)
