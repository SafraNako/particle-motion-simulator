from __future__ import annotations

from parcacik_hareketi_simulatoru.cli import main


def test_siklotron_smoke(capsys, tmp_path):
    png = tmp_path / "s.png"
    assert main(["siklotron", "--periyot-sayisi", "20", "--grafik", str(png)]) == 0
    out = capsys.readouterr().out
    assert "jirasyon yarıçapı" in out
    assert "Boris:" in out and "RK4  :" in out
    assert png.exists() and png.stat().st_size > 0


def test_siklotron_reports_boris_angle_matching_prediction(capsys):
    main(["siklotron", "--periyot-sayisi", "5"])
    line = next(l for l in capsys.readouterr().out.splitlines() if "adım başına jirasyon açısı" in l)
    measured = line.split("ölçülen ")[1].split(" rad")[0]
    predicted = line.split("= ")[1].split(" rad")[0]
    assert measured == predicted


def test_capraz_alanlar_smoke(capsys, tmp_path):
    png = tmp_path / "c.png"
    assert main(["capraz-alanlar", "--periyot-sayisi", "10", "--grafik", str(png)]) == 0
    out = capsys.readouterr().out
    assert "sürüklenme hızı" in out
    assert "kat azalıyor" in out
    assert png.exists()


def test_capraz_alanlar_rejects_superluminal_ratio(capsys):
    assert main(["capraz-alanlar", "--e-orani", "1.2"]) == 1
    assert "Hata" in capsys.readouterr().out


def test_hiperbolik_hareket_smoke(capsys, tmp_path):
    png = tmp_path / "h.png"
    assert main(["hiperbolik-hareket", "--grafik", str(png)]) == 0
    out = capsys.readouterr().out
    assert "ışık hızını" in out
    assert png.exists()


def test_manyetik_ayna_smoke(capsys, tmp_path):
    png = tmp_path / "m.png"
    assert main(["manyetik-ayna", "--sicrama-sayisi", "1", "--grafik", str(png)]) == 0
    out = capsys.readouterr().out
    assert "dönüm noktası" in out
    assert png.exists()


def test_manyetik_ayna_rejects_bad_pitch(capsys):
    assert main(["manyetik-ayna", "--pitch", "95"]) == 1
    assert "Hata" in capsys.readouterr().out


def test_yakinsama_smoke(capsys, tmp_path):
    png = tmp_path / "y.png"
    assert main(["yakinsama", "--grafik", str(png)]) == 0
    out = capsys.readouterr().out
    assert "gözlenen yakınsama mertebesi" in out
    assert png.exists()


def test_nonpositive_inputs_are_rejected(capsys):
    assert main(["siklotron", "--kinetik", "-1"]) == 1
    assert "Hata" in capsys.readouterr().out
