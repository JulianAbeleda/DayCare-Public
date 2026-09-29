from daycare.artifact.record_xml import read, write


def test_xml_record_round_trips_nested_typed_data(tmp_path):
    value = {"name": "trial", "ok": True, "count": 3, "score": 0.25,
             "missing": None, "rows": [{"text": "<tool>&done", "reward": 1.0}]}
    path = tmp_path / "summary.xml"
    write(path, value, root="summary")
    assert read(path) == value
    assert path.read_text().startswith("<?xml")


def test_xml_record_round_trips_xml_illegal_characters(tmp_path):
    value = {"text": "\boxed{7} \x08 \x1f \ud800", "plain": "ok"}  # run 5 update-101: a sampled backspace
    path = tmp_path / "update.xml"
    write(path, value, root="update")
    assert read(path) == value


def test_xml_record_writes_numpy_floats_readably(tmp_path):
    import numpy as np
    path = tmp_path / "update.xml"
    write(path, {"rewards": [np.float64(1.5), np.float64(-2.0) + 0.1]}, root="update")
    assert "np.float64" not in path.read_text()
    assert read(path) == {"rewards": [1.5, -1.9]}

