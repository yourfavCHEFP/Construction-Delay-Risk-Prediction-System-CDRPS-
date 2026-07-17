import pytest
from CDRPS.src.ingestion.file_loader import load_file

def test_load_file_scenarios(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    assert len(load_file(str(csv))) == 2
    
    with pytest.raises(ValueError):
        bad = tmp_path / "bad.txt"
        bad.write_text("hi")
        load_file(str(bad))