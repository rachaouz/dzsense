from src.data.preprocessing import normalize


def test_removes_urls_and_mentions():
    assert normalize("بارك الله http://x.com @user") == "بارك الله"


def test_collapses_elongation():
    assert normalize("جمييييل") == "جمييل"


def test_unifies_alef_and_strips_diacritics():
    assert normalize("أَحْسَن") == "احسن"


def test_empty_stays_empty():
    assert normalize("  ") == ""