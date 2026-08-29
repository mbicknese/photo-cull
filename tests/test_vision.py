import pytest

from photo_cull.vision import parse_burst_comparison, parse_individual_analysis

VALID_JSON = """
{
  "composition": 82,
  "exposure": 78,
  "sharpness": 91,
  "moment": 86,
  "potential": 87,
  "confidence": 92,
  "primary_strength": "Strong interaction between subjects",
  "primary_problem": "Background is slightly distracting",
  "fixable_issues": ["Crop slightly from the left"],
  "nonfixable_issues": [],
  "explanation": "Strong capture with good expressions and accurate focus."
}
"""


def test_parse_individual_analysis_valid() -> None:
    result = parse_individual_analysis(VALID_JSON)
    assert result.composition == 82
    assert result.potential == 87
    assert result.fixable_issues == ["Crop slightly from the left"]
    assert result.nonfixable_issues == []


def test_parse_individual_analysis_with_surrounding_text() -> None:
    text = f"Sure, here is the JSON:\n{VALID_JSON}\nLet me know if you need anything else."
    result = parse_individual_analysis(text)
    assert result.potential == 87


def test_parse_individual_analysis_invalid_json_raises() -> None:
    with pytest.raises(ValueError):
        parse_individual_analysis("this is not json at all")


def test_parse_individual_analysis_missing_field_raises() -> None:
    bad = '{"composition": 80, "exposure": 70, "sharpness": 90, "moment": 85, "confidence": 90}'
    with pytest.raises(ValueError):
        parse_individual_analysis(bad)


def test_parse_individual_analysis_value_out_of_range_raises() -> None:
    bad = (
        '{"composition": 180, "exposure": 70, "sharpness": 90, "moment": 85, '
        '"potential": 80, "confidence": 90}'
    )
    with pytest.raises(ValueError):
        parse_individual_analysis(bad)


def test_parse_individual_analysis_negative_value_raises() -> None:
    bad = (
        '{"composition": -5, "exposure": 70, "sharpness": 90, "moment": 85, '
        '"potential": 80, "confidence": 90}'
    )
    with pytest.raises(ValueError):
        parse_individual_analysis(bad)


def test_parse_burst_comparison_valid() -> None:
    text = """
    {
      "ranking": [
        {"label": "DSCF1001", "rank": 2, "tier": "close_second", "notes": "good"},
        {"label": "DSCF1002", "rank": 1, "tier": "clear_winner", "notes": "best expression"}
      ]
    }
    """
    entries = parse_burst_comparison(text, ["DSCF1001", "DSCF1002"])
    by_label = {e["label"]: e for e in entries}
    assert by_label["DSCF1002"]["tier"] == "clear_winner"
    assert by_label["DSCF1001"]["rank"] == 2


def test_parse_burst_comparison_missing_label_raises() -> None:
    text = """
    {"ranking": [{"label": "DSCF1001", "rank": 1, "tier": "clear_winner", "notes": ""}]}
    """
    with pytest.raises(ValueError):
        parse_burst_comparison(text, ["DSCF1001", "DSCF1002"])


def test_parse_burst_comparison_invalid_tier_raises() -> None:
    text = """
    {"ranking": [{"label": "DSCF1001", "rank": 1, "tier": "amazing", "notes": ""}]}
    """
    with pytest.raises(ValueError):
        parse_burst_comparison(text, ["DSCF1001"])
