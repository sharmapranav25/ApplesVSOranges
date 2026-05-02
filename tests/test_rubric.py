"""Rubric files: must exist, parse to ascending 0->5, contain 6 levels each.

Catches the silent re-order bug where someone copies the paper's descending
text into the prompt files (would corrupt every judge call).
"""

from rubric import load_judge_template, load_rubric


def test_accuracy_rubric_loads_ascending():
    text = load_rubric("accuracy")
    lines = [l for l in text.strip().split("\n") if l.strip()]
    assert len(lines) == 6
    for i, line in enumerate(lines):
        # First token should be the level number.
        assert line.lstrip().startswith(f"{i} "), f"line {i} starts with {line[:20]!r}"
    # Spot-check key phrases verbatim from §4.3.
    assert "entirely incorrect" in lines[0]
    assert "fully accurate" in lines[5]


def test_completeness_rubric_loads_ascending():
    text = load_rubric("completeness")
    lines = [l for l in text.strip().split("\n") if l.strip()]
    assert len(lines) == 6
    for i, line in enumerate(lines):
        assert line.lstrip().startswith(f"{i} ")
    assert "superficial or irrelevant" in lines[0]
    assert "comprehensive" in lines[5]


def test_judge_template_has_required_placeholders():
    t = load_judge_template()
    for ph in ("{criteria}", "{scoring_criteria}", "{joke}",
               "{reference_explanation}", "{model_explanation}"):
        assert ph in t, f"missing placeholder {ph}"
    # Verbatim sentence from Appendix A.6.
    assert "You must only respond with a single number between 0 and 5" in t
    assert "You must produce no other output." in t
