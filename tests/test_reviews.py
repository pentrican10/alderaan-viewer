import pytest

from alderaan_viewer import reviews


def test_latest_status_wins(tmp_path):
    assert reviews.read_statuses(tmp_path) == {}
    reviews.append_status(tmp_path, "K00001", "Minor Issues", "a")
    reviews.append_status(tmp_path, "K00002", "Looks good", "a")
    reviews.append_status(tmp_path, "K00001", "Looks good", "b")
    assert (tmp_path / "review_status.csv").is_file()  # beside the batch catalog
    assert reviews.read_statuses(tmp_path) == {
        "K00001": "Looks good",
        "K00002": "Looks good",
    }


def test_unknown_status_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown review status"):
        reviews.append_status(tmp_path, "K00001", "fine", "a")


def test_comments_append_and_replace(tmp_path):
    assert reviews.read_comments(tmp_path, "K00001") == ""
    reviews.append_comment(tmp_path, "K00001", "first", "a")
    reviews.append_comment(tmp_path, "K00001", "second", "b")
    text = reviews.read_comments(tmp_path, "K00001")
    assert "User: a\nComment: first" in text
    assert "User: b\nComment: second" in text

    reviews.replace_comments(tmp_path, "K00001", "edited")
    assert reviews.read_comments(tmp_path, "K00001") == "edited"
