from app.core.security import token_is_valid


def test_admin_token_requires_both_configured_and_supplied_values():
    assert not token_is_valid(None, "configured-secret")
    assert not token_is_valid("provided-secret", None)
    assert not token_is_valid("provided-secret", "")


def test_admin_token_rejects_mismatch_and_accepts_match():
    assert not token_is_valid("wrong", "configured-secret")
    assert token_is_valid("configured-secret", "configured-secret")
