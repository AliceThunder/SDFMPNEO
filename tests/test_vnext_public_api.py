import sdfmpneo_vnext


def test_public_api_exports_only_defined_unique_names():
    exported = tuple(
        sdfmpneo_vnext.__all__
    )
    assert len(exported) == len(
        set(exported)
    )
    missing = [
        name
        for name in exported
        if not hasattr(
            sdfmpneo_vnext,
            name,
        )
    ]
    assert missing == []
