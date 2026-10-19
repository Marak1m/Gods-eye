# third_party/ard — companion project's preprocessing package (not yet received)

HANDOFF.md section 2.3: a git subtree of the companion project's `ard/` package and
`check_pair.py`. Never edit in place; `scd/` wraps it (`scd/geo/coreg.py` for AROSICS local
co-registration, `tests/test_cloud_parity.py` for the OmniCloudMask parity check).

When the repository access from the plan doc section 17 checklist arrives:

```bash
git subtree add --prefix third_party/ard <companion-repo-url> <branch> --squash
```

Until then everything that depends on it is marked `@pytest.mark.gpu` or raises
`NotImplementedError` pointing here.
