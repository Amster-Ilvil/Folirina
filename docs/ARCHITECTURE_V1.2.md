# Manga-HD-Translation-Transfer v1.2 architecture

## Goal

v1.2 keeps the established Direct / Mask / Reveal behavior, but removes duplicated state ownership. The rule is simple: modules that transform pixels stay independent; modules that share page/review state use one contract.

## Independent layers

### `schema_compat.py`
Boundary for historical JSON compatibility. `bool / dict / list` legacy shapes are normalized here before GUI, Web Review, Workspace or review-apply code consumes them.

### `result_state.py`
Single owner of result lifecycle:

- current result resolution (`final.png` / `final_reviewed.png`)
- immutable manual baseline (`final_auto.png`)
- reviewed-result atomic commit
- `review_sync.json`
- `project.json` artifact synchronization
- invalidation on a fresh automatic process

No Qt dependency.

### `manual_review_service.py`
Single transaction for one manual omission repair:

1. normalize review JSON
2. freeze stable baseline
3. save Reveal mask/patch
4. update overrides without duplicating row ids
5. call review compositor
6. verify `final == final_reviewed`
7. verify Reveal patch is pixel-exact in final output

No Qt dependency. GUI only gathers user input and invokes this service.

### Pixel algorithms

`manual_effect.py`, `text_only_transfer.py`, `direct_containers.py`, `mask_transfer.py`, registration and detection modules remain pixel/geometry focused. They do not own GUI result state.

## Linked layers

### Qt workbench

`gui_qt.py` owns interaction only:

- opens selection / Reveal editors
- supplies callback into `manual_review_service.commit_manual_effect`
- refreshes UI after a verified transaction

It does not implement a second copy of result synchronization or manual-baseline logic.

### Web Review

`review.py` uses the same normalized schema and result resolver. Partial web edits are merged non-destructively so they cannot erase Qt `manual_effect_regions`.

### Review compositor

`review_apply.py` renders pixels, then calls the shared `result_state.commit_reviewed_result` contract. Compatibility wrapper function names are retained for old tests/plugins.

### Workspace

`workspace.py` resolves the visible result through `result_state.resolve_result_state`; it does not maintain a separate newest-file policy.

### Pipeline

A fresh automatic page process calls `result_state.invalidate_manual_review_state`, preventing stale manual baselines from surviving a reprocess.

## Invariants

1. SOURCE background RGB never becomes the authority for a color TARGET.
2. `final_auto.png` is immutable during one manual-edit session.
3. `final.png` and `final_reviewed.png` are byte-equivalent after a successful review commit.
4. A Reveal transaction cannot report success with an empty or non-pixel-exact patch.
5. Web Review may edit its own fields but must not delete Qt manual regions.
6. Historical bool/dict/list JSON shapes are normalized at boundaries, not handled ad hoc deep inside pixel code.
7. A fresh automatic process invalidates manual session state before generating new output.
8. Runtime version has one Python source (`version.py`); GUI and package import it, and package metadata matches it.
