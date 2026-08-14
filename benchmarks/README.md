# Publication Benchmark Dataset Convention

Do **not** commit copyrighted private book pages to this repository. Point `scripts/publication_gate.py` at a local/private benchmark root.

Each work:

```text
benchmark-root/
  work_id/
    primary_cn/
    target_jp/
    secondary_cn_hd/      # optional
    golden/               # optional manual final/mask annotations
    labels.json
```

Minimal `labels.json`:

```json
{
  "schema": "manga-hd-transfer/benchmark-labels/v1",
  "work_id": "example",
  "primary_dir": "primary_cn",
  "target_dir": "target_jp",
  "pages": [
    {
      "id": "001",
      "primary": "primary_cn/001.png",
      "target": "target_jp/001.png",
      "secondary": "secondary_cn_hd/001.png",
      "tags": ["same_size", "vertical_text"],
      "expected_unit_matches": []
    }
  ]
}
```

`expected_unit_matches` is optional until stable sidecar unit IDs are manually golden-labeled. Pairing, residual, border-damage, safe-area, review rate and timing can still be measured without it.
