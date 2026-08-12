# Acknowledgements / 致谢

Manga HD Translation Transfer is informed by the work of the manga translation, OCR, image-registration, segmentation, inpainting, and typesetting communities.

特别感谢以下项目及其维护者提供的公开研究、实现与工程思路：

- `hgmzhn/manga-translator-ui` — replace-translation workflow and dual-image translation-transfer ideas.
- `zyddnys/manga-image-translator` — detection → OCR → mask → inpaint → render pipeline reference.
- `dmMaze/BallonsTranslator` and `thomaswantstobeaskeleton/BallonsTranslator-Pro` — assisted editing, mask repair, lettering, safe-area, and QA workflow ideas.
- `dmMaze/comic-text-detector` — manga/comic text detection and segmentation reference.
- `ogkalu2/comic-translate` — bubble detection, OCR routing, and inpainting architecture reference.
- PaddleOCR — Chinese/general OCR candidate and layout-recognition ecosystem.
- LightGlue and LoFTR — cross-version image matching and registration references.
- LaMa — complex-region inpainting reference.
- SAM 2 — interactive segmentation and mask-refinement reference.
- MangaLens bubble segmentation resources — speech-bubble instance segmentation reference.
- ScanR/TypeR — professional lettering and typesetting workflow reference.
- ImageTrans plugins — modular OCR, mask-generation, and inpainting provider ideas.

A more detailed discussion of what is referenced, what may be reused, and what must be independently implemented is maintained in `docs/REFERENCE_PROJECTS.md`.

## License

Original repository-authored code and documentation remain under the repository's MIT License. Any third-party code, models, weights, datasets, or assets incorporated later must retain and comply with their own licenses; acknowledgement here does not relicense them.
