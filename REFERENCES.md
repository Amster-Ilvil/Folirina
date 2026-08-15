# References and Acknowledgements

This document separates **implementation lineage**, **development research references**, and **academic citations** used by Manga HD Transfer Studio.

A reference does not by itself mean that upstream source code is copied into this repository. For licensing details, see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Implementation / design lineage

### Novel Formatter

- Repository: https://github.com/Amster-Ilvil/Novel-formatter
- License: MIT
- Relevance: Apple OCR architecture and the privacy-safe, source-first multi-platform bootstrap/release approach.
- The Apple OCR relationship is explicitly documented in `src/manga_hd_transfer/apple_live_text.py`.

### KCC-Kindle-CHS

- Repository: https://github.com/Amster-Ilvil/KCC-Kindle-CHS
- License: ISC
- Relevance: UI/macOS palette inspiration. The relationship is explicitly documented in `src/manga_hd_transfer/gui_qt.py`.

## Development research references

These projects were consulted as workflow/architecture references while developing manga text detection, cleaning, inpainting, rendering and manual-review behavior. They are not declared runtime dependencies merely by being listed here.

### manga-image-translator

- Repository: https://github.com/zyddnys/manga-image-translator
- License: GPL-3.0
- Relevance: manga/image translation pipeline research, including detection, OCR, inpainting and rendering decomposition.

### comic-translate

- Repository: https://github.com/ogkalu2/comic-translate
- License: Apache-2.0
- Relevance: comic translation workflow research, including automatic/manual processing, detection, cleaning and rendering stages.

## Academic citations

### LightGlue

Manga HD Transfer can optionally use LightGlue for learned local feature matching during page registration.

Official paper citation:

```bibtex
@InProceedings{Lindenberger_2023_ICCV,
  author    = {Lindenberger, Philipp and Sarlin, Paul-Edouard and Pollefeys, Marc},
  title     = {LightGlue: Local Feature Matching at Light Speed},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2023},
  pages     = {17627-17638}
}
```

Paper: https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html

Code: https://github.com/cvg/LightGlue

### LoFTR

Manga HD Transfer can optionally use Kornia's LoFTR implementation as a registration matcher.

Official paper citation:

```bibtex
@InProceedings{Sun_2021_CVPR,
  author    = {Sun, Jiaming and Shen, Zehong and Wang, Yuang and Bao, Hujun and Zhou, Xiaowei},
  title     = {LoFTR: Detector-Free Local Feature Matching With Transformers},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2021},
  pages     = {8922-8931}
}
```

Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html

### RT-DETR

Manga HD Transfer exposes an optional RT-DETR path through the Transformers/PyTorch stack.

Official CVPR citation:

```bibtex
@InProceedings{Zhao_2024_CVPR,
  author    = {Zhao, Yian and Lv, Wenyu and Xu, Shangliang and Wei, Jinman and Wang, Guanzhong and Dang, Qingqing and Liu, Yi and Chen, Jie},
  title     = {DETRs Beat YOLOs on Real-time Object Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024},
  pages     = {16965-16974}
}
```

Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html

### PaddleOCR 3.0

Manga HD Transfer supports PaddleOCR as an optional OCR backend.

Upstream's PaddleOCR 3.0 technical report citation:

```bibtex
@misc{cui2025paddleocr30technicalreport,
  title        = {PaddleOCR 3.0 Technical Report},
  author       = {Cheng Cui and Ting Sun and Manhui Lin and Tingquan Gao and Yubo Zhang and Jiaxuan Liu and Xueqing Wang and Zelun Zhang and Changda Zhou and Hongen Liu and Yue Zhang and Wenyu Lv and Kui Huang and Yichao Zhang and Jing Zhang and Jun Zhang and Yi Liu and Dianhai Yu and Yanjun Ma},
  year         = {2025},
  eprint       = {2507.05595},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV}
}
```

Report: https://arxiv.org/abs/2507.05595

Code: https://github.com/PaddlePaddle/PaddleOCR

## Citing Manga HD Transfer Studio

GitHub-compatible citation metadata is provided in [`CITATION.cff`](CITATION.cff). If you publish research or a technical report that materially relies on this project, cite the repository/version you used and, where relevant, also cite the underlying algorithm papers above.
