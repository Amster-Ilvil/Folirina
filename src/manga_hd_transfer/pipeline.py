from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .bubbles import build_text_units, detect_seeded_white_bubbles, load_bubble_sidecar
from .config import PipelineConfig
from .debug import mask_overlay, matching_overlay, registration_overlay, structure_overlay
from .export import export_openraster, export_psd_imagemagick, make_text_layer_rgba, write_rgba
from .inpainting import inpaint_image
from .io_utils import read_image, save_json, stem_id, write_image
from .lettering import composite_text, fit_text, polygon_safe_mask
from .masking import build_clear_mask
from .matching import match_units
from .models import BookProject, BubbleInstance, PagePair, PageProject, TextUnit, UnitMatch
from .ocr import OCRBackend, RetryingOCRBackend, build_backend
from .pairing import pair_directories
from .qa import qa_summary, run_page_qa
from .registration import register_images

logger = logging.getLogger(__name__)


class TransferPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        source_ocr: OCRBackend | None = None,
        target_ocr: OCRBackend | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._source_ocr = source_ocr
        self._target_ocr = target_ocr

    @property
    def source_ocr(self) -> OCRBackend:
        if self._source_ocr is None:
            backend_name = self.config.ocr.source_backend or self.config.ocr.backend
            backend = build_backend(self.config.ocr, self.config.ocr.source_lang, backend_name)
            if self.config.ocr.retry_low_confidence and backend_name == "paddle":
                backend = RetryingOCRBackend(backend, self.config.ocr.retry_confidence, self.config.ocr.retry_scale)
            self._source_ocr = backend
        return self._source_ocr

    @property
    def target_ocr(self) -> OCRBackend:
        if self._target_ocr is None:
            backend_name = self.config.ocr.target_backend or self.config.ocr.backend
            self._target_ocr = build_backend(self.config.ocr, self.config.ocr.target_lang, backend_name)
        return self._target_ocr

    def _bubbles(self, image: np.ndarray, blocks, image_path: str | Path) -> list[BubbleInstance]:
        backend = self.config.bubbles.backend.lower()
        if backend == "none":
            return []
        if backend == "seeded_white":
            return detect_seeded_white_bubbles(image, blocks, self.config.bubbles)
        if backend == "sidecar":
            return load_bubble_sidecar(image, image_path, blocks, self.config.bubbles)
        raise ValueError(f"Unsupported bubble backend: {self.config.bubbles.backend}")

    def _accepted_matches(
        self,
        pair: PagePair,
        registration_confidence: float,
        source_units: list[TextUnit],
        target_units: list[TextUnit],
        matches: list[UnitMatch],
    ) -> list[UnitMatch]:
        if pair.confidence < self.config.pairing.confidence_floor:
            return []
        if registration_confidence < self.config.qa.registration_min_confidence:
            return []
        su = {u.id: u for u in source_units}
        tu = {u.id: u for u in target_units}
        accepted = []
        for match in matches:
            s, t = su.get(match.source_unit_id), tu.get(match.target_unit_id)
            if s is None or t is None:
                continue
            if match.relation != "one_to_one":
                continue
            if match.confidence < self.config.matching.review_confidence:
                continue
            if s.confidence < self.config.ocr.min_confidence or t.confidence < self.config.ocr.min_confidence:
                continue
            if s.kind not in self.config.matching.auto_apply_kinds or t.kind not in self.config.matching.auto_apply_kinds:
                continue
            if not s.text.strip():
                continue
            accepted.append(match)
        return accepted

    def process_page(
        self,
        pair: PagePair,
        page_root: str | Path,
        final_path: str | Path | None = None,
    ) -> PageProject:
        page_root = Path(page_root)
        page_root.mkdir(parents=True, exist_ok=True)
        source = read_image(pair.source_path)
        target = read_image(pair.target_path)

        registration = register_images(source, target, self.config.registration)
        source_blocks = self.source_ocr.recognize(source, image_path=pair.source_path)
        target_blocks = self.target_ocr.recognize(target, image_path=pair.target_path)

        source_bubbles = self._bubbles(source, source_blocks, pair.source_path)
        target_bubbles = self._bubbles(target, target_blocks, pair.target_path)
        source_units = build_text_units(source_blocks, source_bubbles, "src")
        target_units = build_text_units(target_blocks, target_bubbles, "dst")
        match_result = match_units(source_units, target_units, registration, self.config.matching)
        matches = match_result.matches
        accepted = self._accepted_matches(pair, registration.confidence, source_units, target_units, matches)

        mask_result = build_clear_mask(
            target.shape[:2],
            target_blocks,
            target_units,
            target_bubbles,
            accepted,
            self.config.masking,
            min_match_confidence=self.config.matching.review_confidence,
            allow_relations={"one_to_one"},
        )
        inpaint_result = inpaint_image(target, mask_result.mask, self.config.inpainting)
        rendered = inpaint_result.image.copy()

        source_by_id = {u.id: u for u in source_units}
        target_by_id = {u.id: u for u in target_units}
        bubbles_by_id = {b.id: b for b in target_bubbles}
        lettering = []
        lettering_masks: list[np.ndarray] = []
        for match in accepted:
            src = source_by_id[match.source_unit_id]
            dst = target_by_id[match.target_unit_id]
            safe = None
            if dst.bubble_id and dst.bubble_id in bubbles_by_id:
                safe = bubbles_by_id[dst.bubble_id].safe_mask
            if safe is None or cv2.countNonZero(safe) == 0:
                safe = polygon_safe_mask(dst, target.shape[:2], margin=max(2, self.config.bubbles.safe_margin_px // 2))
            result = fit_text(target.shape[:2], safe, dst, src.text, self.config.lettering)
            lettering.append(result)
            if result.success and result.text_mask is not None:
                rendered = composite_text(rendered, result, self.config.lettering)
                lettering_masks.append(result.text_mask)

        qa = run_page_qa(
            pair,
            registration,
            source_units,
            target_units,
            matches,
            lettering,
            mask_result,
            inpaint_result.image,
            self.config.qa,
        )

        page_id = stem_id(pair.target_path)
        project = PageProject(
            page_id=page_id,
            pair=pair,
            registration=registration,
            source_blocks=source_blocks,
            target_blocks=target_blocks,
            source_bubbles=source_bubbles,
            target_bubbles=target_bubbles,
            source_units=source_units,
            target_units=target_units,
            matches=matches,
            lettering=lettering,
            qa=qa,
            meta={
                "auto_applied_match_ids": [f"{m.source_unit_id}->{m.target_unit_id}" for m in accepted],
                "auto_applied_count": len(accepted),
                "inpainting": {"method": inpaint_result.method, **inpaint_result.diagnostics},
                "mask_clipped_ratio": mask_result.clipped_ratio,
                "qa_summary": qa_summary(qa),
                "unmatched_source_units": match_result.unmatched_source,
                "unmatched_target_units": match_result.unmatched_target,
                "ambiguous_source_units": match_result.ambiguous_source,
            },
        )

        # Artifacts are intentionally explicit and lossless.
        original_path = page_root / "target_original.png"
        final_local = page_root / "final.png"
        inpainted_path = page_root / "inpainted.png"
        clear_mask_path = page_root / "clear_mask.png"
        text_layer_path = page_root / "text_layer.png"
        write_image(original_path, target)
        write_image(final_local, rendered)
        if self.config.export.save_inpainted:
            write_image(inpainted_path, inpaint_result.image)
        if self.config.export.save_masks:
            write_image(clear_mask_path, mask_result.mask)
            for unit_id, mask in mask_result.per_unit.items():
                write_image(page_root / "masks" / f"{unit_id}.png", mask)
            for bubble in target_bubbles:
                if bubble.mask is not None:
                    write_image(page_root / "bubbles" / f"{bubble.id}.png", bubble.mask)
                if bubble.safe_mask is not None:
                    write_image(page_root / "bubbles" / f"{bubble.id}_safe.png", bubble.safe_mask)

        text_rgba = make_text_layer_rgba(target.shape[:2], lettering_masks, color=self.config.lettering.fill)
        write_rgba(text_layer_path, text_rgba)
        if self.config.export.layer_bundle:
            ora_path = page_root / "editable.ora"
            export_openraster(ora_path, target, inpaint_result.image, text_rgba)
            psd_path = page_root / "editable.psd"
            psd_ok = export_psd_imagemagick(psd_path, original_path, inpainted_path, text_layer_path) if inpainted_path.exists() else False
            project.meta["psd_exported"] = psd_ok

        if self.config.export.save_debug:
            write_image(page_root / "debug_registration.png", registration_overlay(source, target, registration))
            write_image(page_root / "debug_structure.png", structure_overlay(target, target_units, target_bubbles))
            write_image(page_root / "debug_matching.png", matching_overlay(target, source_units, target_units, matches, registration))
            write_image(page_root / "debug_clear_mask.png", mask_overlay(target, mask_result.mask))

        if final_path is not None:
            write_image(final_path, rendered)
            project.artifacts["book_final"] = str(Path(final_path))
        project.artifacts.update(
            {
                "target_original": str(original_path),
                "final": str(final_local),
                "inpainted": str(inpainted_path) if inpainted_path.exists() else "",
                "clear_mask": str(clear_mask_path) if clear_mask_path.exists() else "",
                "text_layer": str(text_layer_path),
                "openraster": str(page_root / "editable.ora") if (page_root / "editable.ora").exists() else "",
                "psd": str(page_root / "editable.psd") if (page_root / "editable.psd").exists() else "",
            }
        )
        save_json(page_root / "qa.json", {"summary": qa_summary(qa), "issues": [x.to_dict() for x in qa]})
        save_json(page_root / "project.json", project.to_dict())
        return project

    def run_book(self, source_dir: str | Path, target_dir: str | Path, output_dir: str | Path) -> BookProject:
        output = Path(output_dir)
        pages_root = output / "pages"
        final_root = output / "final"
        pages_root.mkdir(parents=True, exist_ok=True)
        final_root.mkdir(parents=True, exist_ok=True)

        pairs, unmatched_source, unmatched_target = pair_directories(source_dir, target_dir, self.config.pairing)
        pages: list[PageProject] = []
        for idx, pair in enumerate(pairs, start=1):
            target_name = Path(pair.target_path).stem + ".png"
            page_dir = pages_root / f"{idx:04d}_{stem_id(pair.target_path)}"
            logger.info("Processing page %d/%d: %s", idx, len(pairs), Path(pair.target_path).name)
            page = self.process_page(pair, page_dir, final_root / target_name)
            pages.append(page)

        book = BookProject(
            source_dir=str(source_dir),
            target_dir=str(target_dir),
            output_dir=str(output),
            pages=pages,
            unmatched_source=unmatched_source,
            unmatched_target=unmatched_target,
            meta={
                "page_count": len(pages),
                "qa_errors": sum(1 for p in pages for q in p.qa if q.severity == "error"),
                "qa_warnings": sum(1 for p in pages for q in p.qa if q.severity == "warning"),
            },
        )
        save_json(output / "book_project.json", book.to_dict())
        save_json(
            output / "qa_summary.json",
            {
                "pages": [
                    {"page_id": p.page_id, "summary": qa_summary(p.qa), "project": p.artifacts.get("final", "")}
                    for p in pages
                ],
                "unmatched_source": unmatched_source,
                "unmatched_target": unmatched_target,
            },
        )
        return book
