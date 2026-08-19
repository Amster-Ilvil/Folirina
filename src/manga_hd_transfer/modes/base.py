from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ModeContract:
    name: str
    orchestrator: bool = False
    direct: bool = False
    mask_replace: bool = False
    reletter: bool = False
    transparent_reveal: bool = False
    aligned_reveal: bool = False
    may_use_ocr: bool = False
    may_use_presence_ocr: bool = False
    may_render_text: bool = False
    may_use_mask_pixels: bool = False
    may_fallback_to_mask: bool = False
    may_fallback_to_reletter: bool = False
    manual_reletter_editor: bool = False

    @property
    def explicit_isolated_route(self) -> bool:
        return bool(self.transparent_reveal or self.aligned_reveal)

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ModeSpec:
    key: str
    label: str
    contract: ModeContract
    config_root: str
    visible: bool = True
    legacy: bool = False
    owned_artifacts: tuple[str, ...] = ()
    owned_paths: tuple[str, ...] = ()
    workflow: tuple[str, ...] = ()
    fallback_policy: str = "none"
    ui_defaults: dict = field(default_factory=dict)

    def to_manifest(self) -> dict:
        return {
            "key": self.key, "label": self.label, "config_root": self.config_root,
            "visible": self.visible, "legacy": self.legacy,
            "contract": self.contract.to_dict(),
            "owned_artifacts": list(self.owned_artifacts),
            "owned_paths": list(self.owned_paths),
            "workflow": list(self.workflow),
            "fallback_policy": self.fallback_policy,
            "ui_defaults": dict(self.ui_defaults),
        }
