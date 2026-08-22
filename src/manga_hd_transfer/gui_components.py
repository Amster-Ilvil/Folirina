from __future__ import annotations

"""Reusable lightweight Qt widgets.

No project workflow state lives here. Pages can import these primitives without
importing the monolithic StudioWindow module.
"""

from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import Qt, QSize, QTimer, Signal, QRectF, QPoint
from PySide6.QtGui import QPixmap, QPainter, QColor, QImageReader, QIcon
from PySide6.QtWidgets import (
    QApplication, QDialog, QWidget, QFrame, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QRadioButton, QSizePolicy, QGraphicsView, QGraphicsScene, QListWidget, QListWidgetItem, QComboBox, QMenu, QLineEdit, QAbstractItemView,
    QSpinBox, QDoubleSpinBox, QSlider, QAbstractScrollArea,
)

from .gui_theme import MUTED


class StableSpinBox(QSpinBox):
    """Integer editor whose value can never be changed by the mouse wheel.

    Trackpad/wheel gestures are commonly used to scroll Folirina's long inspector
    panels.  Qt's default spin box consumes those gestures and silently changes
    the current value, even when the user only intended to move the panel.  Keep
    explicit keyboard entry and the arrow buttons, but hand wheel gestures back
    to the parent scroll area.
    """

    def wheelEvent(self, event):  # noqa: N802 - Qt API
        event.ignore()


class StableDoubleSpinBox(QDoubleSpinBox):
    """Floating-point counterpart of :class:`StableSpinBox`."""

    def wheelEvent(self, event):  # noqa: N802 - Qt API
        event.ignore()


class StableSlider(QSlider):
    """Drag-only parameter slider; wheel gestures belong to panel scrolling."""

    def wheelEvent(self, event):  # noqa: N802 - Qt API
        event.ignore()


class StableComboBox(QComboBox):
    """QComboBox with a native top-level menu popup.

    The Studio workspace is rendered through a QGraphicsProxyWidget so the whole
    UI can be uniformly scaled on small screens. Qt's stock QComboBox popup is
    normally parented back into that proxy hierarchy; on macOS/Qt 6 it can lose
    focus or be dismissed while the graphics scene processes an unrelated
    update. v2.0.63 did not embed the whole workspace and therefore did not have
    this failure mode.

    Keep the normal QComboBox model/currentIndex API, but render only the popup as
    an independent top-level QMenu. This restores the stable/native interaction
    contract without giving up whole-page scaling.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stable_popup: QMenu | None = None

    def wheelEvent(self, event):  # noqa: N802 - Qt API
        # Scrolling an inspector must never change a mode/model/enum selection.
        # Ignore the wheel so the enclosing scroll area can consume it.
        event.ignore()

    def _popup_anchor_and_width(self) -> tuple[QPoint, int]:
        # QWidget.mapToGlobal() is not consistently useful for children embedded
        # in QGraphicsProxyWidget. Map through the owning scene/view when present.
        try:
            proxy = self.graphicsProxyWidget()
            if proxy is not None and proxy.scene() is not None and proxy.scene().views():
                view = proxy.scene().views()[0]
                rect = proxy.subWidgetRect(self)
                scene_left = proxy.mapToScene(rect.bottomLeft())
                scene_right = proxy.mapToScene(rect.bottomRight())
                vp_left = view.mapFromScene(scene_left)
                vp_right = view.mapFromScene(scene_right)
                global_left = view.viewport().mapToGlobal(vp_left)
                global_right = view.viewport().mapToGlobal(vp_right)
                return global_left, max(1, abs(global_right.x() - global_left.x()))
        except (AttributeError, RuntimeError, TypeError):
            pass
        left = self.mapToGlobal(QPoint(0, self.height()))
        right = self.mapToGlobal(QPoint(self.width(), self.height()))
        return left, max(1, abs(right.x() - left.x()))

    def _highlight_menu_row(self, row: int) -> None:
        row = int(row)
        if 0 <= row < self.count():
            # Match native QComboBox hover signals so callers that preview a
            # highlighted option keep working with the top-level popup.
            self.highlighted.emit(row)
            self.textHighlighted.emit(self.itemText(row))

    def _activate_menu_row(self, row: int) -> None:
        row = int(row)
        if not (0 <= row < self.count()):
            return
        # Native QComboBox emits activated for a user choice even when the user
        # re-selects the current row.  setCurrentIndex alone only provides
        # currentIndexChanged, which broke the manual-region editor's user-lock
        # logic after StableComboBox was introduced in v2.0.72.
        if self.currentIndex() != row:
            self.setCurrentIndex(row)
        self.activated.emit(row)
        self.textActivated.emit(self.itemText(row))

    def showPopup(self) -> None:  # noqa: N802 - Qt API
        if not self.isEnabled() or self.count() <= 0:
            return
        if self._stable_popup is not None:
            try:
                self._stable_popup.close()
            except RuntimeError:
                pass

        menu = QMenu(None)
        menu.setObjectName("stableComboPopup")
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        menu.setToolTipsVisible(True)
        current = self.currentIndex()
        model = self.model()
        root_index = self.rootModelIndex()
        model_column = self.modelColumn()
        for row in range(self.count()):
            index = model.index(row, model_column, root_index)
            # QComboBox.insertSeparator marks the row as inaccessible/disabled.
            # Preserve it as an actual menu separator instead of a selectable row.
            accessible = self.itemData(row, Qt.ItemDataRole.AccessibleDescriptionRole)
            flags = index.flags() if index.isValid() else Qt.ItemFlag.NoItemFlags
            if accessible == "separator" or not (flags & Qt.ItemFlag.ItemIsSelectable):
                menu.addSeparator()
                continue
            action = menu.addAction(self.itemIcon(row), self.itemText(row))
            action.setEnabled(bool(flags & Qt.ItemFlag.ItemIsEnabled))
            action.setCheckable(True)
            action.setChecked(row == current)
            tooltip = self.itemData(row, Qt.ItemDataRole.ToolTipRole)
            if tooltip:
                action.setToolTip(str(tooltip))
            action.hovered.connect(lambda r=row: self._highlight_menu_row(r))
            action.triggered.connect(lambda _checked=False, r=row: self._activate_menu_row(r))

        anchor, visual_width = self._popup_anchor_and_width()
        menu.setMinimumWidth(max(visual_width, 180))
        self._stable_popup = menu

        def _clear_popup() -> None:
            if self._stable_popup is menu:
                self._stable_popup = None

        menu.aboutToHide.connect(_clear_popup)
        menu.destroyed.connect(lambda _obj=None: _clear_popup())
        menu.popup(anchor)

    def hidePopup(self) -> None:  # noqa: N802 - Qt API
        menu = self._stable_popup
        self._stable_popup = None
        if menu is not None:
            try:
                menu.close()
            except RuntimeError:
                pass

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt API
        # A top-level QMenu has no QWidget parent by design.  Explicitly close it
        # when the combo/page/dialog is hidden so navigation cannot leave an
        # orphaned popup floating above the next Studio page.
        self.hidePopup()
        super().hideEvent(event)


def _configure_responsive_dialog(dialog: QDialog, preferred: tuple[int, int], minimum: tuple[int, int]) -> None:
    """Keep editor dialogs inside the usable desktop on macOS/HiDPI screens.

    Requested sizes such as 1260x860 can exceed the *available* desktop after the
    macOS menu bar/Dock or display scaling is applied.  Qt may then create a
    window whose lower controls are outside the visible area.  Size against
    availableGeometry instead and leave a small safety margin.
    """
    pw, ph = map(int, preferred)
    mw, mh = map(int, minimum)
    screen = dialog.screen() or QApplication.primaryScreen()
    if screen is not None:
        geo = screen.availableGeometry()
        cap_w = max(520, int(round(geo.width() * 0.94)))
        cap_h = max(420, int(round(geo.height() * 0.90)))
        pw = min(pw, cap_w)
        ph = min(ph, cap_h)
        mw = min(mw, cap_w)
        mh = min(mh, cap_h)
    dialog.setMinimumSize(max(520, mw), max(420, mh))
    dialog.resize(max(dialog.minimumWidth(), pw), max(dialog.minimumHeight(), ph))
    dialog.setSizeGripEnabled(True)

def _fit_scene_rect(scene: QGraphicsScene) -> QRectF:
    """Slightly pad scene fitting so the outermost page pixels stay visible."""
    rect = scene.sceneRect()
    if rect.isNull() or rect.isEmpty():
        return rect
    return rect.adjusted(-3.0, -3.0, 3.0, 3.0)

class StableThumbnailList(QListWidget):
    """Thumbnail gallery with deliberately damped trackpad/wheel scrolling.

    macOS pixelDelta gestures can arrive as very large bursts. QListWidget's
    default ScrollPerPixel path applies those bursts almost 1:1, which made a
    small finger movement jump several manga rows.  This widget clamps and
    scales the viewport movement without changing selection/currentItem, so
    scrolling and page switching are separate actions.
    """

    def wheelEvent(self, event):
        pixel = event.pixelDelta()
        angle = event.angleDelta()
        sb = self.verticalScrollBar()
        hsb = self.horizontalScrollBar()
        if not pixel.isNull():
            dy = int(round(pixel.y() * 0.34))
            dx = int(round(pixel.x() * 0.34))
            dy = max(-72, min(72, dy))
            dx = max(-72, min(72, dx))
            if dy:
                sb.setValue(sb.value() - dy)
            if dx:
                hsb.setValue(hsb.value() - dx)
            event.accept()
            return
        if not angle.isNull():
            # One physical wheel notch ~= 42 px, capped at two rows-worth of a
            # fast wheel event. This remains responsive but no longer leaps.
            dy = int(round((angle.y() / 120.0) * 42.0))
            dx = int(round((angle.x() / 120.0) * 42.0))
            dy = max(-84, min(84, dy))
            dx = max(-84, min(84, dx))
            if dy:
                sb.setValue(sb.value() - dy)
            if dx:
                hsb.setValue(hsb.value() - dx)
            event.accept()
            return
        super().wheelEvent(event)

class WorkbenchPageRail(QFrame):
    """Compact, lazy page rail for the transfer workspace.

    Inspired by Koharu's editor rail, but intentionally implemented with native Qt
    primitives and Folirina's existing page state.  It never owns project state:
    selecting an item only emits ``page_selected`` and the StudioWindow remains the
    single source of truth.
    """

    page_selected = Signal(int)
    collapse_requested = Signal()
    THUMB = QSize(54, 76)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("workbenchPageRail")
        self.setMinimumWidth(142)
        self.setMaximumWidth(218)
        self._signature = None
        self._pairs = []
        self._items: dict[int, QListWidgetItem] = {}
        self._thumb_cache: OrderedDict[tuple[str, int, int, int], QIcon] = OrderedDict()
        self._thumb_cache_limit = 72
        self._queue: list[int] = []
        self._loaded: set[int] = set()
        self._generation = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._pump_thumbnail)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 9, 8, 8)
        root.setSpacing(7)
        head = QHBoxLayout(); head.setSpacing(6)
        title = QLabel("页面"); title.setObjectName("pageRailTitle")
        self.count_label = QLabel("0"); self.count_label.setObjectName("pageRailCount")
        self.collapse_button = QPushButton("‹")
        self.collapse_button.setObjectName("pageRailCollapse")
        self.collapse_button.setFixedSize(25,25)
        self.collapse_button.setToolTip("向左收起页面栏")
        head.addWidget(title); head.addStretch(1); head.addWidget(self.count_label); head.addWidget(self.collapse_button)
        root.addLayout(head)

        self.search = QLineEdit()
        self.search.setObjectName("pageRailSearch")
        self.search.setPlaceholderText("搜索页码 / 文件名")
        self.search.setClearButtonEnabled(True)
        root.addWidget(self.search)

        self.list = StableThumbnailList()
        self.list.setObjectName("workbenchPageList")
        self.list.setViewMode(QListWidget.ViewMode.ListMode)
        self.list.setIconSize(self.THUMB)
        self.list.setSpacing(2)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list.verticalScrollBar().setSingleStep(24)
        root.addWidget(self.list, 1)

        self.footer = QLabel("TARGET 页面轨")
        self.footer.setObjectName("pageRailFooter")
        self.footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.footer)

        self.search.textChanged.connect(self._apply_filter)
        self.list.itemSelectionChanged.connect(self._selection_changed)
        self.list.verticalScrollBar().valueChanged.connect(lambda _v: self._schedule_visible())
        self.collapse_button.clicked.connect(self.collapse_requested.emit)

    def _signature_for(self, pairs) -> tuple:
        return tuple((str(getattr(p, "source_path", "")), str(getattr(p, "target_path", ""))) for p in pairs)

    def set_pages(self, pairs, current_index: int = 0) -> None:
        pairs = list(pairs or [])
        signature = self._signature_for(pairs)
        if signature != self._signature:
            self._signature = signature
            self._pairs = pairs
            self._generation += 1
            self._timer.stop()
            self._items.clear()
            self._loaded.clear()
            self.list.blockSignals(True)
            try:
                self.list.clear()
                for index, pair in enumerate(pairs):
                    target = Path(str(getattr(pair, "target_path", "")))
                    label = target.stem or f"Page {index + 1}"
                    if len(label) > 22:
                        label = label[:20] + "…"
                    item = QListWidgetItem(self._placeholder_icon(index), f"{index + 1:03d}  {label}")
                    item.setData(Qt.ItemDataRole.UserRole, index)
                    item.setToolTip(str(target))
                    item.setSizeHint(QSize(120, 84))
                    self.list.addItem(item)
                    self._items[index] = item
            finally:
                self.list.blockSignals(False)
            self.count_label.setText(str(len(pairs)))
            self._queue = []
            self._apply_filter()
        else:
            self._pairs = pairs
            self.count_label.setText(str(len(pairs)))
        self.set_current_page(current_index)
        self._prioritize(current_index)

    def set_current_page(self, index: int) -> None:
        if not self._items:
            return
        index = max(0, min(int(index), len(self._items) - 1))
        item = self._items.get(index)
        if item is None:
            return
        if self.list.currentItem() is item:
            return
        self.list.blockSignals(True)
        try:
            self.list.setCurrentItem(item)
            item.setSelected(True)
            self.list.scrollToItem(item, QAbstractItemView.ScrollHint.EnsureVisible)
        finally:
            self.list.blockSignals(False)

    def _selection_changed(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        value = item.data(Qt.ItemDataRole.UserRole)
        try:
            self.page_selected.emit(int(value))
        except (TypeError, ValueError):
            return

    def _apply_filter(self, *_args) -> None:
        query = self.search.text().strip().casefold() if hasattr(self, "search") else ""
        for index, item in self._items.items():
            pair = self._pairs[index] if 0 <= index < len(self._pairs) else None
            path = str(getattr(pair, "target_path", "")) if pair is not None else ""
            haystack = f"{index + 1} {Path(path).name}".casefold()
            item.setHidden(bool(query and query not in haystack))
        self._schedule_visible()

    def _placeholder_icon(self, index: int) -> QIcon:
        pix = QPixmap(self.THUMB)
        pix.fill(QColor("#F4F6F9"))
        painter = QPainter(pix)
        painter.setPen(QColor("#98A4B3"))
        painter.drawRect(0, 0, self.THUMB.width() - 1, self.THUMB.height() - 1)
        painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, str(index + 1))
        painter.end()
        return QIcon(pix)

    def _thumbnail_icon(self, index: int) -> QIcon:
        if not (0 <= index < len(self._pairs)):
            return self._placeholder_icon(index)
        path = Path(str(getattr(self._pairs[index], "target_path", "")))
        try:
            st = path.stat()
            key = (str(path), int(st.st_mtime_ns), int(st.st_size), self.THUMB.width())
        except OSError:
            return self._placeholder_icon(index)
        cached = self._thumb_cache.pop(key, None)
        if cached is not None:
            self._thumb_cache[key] = cached
            return cached
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid() and size.width() > 0 and size.height() > 0:
            scale = min(self.THUMB.width() / size.width(), self.THUMB.height() / size.height())
            reader.setScaledSize(QSize(max(1, int(size.width() * scale)), max(1, int(size.height() * scale))))
        image = reader.read()
        if image.isNull():
            return self._placeholder_icon(index)
        canvas = QPixmap(self.THUMB); canvas.fill(QColor("#FFFFFF"))
        pix = QPixmap.fromImage(image)
        x = max(0, (self.THUMB.width() - pix.width()) // 2)
        y = max(0, (self.THUMB.height() - pix.height()) // 2)
        painter = QPainter(canvas); painter.drawPixmap(x, y, pix); painter.end()
        icon = QIcon(canvas)
        self._thumb_cache[key] = icon
        while len(self._thumb_cache) > self._thumb_cache_limit:
            self._thumb_cache.popitem(last=False)
        return icon

    def _visible_indices(self) -> list[int]:
        if not self._items:
            return []
        viewport = self.list.viewport().rect().adjusted(0, -120, 0, 120)
        out = []
        for index, item in self._items.items():
            if item.isHidden():
                continue
            if index not in self._loaded and self.list.visualItemRect(item).intersects(viewport):
                out.append(index)
        return out

    def _prioritize(self, index: int) -> None:
        order = []
        for delta in (0, -1, 1, -2, 2, -3, 3, -5, 5):
            value = int(index) + delta
            if 0 <= value < len(self._pairs) and value not in order:
                order.append(value)
        order.extend(i for i in self._visible_indices() if i not in order)
        self._queue = order + [i for i in self._queue if i not in order]
        if self._queue and not self._timer.isActive():
            self._timer.start(0)

    def _schedule_visible(self) -> None:
        visible = self._visible_indices()
        if not visible:
            return
        self._queue = visible + [i for i in self._queue if i not in visible]
        if not self._timer.isActive():
            self._timer.start(24)

    def _pump_thumbnail(self) -> None:
        while self._queue:
            index = self._queue.pop(0)
            item = self._items.get(index)
            if item is None or item.isHidden():
                continue
            item.setIcon(self._thumbnail_icon(index))
            self._loaded.add(index)
            break
        if self._queue:
            self._timer.start(18)


class ImageView(QGraphicsView):
    def __init__(self, parent=None, *, max_decode_side: int = 0):
        super().__init__(parent)
        self._max_decode_side = max(0, int(max_decode_side))
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = None
        self._current_key = None
        # Switching workbench tabs used to synchronously decode the same multi-MB
        # manga PNG/JPEG every time. Keep a tiny LRU of decoded QPixmaps; QPixmap is
        # implicitly shared by Qt, so reuse is cheap while the cache remains bounded.
        self._pixmap_cache: OrderedDict[tuple[str, int, int, int], QPixmap] = OrderedDict()
        self._pixmap_cache_limit = 5
        # Auto-fit previews must not let transient scrollbars change the viewport
        # size while a new page is being installed.  Qt's AsNeeded policy can
        # otherwise create a resize -> fitInView -> scrollbar -> resize feedback
        # loop that is visible as a horizontal/right-edge "shake" on page switch.
        self._fit_pending = False
        self._fit_in_progress = False
        self.setObjectName("workbenchImage")
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # Match Colortina's canvas contract: the scene owns image-pixel space,
        # while the view only handles fit/zoom/pan.  Explicit center alignment
        # prevents a tall manga page from visually sticking to a corner when
        # the central workspace is much wider than the page itself.
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        # ImageView is an always-fit preview (unlike ZoomPreviewView).  A fitted
        # page never needs scrollbars, and keeping them permanently off makes the
        # central canvas width stable across portrait pages of slightly different
        # aspect ratios.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(320)

    def set_image(self, path: str | Path | None):
        if not path:
            if self._item is not None:
                self._scene.clear(); self._item = None
            self._current_key = None
            return False
        p = Path(path)
        try:
            st = p.stat()
        except OSError:
            if self._item is not None:
                self._scene.clear(); self._item = None
            self._current_key = None
            return False
        key = (str(p.resolve()), int(st.st_mtime_ns), int(st.st_size), int(self._max_decode_side))
        if key == self._current_key and self._item is not None:
            return True
        pix = self._pixmap_cache.pop(key, None)
        if pix is None:
            if self._max_decode_side > 0:
                reader = QImageReader(str(p)); reader.setAutoTransform(True)
                size = reader.size()
                if size.isValid() and max(size.width(), size.height()) > self._max_decode_side:
                    target = size.scaled(QSize(self._max_decode_side, self._max_decode_side), Qt.AspectRatioMode.KeepAspectRatio)
                    reader.setScaledSize(target)
                image = reader.read()
                pix = QPixmap.fromImage(image) if not image.isNull() else QPixmap()
            else:
                pix = QPixmap(str(p))
            if pix.isNull():
                # Do not leave the previously selected page visible when a new
                # file exists but cannot be decoded.  A stale preview is worse
                # than an empty preview because it can make the operator edit the
                # wrong page.
                if self._item is not None:
                    self._scene.clear(); self._item = None
                self._current_key = None
                return False
        self._pixmap_cache[key] = pix
        while len(self._pixmap_cache) > self._pixmap_cache_limit:
            self._pixmap_cache.popitem(last=False)
        self._scene.clear(); self._item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(self._item.boundingRect())
        self._current_key = key
        self._schedule_fit()
        return True

    def clear_cache(self):
        """Force the next refresh to decode the file again.

        Manual review can rewrite ``final_reviewed.png`` in-place.  Even though
        mtime-based cache keys normally catch that, explicitly invalidating the
        preview after a committed edit removes filesystem timestamp granularity
        as a source of stale UI pixels.
        """
        self._pixmap_cache.clear()
        self._current_key = None

    def _schedule_fit(self):
        if self._item is None or self._fit_pending:
            return
        self._fit_pending = True
        # Coalesce image replacement and the layout/viewport resize it triggers
        # into one event-loop turn.  This prevents two visibly different scale
        # transforms from being painted during one page switch.
        QTimer.singleShot(0, self._apply_fit)

    def _apply_fit(self):
        self._fit_pending = False
        if self._item is None or self._fit_in_progress:
            return
        if self.viewport().width() < 8 or self.viewport().height() < 8:
            return
        self._fit_in_progress = True
        try:
            self.resetTransform()
            rect = _fit_scene_rect(self._scene)
            if not rect.isNull() and not rect.isEmpty():
                self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
                self.centerOn(self._item)
        finally:
            self._fit_in_progress = False

    def fit_to_window(self):
        if self._item is not None:
            self._schedule_fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_fit()

class Card(QFrame):
    def __init__(self, title: str = "", subtitle: str = "", *, blue=False, parent=None):
        super().__init__(parent)
        self.setObjectName("cardBlue" if blue else "card")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 13, 15, 14)
        self.layout.setSpacing(9)
        self.title_label = None
        self.subtitle_label = None
        if title:
            self.title_label = QLabel(title); self.title_label.setObjectName("cardTitle"); self.layout.addWidget(self.title_label)
        if subtitle:
            self.subtitle_label = QLabel(subtitle); self.subtitle_label.setObjectName("cardSubtitle"); self.subtitle_label.setWordWrap(True); self.layout.addWidget(self.subtitle_label)

class PageHero(QFrame):
    def __init__(self, title: str, subtitle: str, chips: list[str] | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("pageHero")
        lay = QVBoxLayout(self); lay.setContentsMargins(16, 14, 16, 14); lay.setSpacing(8)
        h = QLabel(title); h.setObjectName("heroTitle"); lay.addWidget(h)
        sub = QLabel(subtitle); sub.setObjectName("heroHint"); sub.setWordWrap(True); lay.addWidget(sub)
        if chips:
            row = QHBoxLayout(); row.setContentsMargins(0,0,0,0); row.setSpacing(8)
            for text in chips:
                chip = QLabel(text); chip.setObjectName("infoChip"); row.addWidget(chip)
            row.addStretch(1)
            lay.addLayout(row)

class OptionRow(QFrame):
    def __init__(self, label: str, hint: str, parent=None):
        super().__init__(parent)
        self.setObjectName("optionRow")
        self.setProperty("selected", False)
        lay = QVBoxLayout(self); lay.setContentsMargins(10, 8, 10, 8); lay.setSpacing(2)
        top = QHBoxLayout(); top.setContentsMargins(0,0,0,0); top.setSpacing(8)
        self.radio = QRadioButton(label)
        top.addWidget(self.radio)
        top.addStretch(1)
        lay.addLayout(top)
        self.hint = QLabel(hint); self.hint.setObjectName("optionHint"); self.hint.setWordWrap(True); lay.addWidget(self.hint)

    def set_selected(self, value: bool) -> None:
        self.setProperty("selected", bool(value))
        self.style().unpolish(self); self.style().polish(self)
        self.update()

class PathRow(QWidget):
    changed = Signal(str)

    def __init__(self, label: str, button_text: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(8)
        self.caption = QLabel(label); self.caption.setFixedWidth(84); self.caption.setObjectName("quiet")
        self.path = QLabel("未选择"); self.path.setObjectName("hint"); self.path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.button = QPushButton(button_text); self.button.setObjectName("compactAction")
        lay.addWidget(self.caption); lay.addWidget(self.path, 1); lay.addWidget(self.button)

    def set_path(self, path: str):
        self.path.setText(path or "未选择")
        self.path.setToolTip(path or "")

class ZoomPreviewView(QGraphicsView):
    """Full-page preview with mouse-wheel zoom and hand-drag panning."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = None
        self._zoom = 1.0
        self._auto_fit = True; self._fit_pending = False
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMinimumSize(360, 420)

    def set_image(self, path: str | Path | None):
        self._scene.clear(); self._item = None; self._zoom = 1.0; self.resetTransform()
        if not path:
            self._show_message("无图片")
            return
        p = Path(path)
        if not p.exists():
            self._show_message(f"图片不存在\n{p.name}")
            return
        reader = QImageReader(str(p)); reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            self._show_message(f"无法读取图片\n{p.name}")
            return
        pix = QPixmap.fromImage(image)
        self._item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(self._item.boundingRect())
        self.fit_to_window()

    def _show_message(self, text: str):
        item = self._scene.addText(text)
        item.setDefaultTextColor(QColor(MUTED))
        self._scene.setSceneRect(item.boundingRect().adjusted(-30, -30, 30, 30))

    def _apply_fit(self):
        self._fit_pending = False
        if self._item is None or not self._auto_fit or self.viewport().width() < 8 or self.viewport().height() < 8:
            return
        self.resetTransform()
        self.fitInView(_fit_scene_rect(self._scene), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = max(0.01, float(self.transform().m11()))

    def fit_to_window(self):
        if self._item is not None:
            self._auto_fit = True; self._apply_fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit and not self._fit_pending:
            self._fit_pending = True; QTimer.singleShot(0, self._apply_fit)

    def actual_size(self):
        if self._item is not None:
            self._auto_fit = False; self.resetTransform(); self._zoom = 1.0

    def zoom_by(self, factor: float):
        if self._item is None:
            return
        current = float(self.transform().m11())
        target = current * float(factor)
        if target < 0.05 or target > 12.0:
            return
        self._auto_fit = False
        self.scale(float(factor), float(factor)); self._zoom = target

    def wheelEvent(self, event):
        if self._item is None:
            super().wheelEvent(event); return
        self.zoom_by(1.16 if event.angleDelta().y() > 0 else (1.0 / 1.16))
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.fit_to_window(); event.accept()

__all__ = ['StableThumbnailList', 'ImageView', 'Card', 'PageHero', 'OptionRow', 'PathRow', 'ZoomPreviewView', 'StableComboBox', 'StableSpinBox', 'StableDoubleSpinBox', 'StableSlider', '_fit_scene_rect', '_configure_responsive_dialog']
