import re

path = r'E:\SIH26/src/rf_analyzer/gui/main_window.py'
content = open(path).read()

# Remove unused contextlib import since we're reverting to try/except
content = content.replace('from __future__ import annotations\n\nimport contextlib\n', 'from __future__ import annotations\n')

# Fix 1: Revert the try/except at line ~151-158  
old1 = '''    try:
        if kind == "warning":
            QMessageBox.warning(parent, title, text)
        elif kind == "critical":
            QMessageBox.critical(parent, title, text)
        else:
            QMessageBox.information(parent, title, text)
    except Exception:  # noqa: BLE001,S110
        pass'''
new1 = '''    try:
        if kind == "warning":
            QMessageBox.warning(parent, title, text)
        elif kind == "critical":
            QMessageBox.critical(parent, title, text)
        else:
            QMessageBox.information(parent, title, text)
    except Exception:  # noqa: BLE001,S110
        pass'''
assert old1 in content, 'old1 not found'

# Fix 2: Revert the waterfall colormap
old2 = '''        # Dark, RF-conventional colormap.
        contextlib.suppress(Exception)(
            self.waterfall_view.setColorMap, pg.colormap.get("inferno")
        )'''
new2 = '''        # Dark, RF-conventional colormap.
        try:  # noqa: BLE001,S110
            self.waterfall_view.setColorMap(pg.colormap.get("inferno"))
        except Exception:
            pass'''
assert old2 in content, 'old2 not found'
content = content.replace(old2, new2)

open(path, 'w').write(content)
print('fixed main_window.py')
