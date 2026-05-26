import os

# Run Qt tests headlessly. Must be set before Qt is imported anywhere.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
