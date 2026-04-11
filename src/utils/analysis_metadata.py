# -*- coding: utf-8 -*-
"""
Shared constants describing analysis metadata fields.

Kept in a dedicated module so that both API schemas and business logic
can reference the same patterns without circular imports.
"""

# Regex pattern for ``selection_source`` — the origin of a stock pick.
# Valid values: manual | autocomplete | import | image
SELECTION_SOURCE_PATTERN = r"^(manual|autocomplete|import|image)$"
