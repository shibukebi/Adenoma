"""Minimal OOXML annotation workbook writer and reader.

The project intentionally avoids adding an annotation frontend or a heavy
Excel dependency.  This module writes a single protected worksheet with
unlocked expert-entry cells, hyperlinks, and list data validation.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML_NS = "http://www.w3.org/XML/1998/namespace"

LOCKED_COLUMNS = (
    "annotation_instance_id",
    "roi_id",
    "safe_slide_id",
    "image_file",
)
EXPERT_COLUMNS = (
    "architecture_label",
    "evaluable",
    "pure_or_mixed",
    "expert_confidence",
    "architecture_components",
    "exclusion_reason",
    "notes",
)
ANNOTATION_COLUMNS = LOCKED_COLUMNS + EXPERT_COLUMNS

DROPDOWNS = {
    "architecture_label": ("serrated", "tubular", "villous", "mixed", "uncertain"),
    "evaluable": ("yes", "no"),
    "pure_or_mixed": ("pure", "mixed"),
    "expert_confidence": ("high", "medium", "low"),
    "exclusion_reason": (
        "artifact",
        "insufficient_tissue",
        "blur",
        "folding",
        "poor_crop",
        "non_mucosal",
        "fragmented",
        "other",
    ),
}


def _xml(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _column_name(index: int) -> str:
    value = int(index) + 1
    output = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        output = chr(65 + remainder) + output
    return output


def _column_index(cell_ref: str) -> int:
    letters = "".join(character for character in str(cell_ref or "") if character.isalpha())
    value = 0
    for character in letters.upper():
        value = value * 26 + ord(character) - ord("A") + 1
    return max(0, value - 1)


def _inline_cell(ref: str, value: Any, style: int) -> str:
    text = str(value if value is not None else "")
    preserve = ' xml:space="preserve"' if text != text.strip() else ""
    return '<c r="{0}" s="{1}" t="inlineStr"><is><t{2}>{3}</t></is></c>'.format(
        ref, int(style), preserve, _xml(text)
    )


def _zip_write(archive: ZipFile, name: str, payload: str) -> None:
    info = ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, payload.encode("utf-8"))


def write_annotation_workbook(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write the formal blinded one-row-per-ROI annotation workbook."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row_xml = []
    hyperlinks = []
    hyperlink_rels = []
    header_cells = [_inline_cell("{0}1".format(_column_name(index)), name, 1) for index, name in enumerate(ANNOTATION_COLUMNS)]
    row_xml.append('<row r="1" ht="30" customHeight="1">{0}</row>'.format("".join(header_cells)))
    for row_number, row in enumerate(rows, 2):
        cells = []
        for index, column in enumerate(ANNOTATION_COLUMNS):
            ref = "{0}{1}".format(_column_name(index), row_number)
            if column in LOCKED_COLUMNS:
                style = 4 if column == "image_file" else 2
                value = row.get(column, "")
            else:
                style = 3
                value = row.get(column, "")
            cells.append(_inline_cell(ref, value, style))
            if column == "image_file":
                rel_id = "rId{0}".format(len(hyperlinks) + 1)
                hyperlinks.append('<hyperlink ref="{0}" r:id="{1}"/>'.format(ref, rel_id))
                hyperlink_rels.append(
                    '<Relationship Id="{0}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="{1}" TargetMode="External"/>'.format(
                        rel_id, _xml(row.get(column, ""))
                    )
                )
        row_xml.append('<row r="{0}">{1}</row>'.format(row_number, "".join(cells)))

    last_row = max(2, len(rows) + 1)
    validations = []
    for column, values in DROPDOWNS.items():
        index = ANNOTATION_COLUMNS.index(column)
        letter = _column_name(index)
        formula = '"{0}"'.format(",".join(values))
        validations.append(
            '<dataValidation type="list" allowBlank="1" showErrorMessage="1" showInputMessage="1" '
            'errorTitle="Invalid value" error="Choose a value from the dropdown." sqref="{0}2:{0}{1}">'
            '<formula1>{2}</formula1></dataValidation>'.format(letter, last_row, _xml(formula))
        )
    widths = [24, 16, 18, 30, 22, 14, 18, 20, 34, 26, 44]
    cols = "".join(
        '<col min="{0}" max="{0}" width="{1}" customWidth="1"/>'.format(index + 1, width)
        for index, width in enumerate(widths)
    )
    dimension = "A1:{0}{1}".format(_column_name(len(ANNOTATION_COLUMNS) - 1), last_row)
    sheet = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="{dimension}"/>
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <sheetFormatPr defaultRowHeight="18"/>
  <cols>{cols}</cols>
  <sheetData>{rows}</sheetData>
  <autoFilter ref="A1:{last_column}{last_row}"/>
  <sheetProtection sheet="1" objects="1" scenarios="1" selectLockedCells="1" selectUnlockedCells="0"/>
  <dataValidations count="{validation_count}">{validations}</dataValidations>
  <hyperlinks>{hyperlinks}</hyperlinks>
</worksheet>""".format(
        dimension=dimension,
        cols=cols,
        rows="".join(row_xml),
        last_column=_column_name(len(ANNOTATION_COLUMNS) - 1),
        last_row=last_row,
        validation_count=len(validations),
        validations="".join(validations),
        hyperlinks="".join(hyperlinks),
    )
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="3">
    <font><sz val="11"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font>
    <font><u/><color rgb="FF0563C1"/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="4">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFFF2CC"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left style="thin"><color rgb="FFD9E1F2"/></left><right style="thin"><color rgb="FFD9E1F2"/></right><top style="thin"><color rgb="FFD9E1F2"/></top><bottom style="thin"><color rgb="FFD9E1F2"/></bottom><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="5">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyAlignment="1" applyProtection="1"><alignment wrapText="1" vertical="center"/><protection locked="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyAlignment="1" applyProtection="1"><alignment vertical="center"/><protection locked="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="3" borderId="1" xfId="0" applyAlignment="1" applyProtection="1"><alignment wrapText="1" vertical="top"/><protection locked="0"/></xf>
    <xf numFmtId="0" fontId="2" fillId="0" borderId="1" xfId="0" applyAlignment="1" applyProtection="1"><alignment vertical="center"/><protection locked="1"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""
    workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <bookViews><workbookView xWindow="0" yWindow="0" windowWidth="24000" windowHeight="14000"/></bookViews>
  <sheets><sheet name="Annotation" sheetId="1" r:id="rId1"/></sheets>
  <calcPr calcId="191029"/>
</workbook>"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""
    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""
    sheet_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{0}</Relationships>""".format(
        "".join(hyperlink_rels)
    )
    core = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>Adenoma benchmark preparation</dc:creator><cp:lastModifiedBy>Pathology expert</cp:lastModifiedBy><dc:title>Blinded 5x ROI Architecture Annotation</dc:title>
</cp:coreProperties>"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Adenoma benchmark preparation</Application></Properties>"""

    with ZipFile(str(path), "w") as archive:
        for name, payload in (
            ("[Content_Types].xml", content_types),
            ("_rels/.rels", root_rels),
            ("docProps/core.xml", core),
            ("docProps/app.xml", app),
            ("xl/workbook.xml", workbook),
            ("xl/_rels/workbook.xml.rels", workbook_rels),
            ("xl/styles.xml", styles),
            ("xl/worksheets/sheet1.xml", sheet),
            ("xl/worksheets/_rels/sheet1.xml.rels", sheet_rels),
        ):
            _zip_write(archive, name, payload)
    return path


def _shared_strings(archive: ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    ns = "{" + SPREADSHEET_NS + "}"
    return ["".join(node.text or "" for node in item.findall(".//" + ns + "t")) for item in root.findall(ns + "si")]


def _sheet_path(archive: ZipFile, sheet_name: str) -> str:
    ns = "{" + SPREADSHEET_NS + "}"
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    target_sheet = None
    for sheet in workbook.findall(ns + "sheets/" + ns + "sheet"):
        if sheet.attrib.get("name") == sheet_name:
            target_sheet = sheet
            break
    if target_sheet is None:
        raise ValueError("XLSX workbook has no {0!r} worksheet".format(sheet_name))
    rel_id = target_sheet.attrib.get("{" + OFFICE_REL_NS + "}id", "")
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for relationship in relationships.findall("{" + PACKAGE_REL_NS + "}Relationship"):
        if relationship.attrib.get("Id") == rel_id:
            target = relationship.attrib.get("Target", "")
            if target.startswith("/"):
                return target.lstrip("/")
            if target.startswith("xl/"):
                return target
            return "xl/" + target.lstrip("/")
    raise ValueError("XLSX worksheet relationship is missing")


def read_annotation_workbook(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read the Annotation sheet and return rows plus all visible cell text."""

    path = Path(path)
    ns = "{" + SPREADSHEET_NS + "}"
    with ZipFile(str(path)) as archive:
        shared = _shared_strings(archive)
        sheet_root = ET.fromstring(archive.read(_sheet_path(archive, "Annotation")))
        raw_rows: List[List[str]] = []
        all_text: List[str] = []
        for row in sheet_root.findall(ns + "sheetData/" + ns + "row"):
            values: List[str] = []
            for cell in row.findall(ns + "c"):
                index = _column_index(cell.attrib.get("r", "A1"))
                while len(values) <= index:
                    values.append("")
                cell_type = cell.attrib.get("t", "")
                if cell_type == "inlineStr":
                    value = "".join(node.text or "" for node in cell.findall(".//" + ns + "t"))
                else:
                    value_node = cell.find(ns + "v")
                    value = value_node.text if value_node is not None and value_node.text is not None else ""
                    if cell_type == "s" and value:
                        value = shared[int(value)]
                    elif cell_type == "b":
                        value = "true" if value == "1" else "false"
                formula = cell.find(ns + "f")
                if formula is not None and formula.text:
                    all_text.append(formula.text)
                value = str(value).strip()
                values[index] = value
                all_text.append(value)
            raw_rows.append(values)
        relationship_name = "xl/worksheets/_rels/sheet1.xml.rels"
        if relationship_name in archive.namelist():
            relationship_root = ET.fromstring(archive.read(relationship_name))
            for relationship in relationship_root.findall("{" + PACKAGE_REL_NS + "}Relationship"):
                all_text.append(str(relationship.attrib.get("Target", "")))
    if not raw_rows:
        return [], all_text
    headers = [str(value).strip() for value in raw_rows[0]]
    rows = []
    for raw in raw_rows[1:]:
        values = list(raw) + [""] * max(0, len(headers) - len(raw))
        if not any(str(value).strip() for value in values):
            continue
        rows.append({headers[index]: str(values[index]).strip() for index in range(len(headers)) if headers[index]})
    return rows, all_text


def workbook_structure(path: Path) -> Dict[str, Any]:
    """Return protection, validation and header facts used by audits/tests."""

    ns = "{" + SPREADSHEET_NS + "}"
    with ZipFile(str(path)) as archive:
        sheet = ET.fromstring(archive.read(_sheet_path(archive, "Annotation")))
        protection = sheet.find(ns + "sheetProtection") is not None
        validations = sheet.findall(ns + "dataValidations/" + ns + "dataValidation")
        rows, _ = read_annotation_workbook(path)
        first_row = sheet.find(ns + "sheetData/" + ns + "row")
        headers = []
        if first_row is not None:
            for cell in first_row.findall(ns + "c"):
                headers.append("".join(node.text or "" for node in cell.findall(".//" + ns + "t")))
    return {
        "protected": protection,
        "validation_count": len(validations),
        "validation_ranges": [item.attrib.get("sqref", "") for item in validations],
        "headers": headers,
        "data_rows": len(rows),
    }


def find_forbidden_text(values: Sequence[str], forbidden_patterns: Sequence[str]) -> List[Dict[str, str]]:
    findings = []
    for value in values:
        normalized = str(value or "")
        for pattern in forbidden_patterns:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                findings.append({"pattern": pattern, "value": normalized[:240]})
    return findings
