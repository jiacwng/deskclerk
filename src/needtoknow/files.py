# Reads a company's folder of PDF, Word, PowerPoint, Excel, HTML, Markdown and text files into
# documents, with MarkItDown turning each file into Markdown. An access file says who may read
# what, one JSON object per line: a path pattern matched against the whole path relative to the
# folder, and the users and groups who may read the files it matches, or public for everyone.
# As in a CODEOWNERS file the last matching line wins, and a file no line matches stops the
# load before anything is written.

import os
import re
import stat
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

from markitdown import MarkItDown, MarkItDownException, StreamInfo
from markitdown.converters import (
    DocxConverter,
    HtmlConverter,
    PdfConverter,
    PlainTextConverter,
    PptxConverter,
    XlsxConverter,
)
from pydantic import Field, ValidationError

from needtoknow.company import Readers
from needtoknow.documents import Document

FILE_PREFIX = "file-"
EXTENSIONS = frozenset({".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".txt"})
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


class FilesError(ValueError):
    pass


class AccessLine(Readers):
    path: str = Field(min_length=1)


@dataclass(frozen=True)
class Found:
    documents: list[Document]
    unsupported: int
    oversized: int


def read_access(path: Path) -> list[AccessLine]:
    entries = []
    with path.open(encoding="utf-8-sig") as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                entry = AccessLine.model_validate_json(line)
            except ValidationError as error:
                raise FilesError(f"{path.name} line {number}: {error}") from None
            pattern = PurePosixPath(entry.path)
            if pattern.is_absolute() or ".." in pattern.parts:
                raise FilesError(
                    f"{path.name} line {number}: path {entry.path!r} must stay inside the folder"
                )
            entries.append(entry)
    return entries


def read_files(folder: Path, access_path: Path, max_bytes: int) -> Found:
    access = read_access(access_path)
    root = folder.resolve()
    if not root.is_dir():
        raise FilesError(f"{folder} is not a folder")
    wanted: list[tuple[Path, str, frozenset[str]]] = []
    unmapped = []
    unsupported = 0
    oversized = 0
    for path in _walk(root):
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() not in EXTENSIONS:
            unsupported += 1
            continue
        if path.stat().st_size > max_bytes:
            oversized += 1
            continue
        readers = _readers(relative, access)
        if readers is None:
            unmapped.append(relative)
        else:
            wanted.append((path, relative, readers))
    if unmapped:
        raise FilesError(f"{access_path.name} gives no readers for {', '.join(unmapped)}")
    converter = _converter()
    documents = []
    for path, relative, readers in wanted:
        try:
            text = converter.convert_local(
                path, stream_info=StreamInfo(extension=path.suffix.lower())
            ).text_content
        except MarkItDownException as error:
            raise FilesError(f"{relative}: {error}") from None
        heading = _HEADING.search(text)
        documents.append(
            Document(
                id=FILE_PREFIX + relative,
                title=heading.group(1) if heading else path.stem,
                body=text,
                readers=readers,
            )
        )
    return Found(documents=documents, unsupported=unsupported, oversized=oversized)


def _walk(root: Path) -> list[Path]:
    paths = []
    for directory, subdirectories, names in os.walk(root, followlinks=False):
        subdirectories.sort()
        for name in sorted(subdirectories + names):
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise FilesError(f"{path.relative_to(root)} is a symbolic link")
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise FilesError(f"{path.relative_to(root)} is not a regular file")
            if not path.resolve().is_relative_to(root):
                raise FilesError(f"{path.relative_to(root)} resolves outside {root}")
            paths.append(path)
    return paths


def _readers(relative: str, access: list[AccessLine]) -> frozenset[str] | None:
    readers = None
    for entry in access:
        if fnmatchcase(relative, entry.path):
            readers = entry.principals()
    return readers


# Only the converters for the supported types: no exiftool subprocess, no archive unpacking, no
# plugins.
def _converter() -> MarkItDown:
    converter = MarkItDown(enable_builtins=False, enable_plugins=False)
    for each in (
        PlainTextConverter(),
        HtmlConverter(),
        PdfConverter(),
        DocxConverter(),
        PptxConverter(),
        XlsxConverter(),
    ):
        converter.register_converter(each)
    return converter
