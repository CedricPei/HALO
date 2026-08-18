"""TXT file storage and grep-style search."""

from pathlib import Path

from ._store import keyword_match


class Files:
    """Manage TXT files directly inside one directory."""

    def __init__(self, storage_dir):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name):
        return self.storage_dir / name

    def _file(self, name):
        """Return an existing file path selected by exact filename."""
        path = self._path(name)
        if not path.is_file():
            raise KeyError(f"File not found: {name}")
        return path

    def _check_name(self, name):
        """Keep managed content as unique, plain .txt files."""
        if Path(name).name != name or not name.endswith(".txt"):
            raise ValueError("name must be a plain .txt filename")
        if self._path(name).exists():
            raise ValueError(f"File already exists: {name}")

    def create_file(self, name, content=""):
        """Create a TXT file with optional initial content."""
        self._check_name(name)
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        self._path(name).write_text(content, encoding="utf-8")
        return {"name": name, "content": content}

    def get_file(self, name):
        """Return the name and content of an exact file."""
        return {
            "name": name,
            "content": self._file(name).read_text(encoding="utf-8"),
        }

    def list_files(self):
        """Return the name of every managed file."""
        return [{"name": path.name} for path in sorted(self.storage_dir.glob("*.txt"))]

    def search_files(self, query):
        """Search filenames and full text content."""
        results = []
        for file in self.list_files():
            content = self._path(file["name"]).read_text(encoding="utf-8")
            if keyword_match(query, file["name"], content):
                results.append({**file, "content": content})
        return results

    def append_file(self, name, content):
        """Append content to a file selected by name."""
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        with self._file(name).open("a", encoding="utf-8") as file:
            file.write(content)
        return self.get_file(name)

    def replace_file(self, name, content):
        """Replace all content in a file selected by name."""
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        self._file(name).write_text(content, encoding="utf-8")
        return self.get_file(name)

    def delete_file(self, name):
        """Delete a file selected by name."""
        result = self.get_file(name)
        self._file(name).unlink()
        return result
