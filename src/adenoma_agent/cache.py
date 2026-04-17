from pathlib import Path

from adenoma_agent.utils import ensure_dir, read_json, sha1_payload, write_json


class JsonCache(object):
    def __init__(self, root):
        self.root = ensure_dir(root)

    def key_dir(self, namespace, payload):
        cache_key = sha1_payload(payload)
        return ensure_dir(Path(self.root) / namespace / cache_key)

    def key_path(self, namespace, payload, filename):
        return self.key_dir(namespace, payload) / filename

    def load(self, namespace, payload, filename):
        path = self.key_path(namespace, payload, filename)
        if not path.exists():
            return None
        return read_json(path)

    def save(self, namespace, payload, filename, data):
        path = self.key_path(namespace, payload, filename)
        write_json(path, data)
        return path
