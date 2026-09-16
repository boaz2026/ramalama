from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import Mock

import pytest

from ramalama.model_store.snapshot_file import SnapshotFile
from ramalama.transports.url import URL


@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.parametrize("authority", ["localhost", "localhost:8000", "localhost:9000", "[::1]", "[::1]:8000"])
@pytest.mark.parametrize("path,tag", [("model.gguf", "latest"), ("org/model.gguf:v1", "v1")])
def test_http_identifiers_preserve_authority(tmp_path, scheme, authority, path, tag):
    model_url = f"{authority}/{path}"
    model = URL(model_url, str(tmp_path), scheme)
    organization = f"{authority}/org" if path.startswith("org/") else authority

    assert model.extract_model_identifiers() == ("model.gguf", tag, organization)
    assert model.model == model_url


@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.parametrize("host", ["localhost", "[::1]"])
def test_http_pull_uses_separate_cache_for_each_port(tmp_path, monkeypatch, scheme, host):
    # Each pull uses a new URL instance; the third must reuse the first URL's cache.
    urls = [f"{scheme}://{host}:{port}/model.gguf" for port in (8000, 9000, 8000)]
    contents = {url: f"model from {url}".encode() for url in urls}

    def download(url, dest_path, **kwargs):
        Path(dest_path).write_bytes(contents[url])

    downloader = Mock(side_effect=download)
    monkeypatch.setattr("ramalama.model_store.snapshot_file.download_file", downloader)

    for url, should_download in zip(urls, [True, True, False]):
        downloader.reset_mock()
        model = URL(url.removeprefix(f"{scheme}://"), str(tmp_path), scheme)
        model.pull(Namespace(verify=False))

        if should_download:
            downloader.assert_called_once()
            assert downloader.call_args.kwargs["url"] == url
        else:
            downloader.assert_not_called()

        ref = model.model_store.get_ref_file("latest")
        assert ref is not None
        assert Path(model.model_store.get_blob_file_path(ref.files[0].hash)).read_bytes() == contents[url]


@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.parametrize("host", ["localhost", "[::1]"])
def test_http_remove_preserves_model_on_other_port(tmp_path, monkeypatch, scheme, host):
    models = [URL(f"{host}:{port}/model.gguf", str(tmp_path), scheme) for port in (8000, 9000)]

    def download(url, dest_path, **kwargs):
        Path(dest_path).write_bytes(f"model from {url}".encode())

    downloader = Mock(side_effect=download)
    monkeypatch.setattr("ramalama.model_store.snapshot_file.download_file", downloader)
    for model in models:
        model.pull(Namespace(verify=False))
    assert [call.kwargs["url"] for call in downloader.call_args_list] == [
        f"{scheme}://{model.model}" for model in models
    ]
    downloader.reset_mock()
    removed, retained = models

    assert removed.remove(Namespace(ignore=False))
    assert removed.model_store.get_ref_file("latest") is None
    assert retained.model_store.get_cached_files("latest")[2]

    ref = retained.model_store.get_ref_file("latest")
    assert ref is not None
    expected = f"model from {retained.type}://{retained.model}".encode()
    assert Path(retained.model_store.get_blob_file_path(ref.files[0].hash)).read_bytes() == expected
    retained.pull(Namespace(verify=False))
    downloader.assert_not_called()


@dataclass
class Input:
    Model: str


@dataclass
class Expected:
    URLList: list[str] = field(default_factory=lambda: [])
    Names: list[str] = field(default_factory=lambda: [])


@pytest.mark.parametrize(
    "input,expected",
    [
        (Input(""), Expected()),
        (Input("file:///tmp/models/granite-3b-code-base.Q4_K_M.gguf"), Expected()),
        (
            Input(
                "huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF/resolve/main/Q3_K_M/Qwen3-Coder-480B-A35B-Instruct-Q3_K_M-00001-of-00005.gguf"  # noqa: E501
            ),
            Expected(
                URLList=[
                    "https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF/resolve/main/Q3_K_M/Qwen3-Coder-480B-A35B-Instruct-Q3_K_M-00001-of-00005.gguf",  # noqa: E501
                    "https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF/resolve/main/Q3_K_M/Qwen3-Coder-480B-A35B-Instruct-Q3_K_M-00002-of-00005.gguf",  # noqa: E501
                    "https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF/resolve/main/Q3_K_M/Qwen3-Coder-480B-A35B-Instruct-Q3_K_M-00003-of-00005.gguf",  # noqa: E501
                    "https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF/resolve/main/Q3_K_M/Qwen3-Coder-480B-A35B-Instruct-Q3_K_M-00004-of-00005.gguf",  # noqa: E501
                    "https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF/resolve/main/Q3_K_M/Qwen3-Coder-480B-A35B-Instruct-Q3_K_M-00005-of-00005.gguf",  # noqa: E501
                ],
                Names=[
                    "Qwen3-Coder-480B-A35B-Instruct-Q3_K_M-00001-of-00005.gguf",
                    "Qwen3-Coder-480B-A35B-Instruct-Q3_K_M-00002-of-00005.gguf",
                    "Qwen3-Coder-480B-A35B-Instruct-Q3_K_M-00003-of-00005.gguf",
                    "Qwen3-Coder-480B-A35B-Instruct-Q3_K_M-00004-of-00005.gguf",
                    "Qwen3-Coder-480B-A35B-Instruct-Q3_K_M-00005-of-00005.gguf",
                ],
            ),
        ),
    ],
)
def test__assemble_split_file_list(input: Input, expected: Expected):
    # store path and scheme irrelevant here
    model = URL(input.Model, "/store", "https")
    files: list[SnapshotFile] = model._assemble_split_file_list("doesnotmatterhere")
    file_count = len(files)
    assert file_count == len(expected.Names)
    assert file_count == len(expected.URLList)

    for i in range(file_count):
        assert files[i].url == expected.URLList[i]
        assert files[i].name == expected.Names[i]
