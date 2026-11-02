"""Offline contract tests for the Azure OpenAI v1 image adapter."""

import asyncio
import base64
import json
import socket
from email import policy
from email.parser import BytesParser

import httpx
import pytest

from nanobot.agent.tools.image_generation import ImageGenerationTool
from nanobot.agent.tools.image_generation_config import ImageGenerationToolConfig
from nanobot.config.schema import ProvidersConfig
from nanobot.providers import image_generation as images
from nanobot.providers.image_generation import AzureOpenAIImageGenerationClient
from nanobot.providers.registry import find_by_name

BASE = "https://resource.openai.azure.com/openai/v1"
MODEL = "gpt-image-2.5-flare"
# Valid one-pixel PNG, kept inline to avoid downloads or external fixtures.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
)
PAYLOAD = {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("live network is forbidden in these tests")

    async def forbidden_transport(*args, **kwargs):
        forbidden()

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden_transport)


@pytest.mark.parametrize("base", [None, "", "  ", "/"])
def test_missing_base_fails_at_construction_without_openai_fallback(base, monkeypatch):
    def forbidden_fallback(*args):
        raise AssertionError("public OpenAI fallback must not be used")

    monkeypatch.setattr(images.OpenAIImageGenerationClient, "_default_base_url", forbidden_fallback)
    with pytest.raises(images.ImageGenerationError, match="azureOpenai.apiBase"):
        AzureOpenAIImageGenerationClient(api_key="azure-test-key", api_base=base)


@pytest.mark.parametrize("key", [None, ""])
@pytest.mark.parametrize("editing", [False, True])
def test_missing_key_fails_before_network_or_reference_file_access(key, editing):
    async def run():
        def forbidden(request):
            raise AssertionError("HTTP called without a key")

        async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as http:
            client = AzureOpenAIImageGenerationClient(api_key=key, api_base=BASE, client=http)
            with pytest.raises(images.ImageGenerationError, match="azureOpenai.apiKey"):
                await client.generate(
                    prompt="test", model=MODEL,
                    reference_images=["/does-not-exist.png"] if editing else None,
                )

    asyncio.run(run())


@pytest.mark.parametrize(
    "model,aspect,size,expected_size",
    [(MODEL, "1:1", "1K", "1024x1024"),
     (MODEL, "16:9", None, "1536x1024"),
     ("openai/" + MODEL, None, "2048x1024", "2048x1024")],
)
def test_generation_json_and_bearer_auth(model, aspect, size, expected_size):
    async def run():
        requests = []

        def respond(request):
            requests.append(request)
            assert request.method == "POST"
            assert str(request.url) == BASE + "/images/generations"
            assert not request.url.query  # no legacy api-version query
            assert request.headers["authorization"] == "Bearer azure-test-key"
            assert "api-key" not in request.headers
            assert request.headers["content-type"] == "application/json"
            body = json.loads(request.content)
            assert body == {
                "model": MODEL, "prompt": "a tiny lighthouse",
                "size": expected_size, "quality": "high",
            }
            assert "response_format" not in body
            return httpx.Response(200, json=PAYLOAD)

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            client = AzureOpenAIImageGenerationClient(
                api_key="azure-test-key", api_base=BASE + "/", client=http,
                extra_body={"quality": "high"},
            )
            result = await client.generate(
                prompt="a tiny lighthouse", model=model, aspect_ratio=aspect, image_size=size,
            )
        assert len(requests) == 1
        assert result.images == ["data:image/png;base64," + PAYLOAD["data"][0]["b64_json"]]
        assert result.raw == PAYLOAD

    asyncio.run(run())


def test_editing_preserves_inherited_multipart_support(tmp_path):
    refs = [tmp_path / "first.png", tmp_path / "second.png"]
    for path in refs:
        path.write_bytes(PNG)

    async def run():
        requests = []

        def respond(request):
            requests.append(request)
            assert str(request.url) == BASE + "/images/edits"
            assert request.headers["authorization"] == "Bearer azure-test-key"
            assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
            message = BytesParser(policy=policy.default).parsebytes(
                b"Content-Type: " + request.headers["content-type"].encode() + b"\r\n\r\n" + request.content
            )
            fields, files = {}, []
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if part.get_filename():
                    files.append((name, part.get_filename(), part.get_content_type(), part.get_payload(decode=True)))
                else:
                    fields[name] = part.get_payload(decode=True).decode()
            assert fields == {"model": MODEL, "prompt": "combine", "size": "1024x1536", "quality": "high"}
            assert files == [("image[]", path.name, "image/png", PNG) for path in refs]
            return httpx.Response(200, json=PAYLOAD)

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            client = AzureOpenAIImageGenerationClient(
                api_key="azure-test-key", api_base=BASE, client=http, extra_body={"quality": "high"},
            )
            result = await client.generate(
                prompt="combine", model=MODEL, reference_images=[str(path) for path in refs], aspect_ratio="3:4",
            )
        assert len(requests) == 1
        assert result.images[0].startswith("data:image/png;base64,")

    assert AzureOpenAIImageGenerationClient.generate is images.OpenAIImageGenerationClient.generate
    assert AzureOpenAIImageGenerationClient._post_image_edit is images.OpenAIImageGenerationClient._post_image_edit
    asyncio.run(run())


def test_registry_discovery_config_alias_and_tool_wiring(tmp_path):
    original_openai = {"apiKey": "existing-openai-key", "apiBase": "https://api.openai.com/v1"}
    providers = ProvidersConfig.model_validate({
        "openai": original_openai,
        "azureOpenai": {"apiKey": "azure-test-key", "apiBase": BASE, "extraBody": {"quality": "high"}},
    })
    before = providers.openai.model_dump()
    assert ProvidersConfig.model_fields["azure_openai"].alias == "azureOpenai"
    assert find_by_name("azure_openai") is not None
    assert images.get_image_gen_provider("azure_openai", providers.azure_openai) is AzureOpenAIImageGenerationClient
    assert "azure_openai" in images.image_gen_provider_names(providers)
    configs = images.image_gen_provider_configs(providers)
    assert configs["azure_openai"] is providers.azure_openai
    assert images.get_image_gen_provider("openai") is images.OpenAIImageGenerationClient
    assert AzureOpenAIImageGenerationClient.model_options[0] == MODEL
    config = ImageGenerationToolConfig.model_validate({"provider": "azure_openai", "model": MODEL})
    tool = ImageGenerationTool(workspace=tmp_path, config=config, provider_configs=configs)
    client = tool._provider_client()
    assert isinstance(client, AzureOpenAIImageGenerationClient)
    assert client.api_key == "azure-test-key"
    assert client.api_base == BASE
    assert client.extra_body == {"quality": "high"}
    assert providers.openai.model_dump() == before
    dumped = providers.model_dump(by_alias=True)
    assert dumped["azureOpenai"]["apiBase"] == BASE
    assert dumped["openai"]["apiKey"] == original_openai["apiKey"]


def test_azure_registration_is_available_without_manual_registration():
    assert images.get_image_gen_provider("azure_openai") is AzureOpenAIImageGenerationClient
    assert "azure_openai" in images.image_gen_provider_names()
