# Azure OpenAI image generation

The `azure_openai` image provider uses Azure OpenAI's v1 Images API. It reuses the OpenAI generation and editing implementation with a separate Azure connection. It does not change chat authentication or the existing `providers.openai` connection.

Create an Azure OpenAI resource and a supported image-model deployment before configuring nanobot. Model availability, access and quotas depend on your Azure subscription and region; the suggested model names are not an entitlement guarantee. Select a consumption deployment such as Global Standard rather than dedicated GPU compute when that matches your workload.

Merge these fields into your existing configuration; do not replace the entire configuration with this example:

```json
{
  "providers": {
    "azureOpenai": {
      "apiKey": "<Azure resource key>",
      "apiBase": "https://RESOURCE.openai.azure.com/openai/v1"
    }
  },
  "tools": {
    "imageGeneration": {
      "enabled": true,
      "provider": "azure_openai",
      "model": "gpt-image-2.5-flare",
      "defaultAspectRatio": "1:1",
      "defaultImageSize": "1K",
      "extraBody": {
        "n": 1,
        "quality": "medium",
        "output_format": "png"
      }
    }
  }
}
```

The `model` is the actual Azure deployment name. Keep that name prefixed with `gpt-image` and use it without an `azure_openai/` prefix: inherited model-family detection currently relies on that naming convention. Arbitrary aliases such as `my-images` are not supported by this minimal adapter. Reference-image editing uses the same Azure connection and inherited multipart Images API support.

The base must explicitly end in `/openai/v1`; do not use a legacy `/openai/deployments/...` base or add an `api-version` query. The adapter sends the resource key as a Bearer token. Missing or blank bases fail closed instead of falling back to the public OpenAI endpoint. Explicit bases and extra headers remain trusted configuration; this adapter does not validate an Azure hostname allowlist or refresh Entra ID credentials.

Keep keys in protected local configuration, never in source control. Use explicit size and quality for predictable costs, verify actual usage after an authorized generation, and preserve subscription spending protections. Budget alerts do not impose a hard spending cap. Visual Studio credits have individual dev/test restrictions; verify eligibility for your intended workload and selected Azure meter.

Azure v1 guide: https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/dall-e . Pricing: https://azure.microsoft.com/en-us/pricing/details/azure-openai/ .
