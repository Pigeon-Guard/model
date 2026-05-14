# model

Setup venv with uv

```bash
uv venv .venv
.venv\Scripts\activate
uv sync
```

Set credentials in DVC config file

```bash
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
```

Pull dataset

```bash
dvc pull 
```

Generate labels with [notebook](dataset\generate_txt_labels.ipynb)

Train the yolo model with [notebook](ultralytics_yolo.ipynb)

## Configuring the Pipeline

#### 1. Create a GitHub Fine Grained Personal Access Token

The required permissions to trigger a `repo dispatch` are `Read and write` to `Contents` of the repo.

If working in an organization
- The resource owner must be the organization
- The expiration must conform with the maximum lifetime setting in the organization 
- The organization's settings must allow access to the repos through fine-grained tokens

> You can find these settings at \<Your Organization\> -> Settings -> Third-party Access -> Personal Access Tokens

#### 2. Add Token as WANDB Secret

In your wandb account, go to Team Settings -> Team Secrets and add the previously created token.

#### 3. Create Webhook

In Team Settings -> Webhooks, create a new webhook with URL `https://api.github.com/repos/<org>/<repo_name>/dispatches` and the previously added secret as Access Token

#### 4. Create Automation

In the wandb registry, create an automation:
- For the trigger event use "an artifact alias is added" and set it to `production`
- Trigger the previously created webhook
- The payload expects two keys: `event_type` and `client_payload`, like so:
```json
{
  "event_type": "new-production",
  "client_payload": {
    "artifact_collection_name": "${artifact_collection_name}",
    "artifact_version": "${artifact_version_string}"
  }
}
```

#### 5. Configure Secrets for the GitHub Actions Workflow

The following secrets are required in \<Your Repo\> -> Settings -> Security and Quality -> Secrets and Variables -> Actions:

1. WANDB_API_KEY: Your wandb api login key to be able to fetch the model artifact from wandb.
2. GH_PAT: The same token created earlier to be able to publish a release when working in an organization.

Important note: the workflow trigger must match the payload:
```yaml
on:
  repository_dispatch:
    types: [new-production]
```
