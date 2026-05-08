# model

Setup venv with uv

```bash
uv venv .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
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
