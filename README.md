# Resume ATS Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) friendliness and gives concrete ways to improve it, powered by Google Gemini Flash models.

## Features

- Upload a resume as PDF, DOCX or TXT
- Optional job description for targeted keyword matching
- Estimated ATS score (0-100) built from five weighted categories:
  keyword match, content quality, formatting and parsability, structure, readability
- Keywords found and missing
- Prioritised improvement list (High / Medium / Low)
- Rewritten example bullet points
- Downloadable Markdown report

> The score is an AI estimate, not the output of any real employer's ATS. Use it as guidance.

## Project structure

```
resume-ats-checker/
├── app.py
├── requirements.txt
└── README.md
```

## Run locally

1. Install Python 3.10 or newer.
2. Create a free API key at https://aistudio.google.com/apikey
3. Set up and run:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

export GEMINI_API_KEY="your-key"   # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

You can also skip the environment variable and paste the key into the sidebar.

### Choosing a model

The sidebar lets you pick a Gemini Flash model. The default can be changed with the `GEMINI_MODEL` environment variable. Model names change over time; check https://ai.google.dev/gemini-api/docs/models and edit `MODEL_OPTIONS` in `app.py` if needed.

## Push to GitHub

Never commit your API key. Create a `.gitignore` first:

```bash
printf "venv/\n__pycache__/\n.env\n.streamlit/secrets.toml\n" > .gitignore
```

Then:

```bash
git init
git add .
git commit -m "Initial commit: Resume ATS Checker"
git branch -M main
```

Create an empty repository on https://github.com/new (no README, since you already have one), then:

```bash
git remote add origin https://github.com/<your-username>/resume-ats-checker.git
git push -u origin main
```

GitHub does not accept account passwords for git pushes. When prompted, use a Personal Access Token (GitHub > Settings > Developer settings > Personal access tokens), or sign in with GitHub CLI (`gh auth login`).

## Deploy on Streamlit Community Cloud (free)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** and choose **Deploy a public app from GitHub**.
3. Select your repository, branch `main`, and main file path `app.py`.
4. Open **Advanced settings** and paste your secret in the Secrets box:

   ```toml
   GEMINI_API_KEY = "your-key"
   ```

5. Click **Deploy**. After a few minutes you get a public `*.streamlit.app` URL.

To update the app later, just `git push` to `main`; it redeploys automatically.

Because your key is stored as a secret, anyone using your public app spends your Gemini quota. To avoid that, either make the app private (invite only) or remove the secret so visitors must enter their own key in the sidebar.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Very little text could be extracted" | The resume is probably a scanned image. Export a text-based PDF or use DOCX. |
| 404 / model not found | Pick another model in the sidebar or update `MODEL_OPTIONS`. |
| 429 / quota errors | You hit the free-tier rate limit. Wait a minute or switch model. |
| Deploy fails on install | Make sure `requirements.txt` is in the repo root. |

## Privacy

Resume text is sent to the Gemini API for analysis. The app itself does not store uploads.
