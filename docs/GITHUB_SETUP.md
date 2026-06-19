# GitHub Repository Setup

Follow these steps to publish ShopMate for your assignment submission.

## 1. Initialize and commit

```bash
cd shopmate-agent
git init
git add .
git commit -m "Initial ShopMate agentic support agent implementation"
```

## 2. Create GitHub repository

1. Go to https://github.com/new
2. Name: `shopmate-agent` (or your preferred name)
3. Set visibility to **Public** (or share with module coordinator)
4. Do **not** initialize with README (this project already has one)

## 3. Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/shopmate-agent.git
git branch -M main
git push -u origin main
```

## 4. Link in your report

Include this URL in your PDF report:

```
https://github.com/YOUR_USERNAME/shopmate-agent
```

## 5. Before pushing — checklist

- [ ] `.env` is **not** committed (listed in `.gitignore`)
- [ ] `OPENAI_API_KEY` is only in local `.env`
- [ ] `README.md` has setup instructions
- [ ] Demo video uploaded (YouTube unlisted or Google Drive) and linked in README/report

## 6. Optional: add eval results

After running evaluation with your API key:

```bash
python eval/run_eval.py --mode all
git add eval/results/
git commit -m "Add evaluation results"
git push
```
