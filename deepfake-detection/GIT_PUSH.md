# Pushing the Deepfake Layer to the Team Monorepo

Run in **PowerShell** on your machine. Replace `<TEAM_REPO_URL>` with your repo's URL.

## 1. Clone the team repo (skip if you already have it)
```powershell
cd "C:\Users\Harshini J\Engineering\Projects\Unisys"
git clone <TEAM_REPO_URL> team-repo
cd team-repo
git checkout main
git pull
```

## 2. Check the folder-naming convention
Look at how the other layers are named so yours matches:
```powershell
dir
```
e.g. if you see `liveness/` and `face_verification/`, name yours `deepfake/`.

## 3. Create a feature branch (don't push straight to main)
```powershell
git checkout -b add-deepfake-layer
```

## 4. Copy ONLY the code in (excludes dataset, weights, caches)
`robocopy` with `/XD` skips the big folders so you never copy the 4 GB dataset:
```powershell
robocopy "C:\Users\Harshini J\Engineering\Projects\Unisys\deepfake_layer" ".\deepfake" /E /XD data weights __pycache__ .git
```
This copies your `.py` files, `demo/`, `.gitignore`, and READMEs into `team-repo\deepfake\`.

## 5. Verify what git will commit — IMPORTANT
```powershell
git add deepfake
git status
```
Confirm the list does **NOT** include anything under `deepfake/data/`, any `.pth`
file, or `__pycache__`. If it does, stop — the `.gitignore` didn't apply (see
Troubleshooting below) before committing.

## 6. Commit and push
```powershell
git commit -m "Add deepfake detection layer (Stage 2): visual stream, scripts, demo"
git push -u origin add-deepfake-layer
```

## 7. Open a Pull Request
GitHub will print a PR link after the push, or go to the repo on github.com and
click **Compare & pull request** from `add-deepfake-layer` into `main`. Let your
teammates review, then merge.

---

## Troubleshooting

- **A `.pth` or `data/` file shows up in `git status`:** the `.gitignore` wasn't
  copied or git already tracked it. Make sure `deepfake/.gitignore` exists, then:
  ```powershell
  git rm -r --cached deepfake/weights deepfake/data
  git add deepfake/.gitignore
  git status
  ```
- **`.gitignore` seems missing after robocopy:** it's a hidden dotfile; confirm with
  `dir .\deepfake\ -Force`. If absent, copy it explicitly:
  `copy "C:\Users\Harshini J\Engineering\Projects\Unisys\deepfake_layer\.gitignore" ".\deepfake\"`
- **Git asks for login / push rejected:** you may need a Personal Access Token
  (GitHub no longer accepts passwords). Create one at github.com → Settings →
  Developer settings → Personal access tokens, and use it as the password, or set
  up the GitHub CLI (`gh auth login`).
- **Sharing the trained model:** don't commit the 68 MB `.pth`. Attach it to a
  GitHub Release, or use Git LFS (`git lfs track "*.pth"`) if the team wants it
  versioned.
- **The demo's `__pycache__`:** already ignored; safe to leave.
```
