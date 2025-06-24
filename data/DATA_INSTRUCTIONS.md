
# 📁 DATA_INSTRUCTIONS.md – How to Access the Algonauts 2025 Data

This project uses the **Algonauts 2025 dataset**, which contains large movie and fMRI files. Due to the size of the dataset, we **do not upload data files to GitHub**. Instead, all collaborators must **download the data locally** using [DataLad](https://www.datalad.org/), a tool designed to manage large scientific datasets efficiently.

---

## ⚙️ Step-by-Step Instructions

### 1. ✅ Install DataLad

If you don’t already have it installed:

```bash
pip install datalad
```

Or using `conda`:

```bash
conda install -c conda-forge datalad
```

---

### 2. 📥 Clone the Algonauts Dataset

Open your terminal and run:

```bash
datalad clone https://github.com/algonauts2025/algonauts2025.git algonauts_2025
cd algonauts_2025
```

This creates a lightweight copy of the repository.

---

### 3. 📦 Download the Full Dataset

From within the cloned folder, run:

```bash
datalad get -r -J8 .
```

- `-r` = recursively get all subfolders
- `-J8` = download with 8 parallel jobs for speed

💡 You can also run `datalad get` on specific folders (e.g. `stimuli/` or `participants/`) if you don’t need everything.

---

### 4. 🧠 Use the Data in Your Scripts

All scripts in this project should reference the data **as local file paths** like:

```python
"../algonauts_2025/stimuli/movies/friends/s2/friends_s02e13b.mkv"
```

Make sure your local folder names match this structure.

---

## 📂 Recommended Folder Structure

```
your_project/
│
├── notebooks/
│   └── analysis.ipynb
├── src/
│   └── processing.py
├── algonauts_2025/          ← Cloned & downloaded via DataLad (not tracked in Git)
├── .gitignore
├── README.md
└── DATA_INSTRUCTIONS.md     ← You're reading this
```

---

## 🛡️ Important: Do Not Upload Data to GitHub

Add the following to your `.gitignore`:

```
algonauts_2025/
*.mkv
*.h5
*.nii
```

This prevents large files from being pushed to GitHub.

---
-------------------------


Using Python 3.10

Create virtual environment and install dependencies:

```bash
python -m venv brain-env
```
```bash
python -m pip install --upgrade pip
```

Activate the environement

```bash
# bash
source .brain-env/bin/activate
```
OR

```PowerShell
# PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\brain-env\Scripts\Activate.ps1
```


Then install the package:

```bash
# bash
pip install -e . # run each time multimodal_stimulus_fmri_predict is updated

```

Install the dependencies (optional):

```bash
#bash
pip install -r requirements.txt
```

### How to run the code


### Backup: Pulling Repo

```PowerShell
git status

git pull
```

## Project Roadmap

### Related Work

### Data

### Methodology

### Performance 

### Conclusion

## 👥 Team Members

[Sophie Strassmann] (GitHub Profile)

[Yujie] (GitHub Profile)

[Team Member 3 Name] (GitHub Profile)

[Team Member 4 Name] (GitHub Profile)

## 📝 License
This project is licensed under the [License Name] - see the LICENSE.md file for details.

## 🙏 Acknowledgments
AI4Good Montreal organizers and mentors

[Any other organizations or individuals you want to acknowledge]






