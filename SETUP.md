# DataDetective — Setup Guide

Poora project bana hua hai. Yeh guide shuru se aakhir tak har step batati hai.
Har step ke saath likha hai ke **kya dikhna chahiye** — agar wo nahi dikhta, wahin ruk kar
error dekhein.

---

## Step 1 — Database banayein

1. **pgAdmin4** kholein
2. Left panel: `Servers` → `PostgreSQL 16` (password maangega — wohi jo install ke waqt set kiya tha)
3. `Databases` par **right-click** → `Create` → `Database…`
4. Database naam: **`datadetective`** → **Save**

Left panel mein `datadetective` nazar aana chahiye.

> Password bhool gayi hain? PostgreSQL dobara install karna asaan hai — installer chala kar
> naya password set kar lein.

---

## Step 2 — Project folder kholein

Zip ko kisi permanent jagah extract karein, jaise `C:\Projects\datadetective`
(Desktop ya Downloads mein na rakhein).

Us folder ko **VS Code** mein kholein → terminal kholein (`Ctrl` + `` ` ``).

Terminal mein path aisa dikhna chahiye:

```
PS C:\Projects\datadetective>
```

---

## Step 3 — Virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux par: `source venv/bin/activate`

Terminal ke shuru mein **`(venv)`** aa jana chahiye:

```
(venv) PS C:\Projects\datadetective>
```

> Har baar kaam shuru karne se pehle yeh activate karna hai.

> Windows par agar "running scripts is disabled" error aaye, to PowerShell mein chalayein:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` → `Y`

---

## Step 4 — Libraries install karein

```bash
pip install -r requirements.txt
```

2-4 minute lagenge. Aakhir mein `Successfully installed …` aana chahiye.

---

## Step 5 — .env file banayein (yahan password lagta hai)

Project folder mein `.env.example` file hai. Usko **copy** karein aur naya naam dein: **`.env`**

Phir `.env` kholein aur pehli line mein `YOUR_PASSWORD` ki jagah apna Postgres password likhein:

```
DATABASE_URL=postgresql+psycopg2://postgres:mera_password_yahan@localhost:5432/datadetective
```

Baaki sab lines waisi hi rehne dein. `LLM_PROVIDER=mock` ka matlab hai system **bina kisi
API key ke** chalega.

> Password mein `@` `:` `/` jaise characters hon to masla ho sakta hai. Aisi surat mein
> pgAdmin se password simple rakh lein.

---

## Step 6 — Tables banayein

```bash
alembic upgrade head
```

Aakhri line aisi honi chahiye:

```
INFO  [alembic.runtime.migration] Running upgrade  -> xxxxxx, initial schema
```

> Agar `versions/` folder khali ho to pehle yeh chalayein:
> `alembic revision --autogenerate -m "initial schema"`

---

## Step 7 — Server chalayein

```bash
uvicorn app.main:app --reload
```

Yeh aana chahiye:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

**Terminal ko band na karein** — server isi mein chalta rahega.

---

## Step 8 — Verify karein

Browser mein yeh khol kar dekh lein:

| URL | Kya dikhna chahiye |
|---|---|
| http://127.0.0.1:8000/health | `"database": "connected"` |
| http://127.0.0.1:8000/health/tables | `"count": 23` |
| http://127.0.0.1:8000 | DataDetective ka interface |
| http://127.0.0.1:8000/docs | 36 API endpoints |

Chaaron chal gaye — setup mukammal hai.

---

## Step 9 — Chala kar dekhein

**Naya terminal** kholein (pehla server ke liye chal raha hai), `venv` activate karein, phir:

```bash
python scripts/generate_benchmark.py
```

Yeh `benchmarks/` mein 4 test datasets banayega jinka **asal cause pehle se maloom hai**.

Ab browser mein http://127.0.0.1:8000 kholein:

1. **Upload** — `benchmarks/regional_decline.csv` chunein → Upload
2. **Health** tab — health score, quality issues, columns ka type aur meaning
3. **Clean** tab — proposed operations; destructive wale tick karein → Apply
4. **Ask** tab — likhein: *"Why did revenue decline?"* → Investigate
5. **Report** tab — purani investigations

Ask tab ka expected jawab:

> **region = 'South' contributed 99.08% of the total revenue decline**
> (from 50365.28 to 39688.09 per period, -21.2%)

Yeh bilkul wohi cause hai jo benchmark generator ne dataset mein daala tha.

---

## Step 10 — Tests chalayein

```bash
pytest -q
```

`83 passed` aana chahiye.

```bash
python scripts/run_evaluation.py
```

`Root-cause accuracy: 6/6 = 100%` aana chahiye (server chalta rehna chahiye).

### Data quality aur internals (proposal §20)

```bash
python scripts/run_quality_evaluation.py
```

Server ki zaroorat nahi. Profiling ki precision/recall, cleaning correctness, numerical
accuracy, hypothesis relevance, verification accuracy aur statistical test appropriateness
sab score hote hain.

### Architecture comparison (proposal §20)

```bash
python scripts/run_comparison.py --ablations
```

Yeh server ke baghair chalta hai. Baselines A/B/C poori tarah LLM par depend karte hain,
is liye `mock` provider ke saath sirf plumbing test hoti hai — script khud warning deti hai.
Report mein quote karne se pehle asli LLM provider lagana zaroori hai.

Ablations mock par bhi meaningful hain.

---

## Login zaroori hai

`http://127.0.0.1:8000` par pehle login screen aayega. "Create one" se account banayein —
**pehla account admin banta hai**, baad wale analyst.

## Asli LLM lagana (optional)

Abhi `mock` provider chal raha hai — poora system chalta hai, lekin hypotheses aur report ki
zubaan template se aati hai. Asli LLM lagane ke liye `.env` mein:

```
LLM_PROVIDER=gemini
LLM_API_KEY=aapki_key_yahan
EMBEDDING_PROVIDER=gemini
EMBEDDING_API_KEY=aapki_key_yahan
```

Gemini ka free tier FYP ke liye kaafi hai. `anthropic` aur `openai` bhi support hain.
Server restart karein — aur kuch change karne ki zaroorat nahi.

---

## Roz ka kaam shuru karna

```bash
cd C:\Projects\datadetective
venv\Scripts\activate
uvicorn app.main:app --reload
```

---

## Common errors

| Error | Wajah aur hal |
|---|---|
| `connection refused ... port 5432` | PostgreSQL band hai. Windows: `services.msc` → `postgresql-x64-16` → Start |
| `password authentication failed` | `.env` mein password ghalat hai |
| `database "datadetective" does not exist` | Step 1 reh gaya |
| `ModuleNotFoundError` | `venv` activate nahi hai, ya Step 4 reh gaya |
| `Target database is not up to date` | `alembic upgrade head` chalayein |
| `running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Port 8000 already in use | `uvicorn app.main:app --reload --port 8001` |

---

## Docker se chalana (optional)

Agar kabhi Docker install kar lein, to yeh sab kuch ek command mein karta hai:

```bash
docker compose up --build
```

Iske liye `.env` ya pgAdmin ki zaroorat nahi — apna database khud bana leta hai.

---

## Git (zaroori)

```bash
git init
git add .
git commit -m "DataDetective: complete implementation"
```

Phir GitHub par private repo bana kar push kar dein. `.gitignore` already mojood hai —
`.env` aur `storage/` git mein nahi jayenge.
