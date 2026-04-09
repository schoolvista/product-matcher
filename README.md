```powershell
python replacement_llm.py
python replacement_llm.py --no-prompt --source-brand SAL -c SKU --target-brand Haneco
python replacement_llm.py -i input.xlsx -o out.xlsx --source-brand SAL --target-brand Haneco
streamlit run app.py
```

Streamlit Cloud: main file `app.py`; Secrets — `ANTHROPIC_API_KEY`, `TAVILY_API_KEY` (optional: `ANTHROPIC_MODEL`, `TAVILY_SEARCH_DEPTH`, `TAVILY_TARGET_DOMAINS`).
