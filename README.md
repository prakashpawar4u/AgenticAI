echo "# AgenticAI" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/prakashpawar4u/AgenticAI.git
git push -u origin main

AgenticAI/
├── agents/
│   └── simple_agent.py
├── rag/
│   └── simple_rag.py
├── utils/
│   └── helpers.py
├── notebooks/
│   └── intro.ipynb
├── requirements.txt
├── README.md
├── .gitignore
└── notes.md
