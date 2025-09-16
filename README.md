### Project Structure
```text
Struct-Syn/
└── Struct-Syn/
    ├── src/
    │   ├── LM/
    │   │   ├── chatgpt.py
    │   │   └── config.json           # Put your LLM API configuration here
    │   └── syn_Agent/
    │       └── [core agent implementation]
    ├── Topic_Summary_Generator.py    # Generate Topic–Summary training datasets
    ├── Topic_Generate.py             # Generate diverse topics
    └── Agent_Caller.py               # Orchestrate and run different synthesis agents
```

### Configuration
- Update `src/LM/config.json` with your API key, base URL, model name, etc.

### Usage
```bash
python Topic_Summary_Generator.py
```

- Generate diverse topics
```bash
python Topic_Generate.py
```

- Run synthesis agents (adjust agent parameters and workflow as needed before running)
```bash
python Agent_Caller.py
```





