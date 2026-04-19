# RAG AI Agent - Student Assignments

This assignment uses the same codebase for everyone:

- `ingestion.py` for document loading, chunking, and Chroma vector storage
- `rag_agent.py` for the LangGraph workflow
- `main.py` for the interactive CLI

Each student will keep the same project structure, but adapt the agent to a different real-world use case by:

1. Choosing and uploading their own PDF documents into `data/`
2. Updating the prompts and wording inside `rag_agent.py` to match their assigned domain
3. Testing the project with domain-specific questions
4. Pushing their final work to their own Git repository

## Common Instructions For Everyone

### 1. Clone or fork the project

Repository:

`https://github.com/NisargKadam/RAG_AI_Agent_Assignment`

### 2. Set up the environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Then add your `OPENAI_API_KEY` inside `.env`.

### 3. Add your own documents

Every student must choose their own PDF documents for their assigned domain.

Rules:

- Upload at least 2 PDF files
- Use documents that genuinely match your assigned use case
- Place them inside the `data/` folder
- You may use public PDFs, course material, reports, manuals, guides, or your own compiled PDFs

### 4. Run the project

```powershell
python ingestion.py
python main.py
```

Optional direct agent test:

```powershell
python rag_agent.py
```

### 5. What students are allowed to change

You should keep the overall project structure the same, but you are encouraged to modify:

- prompts inside `rag_agent.py`
- `TOP_K`
- `LLM_MODEL`
- `TEMPERATURE`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- answer style and wording

You should not rewrite the whole project into a different architecture.

## Submission Requirements

Each student must submit:

1. A link to their GitHub repository
2. Their updated code
3. Their selected PDF documents or document source links
4. A `REPORT.md` containing:
   - assigned use case
   - what documents they used
   - what they changed in the prompts or settings
   - 3 test questions
   - outputs or screenshots
   - what worked well and what did not
5. At least 2 screenshots of the running project

## Student Assignments

### 1. Sanjay Kumar Shaswani

**Use Case:** Personal Finance Education Assistant

Use this project to answer questions from PDF documents about budgeting, saving, debt management, and beginner financial literacy.

**Suggested document ideas:**
- personal finance guides
- banking literacy PDFs
- financial planning booklets

### 2. somya jain

**Use Case:** Nutrition and Healthy Eating Guide

Use this project to answer questions from PDF documents about balanced diets, meal planning, nutrients, and healthy food habits.

**Suggested document ideas:**
- nutrition guides
- diet planning PDFs
- healthy eating manuals

### 3. Rohan Sawant

**Use Case:** Travel Planning Assistant

Use this project to answer questions from PDF documents about destinations, itineraries, travel tips, safety, and budgeting.

**Suggested document ideas:**
- travel guides
- tourism brochures
- itinerary planning PDFs

### 4. Shiva Kumar

**Use Case:** Cybersecurity Awareness Bot

Use this project to answer questions from PDF documents about online safety, passwords, phishing, malware, and security best practices.

**Suggested document ideas:**
- cybersecurity handbooks
- awareness PDFs
- OWASP or NIST guides

### 5. Keerthana S

**Use Case:** Mental Wellness and Stress Support Assistant

Use this project to answer questions from PDF documents about stress management, mindfulness, emotional health, and self-care strategies.

**Suggested document ideas:**
- wellness guides
- mental health awareness PDFs
- stress management resources

### 6. Shubham Gotal

**Use Case:** Programming Concepts Tutor

Use this project to answer questions from PDF documents about programming basics, Python concepts, data structures, and beginner coding practice.

**Suggested document ideas:**
- programming notes
- Python tutorials
- CS learning PDFs

### 7. Lipsha Jena

**Use Case:** Environmental Awareness Assistant

Use this project to answer questions from PDF documents about climate change, pollution, sustainability, conservation, and eco-friendly practices.

**Suggested document ideas:**
- climate reports
- sustainability guides
- environmental education PDFs

### 8. Ramesh Kola

**Use Case:** Agriculture and Farming Knowledge Bot

Use this project to answer questions from PDF documents about crops, soil care, fertilizers, irrigation, and farming techniques.

**Suggested document ideas:**
- agricultural guides
- crop management PDFs
- farming extension documents

### 9. Nutan Mahale

**Use Case:** Education Policy and Curriculum Assistant

Use this project to answer questions from PDF documents about school policy, curriculum frameworks, teaching approaches, and education reform.

**Suggested document ideas:**
- curriculum documents
- policy PDFs
- education framework reports

### 10. Subramanian P

**Use Case:** Automotive Maintenance Assistant

Use this project to answer questions from PDF documents about vehicle maintenance, servicing, troubleshooting, and car care.

**Suggested document ideas:**
- vehicle manuals
- maintenance guides
- repair knowledge PDFs

### 11. Julie A

**Use Case:** Recipe and Cooking Assistant

Use this project to answer questions from PDF documents about recipes, ingredients, cooking methods, substitutions, and meal preparation.

**Suggested document ideas:**
- cookbooks
- recipe PDFs
- culinary guides

### 12. Sanjana Narkar

**Use Case:** Psychology Learning Assistant

Use this project to answer questions from PDF documents about psychology concepts, behavior, cognition, emotions, and mental processes.

**Suggested document ideas:**
- psychology notes
- textbook chapters
- academic psychology PDFs

### 13. Zohra Lanewala

**Use Case:** HR Policy and Workplace Assistant

Use this project to answer questions from PDF documents about employee policies, leave rules, conduct guidelines, and workplace processes.

**Suggested document ideas:**
- employee handbooks
- HR policy manuals
- workplace rulebooks

### 14. Ashirvad Gandham

**Use Case:** Sports Analytics and Training Bot

Use this project to answer questions from PDF documents about sports performance, athlete training, match analysis, and sports science.

**Suggested document ideas:**
- sports science PDFs
- player analysis reports
- training manuals

### 15. Jignesh Shah

**Use Case:** Real Estate Information Assistant

Use this project to answer questions from PDF documents about housing, property buying, renting, pricing, and legal checklists.

**Suggested document ideas:**
- property guides
- housing market reports
- real estate information PDFs

### 16. S M Shohan

**Use Case:** Legal Document Reader

Use this project to answer questions from PDF documents about contracts, terms, rights, policies, and legal language in simpler form.

**Suggested document ideas:**
- agreements
- policy documents
- legal awareness PDFs

### 17. Reddy Rani Ayyappaneni

**Use Case:** Women's Health Education Assistant

Use this project to answer questions from PDF documents about women's wellness, preventive care, nutrition, and health awareness.

**Suggested document ideas:**
- public health PDFs
- women's health guides
- awareness booklets

### 18. Avanish Tiwari

**Use Case:** Startup and Entrepreneurship Knowledge Bot

Use this project to answer questions from PDF documents about startups, business planning, idea validation, funding, and entrepreneurship basics.

**Suggested document ideas:**
- startup guides
- entrepreneurship manuals
- business planning PDFs

### 19. Prem K Sundar

**Use Case:** Product Manual and Customer Support Agent

Use this project to answer questions from PDF documents about how to use a product, troubleshoot it, and explain setup steps clearly.

**Suggested document ideas:**
- device manuals
- appliance guides
- user documentation PDFs

### 20. Rajavel Thiruvu

**Use Case:** History and Culture Explorer

Use this project to answer questions from PDF documents about historical events, civilizations, cultural heritage, and important timelines.

**Suggested document ideas:**
- history book chapters
- museum PDFs
- culture and heritage resources

### 21. Pranoti Meshram

**Use Case:** Research Paper Explainer

Use this project to answer questions from PDF documents about academic papers, research findings, methods, and conclusions in simpler language.

**Suggested document ideas:**
- research papers
- journal articles
- conference PDFs

### 22. Naveen

**Use Case:** Government Scheme Information Assistant

Use this project to answer questions from PDF documents about public schemes, eligibility, benefits, application process, and citizen services.

**Suggested document ideas:**
- government scheme PDFs
- policy brochures
- official citizen guides

### 23. Swapnil Jadhav

**Use Case:** Gaming Strategy and Esports Knowledge Bot

Use this project to answer questions from PDF documents about games, esports strategy, player roles, mechanics, and competitive insights.

**Suggested document ideas:**
- esports reports
- game manuals
- strategy guides

## Recommended Task For Every Student

Every student should complete the following work using their own assigned use case:

1. Add their own PDFs to `data/`
2. Run `python ingestion.py`
3. Run `python main.py`
4. Update the prompts in `rag_agent.py` so the assistant matches their domain
5. Test at least 3 domain-specific questions
6. Tune at least 2 settings:
   - one retrieval or generation setting
   - one chunking setting
7. Document the impact in `REPORT.md`
8. Push the final project to their own GitHub repository

## Grading Rubric

| Criteria | Points |
|----------|--------|
| Project runs successfully | 20 |
| Documents chosen well for the assigned use case | 15 |
| Prompt and agent behavior adapted to the domain | 20 |
| Good testing with meaningful questions | 15 |
| REPORT.md quality | 15 |
| GitHub submission and screenshots | 15 |
| **Total** | **100** |

## Tips For Students

- Choose documents that actually contain the answers you want the agent to give
- Do not upload random PDFs just to fill the folder
- If you change chunk settings, delete `chroma_db/` and run ingestion again
- Test simple questions first, then more detailed ones
- Keep your modifications focused and explain why you made them
- Use the same codebase structure and make your changes clearly understandable
