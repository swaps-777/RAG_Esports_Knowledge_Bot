# RAG AI Agent

A simple, student-friendly Retrieval-Augmented Generation (RAG) project built with **LangGraph**, **ChromaDB**, **Pydantic**, and **local HuggingFace embeddings**.

The current version is designed as a **RAG Esports Knowledge Bot**. It uses documents about game manuals, mechanics, strategic play, tactics, tournament structure and team roles guides to teach students how an agentic LangGraph workflow can search a vector database, run parallel specialist nodes, and produce grounded answers.

## What This Project Teaches

This project is split into three clear parts:

1. `ingestion.py`
   Turns PDF files into a searchable vector database.
2. `rag_agent.py`
   Uses a LangGraph workflow with retrieval, parallel specialists, and routing.
3. `main.py`
   Runs a simple interactive CLI so users can ask questions.

## Architecture

```text
PDF files
  -> ingestion.py
  -> chunks + embeddings
  -> Chroma vector database

User question
  -> LangGraph RAG workflow
  -> search_index retrieves relevant chunks
  -> game_manual, game_strategy, and esports specialists run in parallel
  -> planner chooses quick or detailed response
  -> final response with sources
```

## Project Flow

### 1. Ingestion Phase

`ingestion.py` prepares the knowledge base.

It does four things:

1. Load PDF files from `data/`
2. Split PDF pages into smaller chunks
3. Create embeddings for those chunks using a local HuggingFace model
4. Save the chunks and embeddings in `chroma_db/`

Important idea for students:

Ingestion is like preparing a library before the assistant can answer questions.

### 2. Esports Knowledge Agent Phase

`rag_agent.py` now defines a more agentic LangGraph workflow that matches a classroom-friendly pattern:

```text
START
  -> understand_question
  -> search_index
  -> game_manual_speciaist
  -> game_strategy_specialist
  -> esports_specialist
  -> pick_response_mode
  -> quick_answer OR detailed_answer
  -> END
```

This graph teaches several LangGraph ideas clearly:

1. `understand_question`
   Interprets what the user is asking before retrieval
2. `search_index`
   Explicitly searches the Chroma vector database
3. `game_manual_specialist`, `game_strategy_specialist`, `esports_specialist`
   Three parallel nodes that each interpret the same retrieved context from a different angle
4. `pick_response_mode`
   A fan-in decision node that chooses whether the final answer should be quick or detailed
5. `quick_answer` / `detailed_answer`
   Conditional routes that show students how LangGraph can branch based on state

### 3. Application Phase

`main.py` is the interactive command-line app.

It:

1. Checks whether the vector database exists
2. Starts a question-answer loop
3. Sends each user question through the LangGraph RAG workflow

## Project Structure

```text
RAG_AI_Agent/
|-- main.py            # Interactive CLI app
|-- ingestion.py       # PDF loading, chunking, embeddings, Chroma storage
|-- rag_agent.py       # Esports Knowledge LangGraph workflow with parallel nodes
|-- requirements.txt   # Python dependencies
|-- .env.example       # Environment variable template
|-- data/              # Put your PDF files here
`-- chroma_db/         # Local vector database created after ingestion
```

## Setup

### 1. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Create your `.env` file

On Windows PowerShell:

```powershell
copy .env.example .env
```

Then open `.env` and add your `OPENAI_API_KEY`.

Note:

- The LLM answer generation uses OpenAI.
- The embedding model is local HuggingFace, so embeddings do not need an OpenAI embedding API call.

### 4. Add PDFs

Put your PDF files into the `data/` folder.

## How To Run

### Step 1: Build the vector database

```powershell
python ingestion.py
```

This reads PDFs from `data/` and creates the local Chroma vector database in `chroma_db/`.

### Step 2: Start the RAG app

```powershell
python main.py
```

Then ask questions such as:

```text
What are the main trade associations listed in the Guide to Esports?
According to the World of Warcraft Classic Manual, what is the role of the Paladin class?
How many ranks does League of Legends have?
What does the ALGS rules document say about team roster structure?
According to the Warcraft III Manual, what are heroes?
```

### Optional: Run the agent file directly

If you want to test only the LangGraph agent without the chat loop:

```powershell
python rag_agent.py
```

To exit:

```text
quit
```

## Rebuilding the Vector Database

If you change the PDFs or change chunking settings in `ingestion.py`, rebuild the vector database.

On Windows PowerShell:

```powershell
Remove-Item -Recurse -Force .\chroma_db
python ingestion.py
```

Then run:

```powershell
python main.py
```

## Student-Friendly Customization Points

Students can safely experiment with these values first:

In `ingestion.py`:

- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `EMBEDDING_MODEL`

In `rag_agent.py`:

- `TOP_K`
- `LLM_MODEL`
- `TEMPERATURE`
- prompts inside `understand_question`
- prompts inside the specialist nodes
- routing logic in `pick_response_mode`

These are good starting points for assignments because they let students see how retrieval and generation behavior changes without needing to redesign the whole project.

## Suggested Learning Path

1. Read `ingestion.py` to understand how documents become searchable.
2. Read `rag_agent.py` to understand the LangGraph workflow.
3. Run `python ingestion.py`
4. Run `python rag_agent.py`
5. Run `python main.py`
6. Change one setting at a time and observe the result.

## Current LangGraph Design

This project now uses LangGraph in a more agentic but still teachable way.

The current graph is:

```text
START
  -> understand_question
  -> search_index
  -> game_manual_specialist
  -> game_strategy_specialist
  -> esports_specialist
  -> pick_response_mode
  -> quick_answer OR detailed_answer
  -> END
```

That makes it a strong base for future student extensions such as:

- adding a router node
- adding query rewriting
- adding answer checking
- adding conversation memory
- switching retrieval strategies
- adding more specialist branches
- using different routing rules for beginner vs advanced fitness users

## Quick Start Commands

If you just want the minimum commands on Windows PowerShell:

```powershell
cd "c:\Users\swapnil-pc\Documents\AI Builder 3\Projects\RAG_AI_Agent"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python ingestion.py
python main.py
```
