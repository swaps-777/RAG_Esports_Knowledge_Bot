# REPORT.md

## 1. Assigned Use Case

This project was built for the use case **Gaming Strategy and Esports Knowledge Bot**.

The objective was to create a RAG-based system that can answer questions from PDF documents related to:

- games
- esports strategy
- player roles
- game mechanics
- competitive insights

The original multi-specialist architecture was adapted to this domain using separate nodes for:

- game manuals
- game strategy content
- esports reports and competitive insights

---

## 2. Documents Used

The following PDF documents were indexed and used in the RAG system:

1. **World of Warcraft Classic Manual**  
   Used for class roles, player responsibilities, and manual-based game information.

2. **Warcraft III Manual**  
   Used for game mechanics, heroes, units, and RTS gameplay concepts.

3. **Sid Meier’s Civilization V Quick Start Manual**  
   Used for strategy-game systems and mechanics.

4. **NBA 2K24 Online Manual**  
   Used for sports-game systems, controls, and gameplay modes.

5. **Identifying Strategies of eSports Players at Various Proficiency Levels in League of Legends**  
   Used for player strategy and skill-level differences.

6. **Guide to Esports**  
   Used for esports ecosystem, trade associations, and competitive context.

7. **Esports Industry Report - Pillsbury**  
   Used for esports business and industry structure.

8. **ALGS Official Rules**  
   Used for tournament structure, roster rules, and team responsibilities.

---

## 3. Architecture Used

The RAG system follows a LangGraph-based multi-node flow:

```text
START
  |
understand_question
  |
search_index
  |
  +---> game_manual_specialist --------+
  |                                    |
  +---> game_strategy_specialist ------+---> pick_response_mode
  |                                    |            |
  +---> esports_specialist ------------+      conditional routing
                                               /          \
                                          quick_answer   detailed_answer
                                               |              |
                                              END            END
```

The main nodes are:

- `understand_question`
- `search_index`
- `game_manual_specialist`
- `game_strategy_specialist`
- `esports_specialist`
- `pick_response_mode`
- `quick_answer`
- `detailed_answer`

---

## 4. Baseline Settings

The initial baseline configuration used was:

```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
```

These settings were used as the starting point for testing retrieval and answer quality.

---

## 5. Prompt and Setting Changes

### Prompt Changes

Initially, the prompts were too summary-oriented. Because of that, the system sometimes gave vague answers even when the correct chunk had already been retrieved, especially for table-based content.

The prompts were updated so that the model:

- extracts information from both paragraphs and tables/lists
- lists exact items when available
- avoids replacing factual entries with generic summaries
- stays grounded in retrieved context only
- clearly says when the retrieved context does not contain the answer

This improved the answer quality for structured content such as tables in the **Guide to Esports** PDF.

### Settings Tested

#### Baseline

```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
```

#### Experiment 2: Retrieval Tuning

```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 6
```

#### Experiment 3: Chunking Tuning

```python
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
TOP_K = 4
```

---

## 6. Main Test Questions

The following three questions were used as the main domain-specific tests:

1. **What are the main trade associations listed in the Guide to Esports?**
2. **According to the World of Warcraft Classic Manual, what is the role of the Paladin class?**
3. **How many ranks does League of Legends have?**

---

## 7. Outputs / Screenshots

### Test Question 1

**Question:**  
What are the main trade associations listed in the Guide to Esports?

**Expected Output:**  
The system should list the trade associations from the table in the Guide to Esports.

**Final Output Summary:**  
After prompt tuning, the system correctly extracted and listed:

- Entertainment Software Association - ESA
- Entertainment Software Association of Canada - ESAC
- Interactive Games and Entertainment Association - IGEA
- Interactive Software Federation of Europe - ISFE
- Korea Association of Game Industry - K-Games

**Result:** Pass

**Screenshot:**  
_Add screenshot here_

---

### Test Question 2

**Question:**  
According to the World of Warcraft Classic Manual, what is the role of the Paladin class?

**Final Output Summary:**  
The system correctly described the Paladin as a hybrid class with defensive, healing, and support abilities.

**Result:** Pass

**Screenshot:**  
_Add screenshot here_

---

### Test Question 3

**Question:**  
How many ranks does League of Legends have?

**Final Output Summary:**  
The system correctly answered that League of Legends has 9 ranks:

- Iron
- Bronze
- Silver
- Gold
- Platinum
- Diamond
- Master
- Grandmaster
- Challenger

**Result:** Pass

**Screenshot:**  
_Add screenshot here_

---

## 8. Experiment Summary

### Experiment 1: Prompt / Generation Tuning

**Configuration**

```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
```

**Question:**  
What are the main trade associations listed in the Guide to Esports?

**Result Before Prompt Tuning:** Fail

**Observation:**  
The correct chunk was retrieved, but the answer gave a vague summary instead of listing the exact table entries.

**Tuning Applied:**  
The prompts were updated to explicitly extract information from both paragraphs and tables/lists, and to list exact entries when available.

**Result After Prompt Tuning:** Pass

**Observation:**  
The system correctly listed the trade associations from the retrieved table.

**Summary:**  
The issue was mainly in answer generation, not retrieval. Prompt tuning improved factual grounding and made the system more reliable for table-based PDF content.

---

### Experiment 2: Retrieval Tuning

**Configuration**

```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 6
```

**Question 1:**  
According to the World of Warcraft Classic Manual, what kinds of player roles or class-based responsibilities exist?

**Result:** Partial Pass

**Observation:**  
The answer was grounded and relevant, but it only covered some classes such as:

- Druid
- Hunter
- Mage
- Warrior
- Warlock

It missed other classes such as:

- Paladin
- Priest
- Rogue
- Shaman

**Question 2:**  
According to the World of Warcraft Classic Manual, what is the role of the Paladin class?

**Result:** Pass

**Observation:**  
The answer was specific, grounded, and correctly described the Paladin as a hybrid class with defensive, healing, and support abilities.

**Summary:**  
Increasing `TOP_K` improved retrieval breadth, but broad questions still did not guarantee full coverage. Narrow questions worked better because retrieval could focus on the exact class mentioned.

---

### Experiment 3: Chunking Tuning

**Configuration**

```python
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
TOP_K = 4
```

**Question 1:**  
Games were scored using what types of strategies?

**Result:** Fail

**Observation:**  
The document contained the exact four strategy types:

- tactical
- operational
- logistic
- political

However, the system returned a generic strategy summary instead of the correct categories.

**Question 2:**  
How many ranks does League of Legends have?

**Result:** Pass

**Observation:**  
The answer was direct, specific, and complete.

**Question 3:**  
What is the projected audience reach of esports by 2022?

**Result:** Fail

**Observation:**  
The exact answer existed in the PDF, but the relevant chunk was not retrieved. The correct figure was **557 million**, but the system did not pick the page containing this information.

**Question 4:**  
During the 2019 League of Legends World Championship tournament, the competing teams had agreements with how many brands?

**Result:** Pass

**Observation:**  
The answer was specific and factually correct. The system answered that the 24 competing teams had agreements with **87 distinct brands**.

**Summary:**  
The smaller chunking setup worked for some narrow factual questions, but missed other exact facts that were present in the PDFs. Overall, it was less reliable than the baseline for this corpus.

---

## 9. What Worked Well

- The system worked well for specific and focused questions.
- Prompt tuning improved extraction from table-based content.
- Questions related to a single class, rule, or fact gave better answers.
- The gaming/esports specialist structure helped organize responses effectively.
- The system correctly handled paragraph-based and table-based content after prompt improvements.
- The system was able to cite relevant sources in the final answer.

---

## 10. What Did Not Work Well

- Broad questions sometimes gave incomplete answers.
- Some exact facts in the PDFs were missed because the relevant chunk was not always retrieved.
- Smaller chunk settings did not improve the overall performance for this document set.
- Table-heavy content required better prompt design to be handled properly.
- Top-k similarity retrieval did not always guarantee full coverage for wide-scope questions.

---

## 11. Final Chosen Settings

Based on the experiments, the best-performing setup for this corpus was:

```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
```

This configuration gave the most balanced performance across:

- table extraction
- paragraph-based answers
- class/role questions
- esports ecosystem questions
- source-grounded responses

---

## 12. Final Conclusion

This project successfully adapted a RAG architecture to the **Gaming Strategy and Esports Knowledge Bot** use case.

The most important improvement came from prompt tuning, especially for handling both paragraph text and table-based content.

The experiments showed that:

- retrieval quality strongly affects answer completeness
- prompt design strongly affects how well the model uses retrieved content
- smaller chunks did not always improve answer quality
- broad questions require better retrieval coverage
- focused questions produce stronger grounded answers

Overall, the RAG system performed well for grounded gaming and esports questions. The main area for future improvement is improving retrieval coverage for broad questions and exact facts that may appear in isolated chunks or table-heavy PDF pages.
