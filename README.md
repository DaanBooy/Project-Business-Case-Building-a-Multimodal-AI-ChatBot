# Project-Business-Case-Building-a-Multimodal-AI-ChatBot

The goal of the project is to develop a RAG system/AI bot that can answer questions about a chosen subject. The bot should be able to retrieve information out of a self made database, use tools and have session memory. The Bot also needs to be trackable and deployed, and allow for speech to text input and text to speech output.

I chose to make a chatbot that can assist in growing your own fruits and vegetables.  
Which I named: 🌱 GrowGuide 🌱

Full project guidelines can be found here 👇  
https://github.com/ironhack-labs/project-3-business-case-multimodal-ai-chatbot-for-yt-video-qa

Skip to deployment directly here 👇  
https://huggingface.co/spaces/DaanBooy/GrowGuide

Could be that this space has been set to private in the future to avoid token usage. If so and you would still like to try it out, then feel free to contact me.

## 📋 Approach

 - **Collected data from websites and articles to make dataset.**
 - **Cleaned, Segmented, chunked and embedded the data.**
 - **Used ChromaDB for vector storage.**
 - **Created QA agent chain using gpt-4o-mini, with multiple tools(see below).**
 - **Used LLM as judge to evaluate agent performance.**
 - **Deployed agent on HuggingFace Spaces, with speech to text input and text to speech output.**

 ## 🛠️ Features/Tools

 - **Self Query Retrieving:** Using LLM (gpt-4o-mini) to retrieve documents from vector storage. Resulting in more relevant context.
 - **Memory:** Session memory, allows agent to consider past messages when creating new responses or when using tools.
 - **USDA Hardiness Zone Lookup:** Agent can lookup the USDA Hardiness zone (climate zone system) of your city to personalize advise.
 - **PDF Planning Generation:** A personlized month by month PDF planning can be requested, which will be output as a downloadable link.


## 📊 Evaluation
LLM as judge results, scores from 1 to 10. (mean score of 10 test questions w/ 3 repeats)
- **Correctness, (factual accuracy):** 9.73 
- **Groundedness, (supported by the provided context):** 8.8 
- **Relevance, (addresses the question):** 9.87 
- **Completeness, (covers the key steps/details expected):** 8.53 
- **Conciseness, (clear and not verbose):** 8.87 
- **Consistency, (similarity of repeats):** 8.1  

- **Low cost**, $0.0001 to $0.0005 per interaction
- **Response time**, most outputs are returned between 8 to 15 seconds. But can occasionally be up to 30-40 seconds.


## 🤗 Deployment on HuggingFace

🌱 GrowGuide — Your assistant for growing fruits and vegetables at home! 🌱

Your assistant for growing fruits and vegetables at home. Ask about seeding, soil conditions, harvesting, or anything that comes to mind about growing your own crops.
Want more personalized advice? Tell the chatbot what city you live in and it will find your USDA hardiness climate zone and adjust the advice accordingly.
Don’t know how to start, or when you should do certain things? It can also create a planning for you! Simply request a PDF planning and it will create an easy, personalized plan for you to follow.

👉 https://huggingface.co/spaces/DaanBooy/GrowGuide 👈

Could be that this space has been set to private in the future to avoid token usage. If so and you would still like to try it out, then feel free to contact me.

## ⚙️ How to Run & Reproduce Results
**Clone the repository**

```bash
git clone <your-repo-url>
cd <repo-folder>
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Data Preparation**  
Run `data_prep.ipynb` to preprocess the data set and create the vector storage. Can be skipped if by using `/chroma_growguide/` directly

**QA Chain**  
Run `QA_model` to load vector storage, make tools, wrap in to agent, test and evaluate.

**Deployment**  
Create a HuggingFace Space and upload `agent.py`, `app.py` and `requirements.txt`.
Vector storage includes binary, make sure this is tracked with git lfs when pushing to Space.
API keys and variables need the be added into the settings of the Space. Set keys as Secrets, and variables as Variables. See notes.

**Notes:**  
Everything can be ran in VSCode.  
To run `QA_model`, it could be that you run in to some errors. There are some debugging cells that can be uncommented. Which should fix these errors.
Make sure you create a `.env` file in your project folder containing your keys and variables. You should have:
```
OPENAI_API_KEY = your_api_key_here
SERPAPI_API_KEY = your_api_key_here
LANGSMITH_API_KEY = your_api_key_here

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT= your_project_name_here
```

---

## 📂 Repository Contents

| File Name | Description |
:-----------:|:-------------:|
| `requirements.txt` | Lists all Python dependencies needed to run and reproduce the entire project, also used in app deployment |
| `dataset.txt` | The dataset that was made and used in this project |
| `/chroma_growguide/` | The created vector storage |
| `data_prep.ipynb` | Used to prepare data and to make the Chromadb vector storage |
| `QA_model_v2.ipynb` | Notebook with full QA chatbot chain. Includes tools, testing and evaluation |
| `agent.py` | QA chain converted to a py file so it can be used in deployment |
| `app.py` | Gradio app file that uses agent.py, was used to deploy on HuggingFace Spaces |
| `planner_1758012885.pdf` | Example of a PDF generated by the agent |
| `Presentation.pptx` | Powerpoint Presentation of this project |

---
