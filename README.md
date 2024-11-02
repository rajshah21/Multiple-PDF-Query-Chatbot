# Multiple PDF Query Chatbot 📄🤖

A powerful PDF-based chatbot application built using **Python**, **Streamlit**, **LangChain**, and the **OpenAI API**. This project leverages **Retrieval-Augmented Generation (RAG)** to answer complex questions based on the content of uploaded PDF documents. Ideal for anyone looking to interact with PDF documents conversationally, providing quick, accurate answers.

To Test out the RAG Implementation: https://multiple-pdf-rag-chatbot.streamlit.app/

---

## 🌟 Key Features

- **Upload Multiple PDFs**: Supports uploading and processing multiple PDFs simultaneously.
- **Efficient Retrieval**: Utilizes FAISS (Facebook AI Similarity Search) to store text embeddings for rapid information retrieval.
- **Context-Aware Responses**: With OpenAI's GPT API, the chatbot generates contextually accurate answers directly from PDF content.
- **Conversational Memory**: Remembers previous questions within a session, enabling seamless, follow-up questions without rephrasing context.
- **Scoped Responses**: For questions unrelated to the PDFs, the chatbot returns no answer, keeping responses relevant to the uploaded documents.

---

## 📂 Libraries and Tools Used

- **Streamlit**: For building the web application interface.
- **LangChain**: To process PDF text, converting it to embeddings for storage and retrieval.
- **FAISS**: Vector database for efficient similarity search and retrieval of relevant text embeddings.
- **OpenAI API**: For generating responses based on retrieved text data.

---

## 🚀 Demo

### Step 1: Upload PDFs
- Use the **'Browse files'** option to select your PDF files and click **'Process'** to initiate document processing.

### Step 2: Convert to Embeddings
- The application converts PDF text into embeddings using LangChain and stores them in FAISS, a high-speed vector database.

### Step 3: Ask Your Questions
- Once processing is complete, you can ask questions related to the PDF content.
- The chatbot provides accurate answers based on the retrieved data.

### Step 4: Conversational Memory
- The chatbot retains the memory of previous questions, allowing for a conversational flow.
- You can review answers and return to previous questions, similar to a typical chatbot experience.

### Step 5: Contextual Responses
- For questions outside the scope of the PDFs, the chatbot will return no answer, ensuring focus on document-specific information.

---

## 🛠 How to Run Locally

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rajshah21/multiple-pdf-query-chatbot.git
   cd multiple-pdf-query-chatbot
2. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
3. **Run the App**:
   ```bash
   streamlit run app2.py
