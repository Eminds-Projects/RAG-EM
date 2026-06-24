import { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [status, setStatus] = useState("");

  const API_URL = "http://127.0.0.1:8000";

  async function uploadFile() {
    if (!file) {
      setStatus("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setStatus("Uploading file...");

    const response = await fetch(`${API_URL}/upload`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    setStatus(data.message);
  }

  async function indexDocuments() {
    setStatus("Indexing documents...");

    const response = await fetch(`${API_URL}/index`, {
      method: "POST",
    });

    const data = await response.json();
    setStatus(`${data.message} Chunks indexed: ${data.chunks_indexed}`);
  }

  async function askQuestion() {
    if (!question.trim()) {
      setStatus("Please enter a question.");
      return;
    }

    setStatus("Generating answer...");
    setAnswer("");
    setSources([]);

    const response = await fetch(`${API_URL}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    });

    const data = await response.json();

    setAnswer(data.answer);
    setSources(data.sources || []);
    setStatus("Answer generated.");
  }

  return (
    <div className="container">
      <h1>Local RAG System with Ollama</h1>

      <div className="card">
        <h2>1. Upload Document</h2>
        <input
          type="file"
          accept=".txt,.pdf"
          onChange={(event) => setFile(event.target.files[0])}
        />
        <button onClick={uploadFile}>Upload</button>
      </div>

      <div className="card">
        <h2>2. Index Documents</h2>
        <button onClick={indexDocuments}>Index Documents</button>
      </div>

      <div className="card">
        <h2>3. Ask Question</h2>
        <textarea
          placeholder="Ask a question about your documents..."
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
        />
        <button onClick={askQuestion}>Ask</button>
      </div>

      {status && (
        <div className="card">
          <strong>Status:</strong> {status}
        </div>
      )}

      {answer && (
        <div className="card">
          <h2>Answer</h2>
          <p>{answer}</p>

          <h3>Sources</h3>
          {sources.length > 0 ? (
            <ul>
              {sources.map((source, index) => (
                <li key={index}>{source}</li>
              ))}
            </ul>
          ) : (
            <p>No sources returned.</p>
          )}
        </div>
      )}
    </div>
  );
}

export default App;