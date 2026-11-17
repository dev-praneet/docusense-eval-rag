import { useEffect, useState } from "react";

const API_URL = "http://localhost:8000/api/v1";

interface Account {
  id: string;
  email: string;
}

interface DocumentItem {
  id: string;
  filename: string;
}

interface Source {
  score: number;
  file_name: string;
  page_label: string;
  snippet: string;
}

interface EvalScore {
  faithfulness: number;
  relevancy: number;
}

export function Main() {
  const [account, setAccount] = useState<Account | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [authMode, setAuthMode] = useState<"login" | "register" | null>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [question, setQuestion] = useState("");
  const [questions, setQuestions] = useState<string[]>([]);
  const [answers, setAnswers] = useState<string[]>([]);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [scores, setScores] = useState<EvalScore | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);

  const refreshDocuments = async () => {
    const response = await fetch(`${API_URL}/documents`, { credentials: "include" });
    if (!response.ok) throw new Error("Could not load documents.");
    setDocuments(await response.json());
  };

  useEffect(() => {
    const load = async () => {
      try {
        const response = await fetch(`${API_URL}/auth/me`, { credentials: "include" });
        if (!response.ok) throw new Error("Could not load account.");
        setAccount((await response.json()).user);
        await refreshDocuments();
      } catch (error) {
        setMessage(error instanceof Error ? error.message : "Connection failed.");
      }
    };
    void load();
  }, []);

  const handleAuth = async (event: React.SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!authMode) return;
    setBusy(true);
    setMessage("");
    try {
      const response = await fetch(`${API_URL}/auth/${authMode}`, {
        method: "POST", credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Account request failed.");
      setAccount(result.user);
      setAuthMode(null);
      setPassword("");
      setAnswer("");
      setSources([]);
      await refreshDocuments();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Account request failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleLogout = async () => {
    setBusy(true);
    try {
      const response = await fetch(`${API_URL}/auth/logout`, { method: "POST", credentials: "include" });
      if (!response.ok) throw new Error("Could not log out.");
      setAccount(null);
      setAnswer("");
      setSources([]);
      setScores(null);
      await refreshDocuments();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not log out.");
    } finally {
      setBusy(false);
    }
  };

  const handleUpload = async (event: React.SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    setMessage("");
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(`${API_URL}/documents`, {
        method: "POST", credentials: "include", body: form,
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Upload failed.");
      setFile(null);
      const input = document.querySelector<HTMLInputElement>("#pdf-upload");
      if (input) input.value = "";
      await refreshDocuments();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (documentId: string) => {
    setBusy(true);
    setMessage("");
    try {
      const response = await fetch(`${API_URL}/documents/${documentId}`, {
        method: "DELETE", credentials: "include",
      });
      if (!response.ok) throw new Error("Could not delete document.");
      setAnswer("");
      setSources([]);
      await refreshDocuments();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not delete document.");
    } finally {
      setBusy(false);
    }
  };

  const handleQuery = async () => {
    if (!question.trim() || isStreaming) return;
    setQuestions((prev) => [...prev, question]);


    // Reset state for new query
    setAnswer("");
    setSources([]);
    setScores(null);
    setIsStreaming(true);

    try {
      // 1. Send POST request to FastAPI SSE Endpoint
      const response = await fetch("http://localhost:8000/api/v1/query/stream", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, similarity_top_k: 3 }),
      });

      if (!response.body) {
        throw new Error("No response body received from server.");
      }

      // 2. Attach Stream Reader & Text Decoder
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let accumulatedAnswer = "";
      let retrievedContexts: string[] = [];

      // 3. Loop through incoming binary chunks
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        // SSE messages are delimited by double newlines
        const events = chunk.split("\n\n");

        for (const event of events) {
          if (!event.trim()) continue;

          // Process Metadata Event (Sources & Contexts)
          if (event.includes("event: metadata")) {
            const dataMatch = event.match(/data:\s*({.*})/);
            if (dataMatch) {
              const parsed = JSON.parse(dataMatch[1]);
              setSources(parsed.sources || []);
              retrievedContexts = parsed.contexts || [];
            }
          }

          // Process Delta Event (Live Text Tokens)
          if (event.includes("event: delta")) {
            const dataMatch = event.match(/data:\s*({.*})/);
            if (dataMatch) {
              const parsed = JSON.parse(dataMatch[1]);
              const token = parsed.text || "";
              accumulatedAnswer += token;
              setAnswer((prev) => prev + token);
            }
          }
        }
      }

      setIsStreaming(false);
      setQuestion("");
      setAnswers((prev) => [...prev, accumulatedAnswer]);
      setAnswer("");

      // 4. Asynchronous DeepEval Quality Score Check after stream completion (commented out for now)
      /**
      if (accumulatedAnswer && retrievedContexts.length > 0) {
        setIsEvaluating(true);
        const evalRes = await fetch("http://localhost:8000/api/v1/eval", {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question,
            answer: accumulatedAnswer,
            retrieved_contexts: retrievedContexts,
          }),
        });

        if (evalRes.ok) {
          const evalData = await evalRes.json();
          setScores({
            faithfulness: evalData.faithfulness,
            relevancy: evalData.relevancy,
          });
        }
        setIsEvaluating(false);
    }
    */
    } catch (error) {
      console.error("Streaming failed:", error);
      setIsStreaming(false);
      setIsEvaluating(false);
    }
  };

  const  qna: string[] = [];
  questions.forEach((q, idx) => {
    qna.push(q);
    answers[idx] && qna.push(answers[idx]);
  });

  return (
    <div className="workspace">
      <header className="workspace-header">
        <h2>DocuSense AI</h2>
        <div className="account-actions">
          <span>{account ? account.email : "Guest session"}</span>
          {account ? <button type="button" disabled={busy || isStreaming} onClick={handleLogout}>Log out</button> : (
            <>
              <button type="button" disabled={isStreaming} onClick={() => setAuthMode("login")}>Log in</button>
              <button type="button" disabled={isStreaming} onClick={() => setAuthMode("register")}>Create account</button>
            </>
          )}
        </div>
      </header>
      {authMode && (
        <form className="account-form" onSubmit={handleAuth}>
          <div className="form-title">
            <strong>{authMode === "login" ? "Log in" : "Create account"}</strong>
            <button type="button" onClick={() => setAuthMode(null)} aria-label="Close account form">×</button>
          </div>
          <input type="email" placeholder="Email" aria-label="Email" required value={email} onChange={(event) => setEmail(event.target.value)} />
          <input type="password" placeholder="Password" aria-label="Password" minLength={12} required value={password} onChange={(event) => setPassword(event.target.value)} />
          <button type="submit" disabled={busy || isStreaming}>{authMode === "login" ? "Log in" : "Create account"}</button>
        </form>
      )}
      <section className="document-section">
        <h3>Documents</h3>
        <form className="upload-form" onSubmit={handleUpload}>
          <input id="pdf-upload" type="file" accept="application/pdf,.pdf" aria-label="Choose PDF" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
          <button type="submit" disabled={busy || !file}>{busy ? "Working…" : "Upload PDF"}</button>
        </form>
        <ul className="document-list">
          {documents.map((item) => (
            <li key={item.id}>
              <span title={item.filename}>{item.filename}</span>
              <button type="button" disabled={busy} onClick={() => handleDelete(item.id)} aria-label={`Delete ${item.filename}`}>Delete</button>
            </li>
          ))}
        </ul>
      </section>
      {message && <p role="alert" className="workspace-error">{message}</p>}
      <div className="qna-container">
          {qna.map((x, idx) => (
            <div style={{ padding: "16px", border: "1px solid #e0e0e0", borderRadius: "6px", marginBottom: "20px" }}>
              <h3 style={{ marginTop: 0 }}>{idx % 2 ? "Answer" : "Question"}:</h3>
              <p style={{ whiteSpace: "pre-wrap", lineHeight: "1.6", margin: 0 }}>{x}</p>
            </div>
          ))}
      </div>
      {/* Answer Output Container */}
      {answer && (
        <div style={{ padding: "16px", border: "1px solid #e0e0e0", borderRadius: "6px", marginBottom: "20px" }}>
          <h3 style={{ marginTop: 0 }}>Answer:</h3>
          <p style={{ whiteSpace: "pre-wrap", lineHeight: "1.6", margin: 0 }}>{answer}</p>
        </div>
      )}

      {/* Query Input Section */}
      <div style={{ display: "flex", gap: "10px", marginBlock: "20px" }}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about your documents..."
          style={{ flex: 1, padding: "12px", fontSize: "16px", borderRadius: "4px", border: "1px solid #ccc" }}
          onKeyDown={(e) => e.key === "Enter" && handleQuery()}
        />
        <button
          onClick={handleQuery}
          disabled={isStreaming}
          style={{
            padding: "12px 24px",
            fontSize: "16px",
            cursor: isStreaming ? "not-allowed" : "pointer",
            backgroundColor: "#0070f3",
            color: "#fff",
            border: "none",
            borderRadius: "4px",
          }}
        >
          {isStreaming ? "Streaming..." : "Ask"}
        </button>
      </div>

      {/* DeepEval Scorecard Dashboard */}
      {/* {isEvaluating && (
        <div style={{ padding: "12px", backgroundColor: "#eef6ff", borderRadius: "4px", marginBottom: "20px" }}>
          ⏳ Computing DeepEval metrics (Faithfulness & Relevancy)...
        </div>
      )} */}
      {/* {scores && (
        <div style={{ padding: "16px", backgroundColor: "#f9f9f9", border: "1px solid #ddd", borderRadius: "6px", marginBottom: "20px" }}>
          <h4 style={{ marginTop: 0, marginBottom: "12px" }}>Quality Control Scorecard</h4>
          <p style={{ margin: "4px 0" }}>
            <strong>Faithfulness:</strong> {scores.faithfulness} / 1.0{" "}
            {scores.faithfulness >= 0.7 ? "✅ Pass" : "❌ Flagged"}
          </p>
          <p style={{ margin: "4px 0" }}>
            <strong>Answer Relevancy:</strong> {scores.relevancy} / 1.0{" "}
            {scores.relevancy >= 0.7 ? "✅ Pass" : "❌ Flagged"}
          </p>
        </div>
      )} */}

      {/* Source Citations Accordions */}
      {sources.length > 0 && (
        <div>
          <h4 style={{ marginBottom: "12px" }}>Retrieved Source Citations</h4>
          {sources.map((src, idx) => (
            <details key={idx} style={{ marginBottom: "10px", padding: "10px", border: "1px solid #eee", borderRadius: "4px" }}>
              <summary style={{ cursor: "pointer", fontWeight: "bold" }}>
                [{idx + 1}] {src.file_name} (Page {src.page_label}) — Similarity Score: {src.score}
              </summary>
              <p style={{ fontSize: "14px", color: "#555", marginTop: "10px", marginBottom: 0 }}>
                "{src.snippet}"
              </p>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}

export default Main;
