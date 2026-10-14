import { useState } from "react";

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
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [scores, setScores] = useState<EvalScore | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);

  const handleQuery = async () => {
    if (!question.trim() || isStreaming) return;

    // Reset state for new query
    setAnswer("");
    setSources([]);
    setScores(null);
    setIsStreaming(true);

    try {
      // 1. Send POST request to FastAPI SSE Endpoint
      const response = await fetch("http://localhost:8000/api/v1/query/stream", {
        method: "POST",
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

      // 4. Asynchronous DeepEval Quality Score Check after stream completion
      if (accumulatedAnswer && retrievedContexts.length > 0) {
        setIsEvaluating(true);
        const evalRes = await fetch("http://localhost:8000/api/v1/eval", {
          method: "POST",
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
    } catch (error) {
      console.error("Streaming failed:", error);
      setIsStreaming(false);
      setIsEvaluating(false);
    }
  };

  return (
    <div style={{ maxWidth: "800px", margin: "40px auto", fontFamily: "sans-serif", padding: "0 20px" }}>
      <h2 style={{ marginBottom: "20px" }}>DocuSense AI - Production RAG Interface</h2>

      {/* Query Input Section */}
      <div style={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
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

      {/* Answer Output Container */}
      {answer && (
        <div style={{ padding: "16px", border: "1px solid #e0e0e0", borderRadius: "6px", marginBottom: "20px" }}>
          <h3 style={{ marginTop: 0 }}>Answer:</h3>
          <p style={{ whiteSpace: "pre-wrap", lineHeight: "1.6", margin: 0 }}>{answer}</p>
        </div>
      )}

      {/* DeepEval Scorecard Dashboard */}
      {isEvaluating && (
        <div style={{ padding: "12px", backgroundColor: "#eef6ff", borderRadius: "4px", marginBottom: "20px" }}>
          ⏳ Computing DeepEval metrics (Faithfulness & Relevancy)...
        </div>
      )}
      {scores && (
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
      )}

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
