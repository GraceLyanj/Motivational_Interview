import { useState } from "react";

type Role = "client" | "counselor";
type Kind = "thinking" | "message";

interface Message {
  id: number;
  kind: Kind;
  role: Role;
  text: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const renderWithBold = (text: string) => {
    const parts = text.split("**");
    return parts.map((part, idx) =>
      idx % 2 === 1 ? <strong key={idx}>{part}</strong> : <span key={idx}>{part}</span>,
    );
  };

  const renderValueWithBullets = (label: string, value: string) => {
    const trimmed = value.trim();

    // Case 1: Python-like list: ['A', 'B']
    if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
      const inner = trimmed.slice(1, -1);
      const items = inner
        .split(",")
        .map((s) => s.replace(/^[\s'"]+|[\s'"]+$/g, "").trim())
        .filter(Boolean);
      return (
        <ul className="list-disc pl-4 space-y-0.5">
          {items.map((it, i) => (
            <li key={i}>{renderWithBold(it)}</li>
          ))}
        </ul>
      );
    }

    // Case 2: numbered list like "1. xxx 2. yyy"
    const numberedParts = trimmed.split(/\s(?=\d+\.\s)/g);
    if (numberedParts.length > 1) {
      return (
        <ul className="list-disc pl-4 space-y-0.5">
          {numberedParts.map((segment, i) => {
            const cleaned = segment.replace(/^\d+\.\s*/, "");
            return (
              <li key={i}>
                {renderWithBold(cleaned)}
              </li>
            );
          })}
        </ul>
      );
    }

    // Case 3: dash bullets like "- Affirm: ... - Open Question: ..."
    if (trimmed.startsWith("- ")) {
      // Remove leading "- " then split on " - " between items
      const dashStr = trimmed.slice(2);
      const dashParts = dashStr.split(/\s-\s(?=[A-Za-z0-9])/g);
      if (dashParts.length > 0) {
        return (
          <ul className="list-disc pl-4 space-y-0.5">
            {dashParts.map((segment, i) => (
              <li key={i}>{renderWithBold(segment.trim())}</li>
            ))}
          </ul>
        );
      }

    // Case 4: special handling for "Strategy Selection" to create nested bullets
    if (
      label === "Strategy Selection" &&
      /Selected strategies for (the )?next response \(max 2\)/i.test(trimmed)
    ) {
      const splitPoint = trimmed.match(
        /Selected strategies for (the )?next response \(max 2\):\s*/i,
      );
      const analysisPart = splitPoint
        ? trimmed.slice(0, splitPoint.index).trim()
        : trimmed;
      const restPart = splitPoint
        ? trimmed.slice((splitPoint.index ?? 0) + splitPoint[0].length)
        : "";
      // Keep only the analysis sentence(s); drop any trailing "Selected strategies..." or bracket list
      const analysis = analysisPart
        .replace(/\s*[-–]?\s*\*\*Selected strategies.*$/i, "")
        .replace(/^\*\*Analysis \(brief\):\*\*\s*/i, "")
        .trim();
      const rest = restPart.trim();

      // Sub-bullets: either bracket list [A – x., B – y.] or numbered "1. ... 2. ..."
      let subItems: string[] = [];
      if (rest.startsWith("[") && rest.includes("]")) {
        const inner = rest.slice(1, rest.indexOf("]")).trim();
        // Split by ", " but only when next segment looks like "StrategyName – " (reason can contain commas)
        const rawParts = inner.split(/,\s+/);
        const merged: string[] = [];
        for (const p of rawParts) {
          const t = p.trim();
          if (!t) continue;
          if (t.includes(" – ")) {
            merged.push(t);
          } else if (merged.length > 0) {
            merged[merged.length - 1] += ", " + t;
          } else {
            merged.push(t);
          }
        }
        subItems = merged;
      } else {
        const numbered = rest.split(/\s(?=\d+\.\s)/g);
        if (numbered.length > 1) {
          subItems = numbered.map((seg) =>
            seg.replace(/^\d+\.\s*/, "").trim(),
          );
        }
      }

      return (
        <ul className="list-disc pl-4 space-y-0.5">
          {/* Top-level bullet 1: Analysis (brief) */}
          {analysis && <li>{renderWithBold(analysis)}</li>}

          {/* Top-level bullet 2: Selected strategies with nested sub-bullets */}
          <li>
            {renderWithBold("Selected strategies for the next response (max 2):")}
            {subItems.length > 0 && (
              <ul className="list-disc pl-5 mt-1 space-y-0.5">
                {subItems.map((seg, i) => (
                  <li key={i}>{renderWithBold(seg)}</li>
                ))}
              </ul>
            )}
          </li>
        </ul>
      );
    }
    }

    // Default: plain paragraph
    return <div>{renderWithBold(trimmed)}</div>;
  };

  const splitThinking = (text: string): { label: string; value: string }[] => {
    // Strip outer brackets if present
    const trimmed = text.trim().replace(/^\[/, "").replace(/\]$/, "");
    // Split on top-level "||" — nested brackets like ['Open Question', 'Structure']
    // don't contain "||", so this is safe enough for our current format.
    const parts = trimmed
      .split("||")
      .map((p) => p.trim())
      .filter(Boolean);

    return parts.map((part) => {
      const [rawLabel, ...rest] = part.split(":");
      const label = (rawLabel ?? "").trim();
      // Also trim any leading/trailing quote from single-value fields
      const value = rest
        .join(":")
        .trim()
        .replace(/^['"]/, "")
        .replace(/['"]$/, "");
      return { label, value };
    });
  };

  const runSimulation = () => {
    if (isRunning) return;
    setIsRunning(true);
    setError(null);
    setMessages([]);

    const source = new EventSource("http://localhost:8000/auto_session_stream");

    source.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as {
          kind: Kind;
          role: Role;
          text: string;
        };
        setMessages((prev) => [
          ...prev,
          {
            id: prev.length + 1,
            kind: data.kind,
            role: data.role,
            text: data.text,
          },
        ]);
      } catch (e) {
        console.error("Failed to parse event data", e);
      }
    };

    source.onerror = () => {
      source.close();
      setIsRunning(false);
      setError("Stream ended or failed. Check backend logs and try again.");
    };
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-3xl bg-white shadow-lg rounded-xl flex flex-col h-[80vh]">
        <header className="px-4 py-3 border-b flex items-center justify-between">
          <h1 className="text-lg font-semibold">
            CAMI – AI Client &amp; Counselor
          </h1>
          <button
            className="px-3 py-1 rounded-full border border-blue-600 text-blue-600 text-sm hover:bg-blue-50 disabled:opacity-60"
            type="button"
            onClick={runSimulation}
            disabled={isRunning}
          >
            {isRunning ? "Running…" : "Run CAMI Session"}
          </button>
        </header>

        <main className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
          {messages.length === 0 && !isRunning && (
            <p className="text-sm text-slate-500">
              Click &quot;Run CAMI Session&quot; to simulate a full dialogue
              where both the client and counselor are AI. Messages will appear
              live as they are generated.
            </p>
          )}

          {messages.map((m) =>
            m.kind === "thinking" ? (
              <div key={m.id} className="text-xs text-slate-500">
                <div className="mb-1 italic">Thinking</div>
                <div className="border border-slate-200 bg-slate-50 rounded-md px-3 py-2 space-y-1">
                  {splitThinking(m.text).map((item, idx) => (
                    <div key={`${m.id}-${idx}`} className="whitespace-pre-line">
                      {item.label && (
                        <div className="font-semibold">{item.label}</div>
                      )}
                      {item.value && (
                        <div>{renderValueWithBullets(item.label, item.value)}</div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div
                key={m.id}
                className={`flex ${
                  m.role === "client" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[75%] rounded-2xl px-3 py-2 text-sm shadow-sm ${
                    m.role === "client"
                      ? "bg-blue-600 text-white rounded-br-none"
                      : "bg-slate-100 text-slate-900 rounded-bl-none"
                  }`}
                >
                  <div className="whitespace-pre-line">
                    <span className="font-semibold">
                      {m.role === "client" ? "Client: " : "Counselor: "}
                    </span>
                    {m.text}
                  </div>
                </div>
              </div>
            ),
          )}
        </main>

        <footer className="border-t px-4 py-3 text-[11px] text-slate-500">
          {error ? (
            <p className="text-red-500">{error}</p>
          ) : (
            <p>
              Both sides are AI; this view shows the client–counselor
              conversation as it is generated in real time.
            </p>
          )}
        </footer>
      </div>
    </div>
  );
}

export default App;

