import { useState, useEffect } from 'react'

const METHODS = [
  { id: 'm1', name: 'M1 — Naive RAG',        desc: 'Dense Retrieval + Cosine Similarity' },
  { id: 'm2', name: 'M2 — Hybrid Search',     desc: 'BM25 + Dense + RRF Fusion' },
  { id: 'm3', name: 'M3 — RAG-Fusion',        desc: 'Multi-query + Hybrid + RRF' },
  { id: 'm4', name: 'M4 — CRAG',              desc: 'Retrieval Evaluator + Self-correction' },
  { id: 'm5', name: 'M5 — Self-RAG Lite',     desc: 'Adaptive Retrieval + Reflection' },
]

const METRIC_LABELS = [
  { key: 'exact_match',      label: 'Exact Match' },
  { key: 'rouge_l',          label: 'ROUGE-L' },
  { key: 'context_hit_rate', label: 'Context Hit Rate' },
  { key: 'context_recall',   label: 'Context Recall (RAGAS)' },
  { key: 'faithfulness',     label: 'Faithfulness (RAGAS)' },
  { key: 'answer_relevancy', label: 'Answer Relevancy (RAGAS)' },
]

// ── API helpers ──────────────────────────────────────────────────────────────

async function askAPI(question, method) {
  const res = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, method }),
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Yêu cầu thất bại')
  return data
}

async function uploadAPI(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/upload', { method: 'POST', body: form })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Upload thất bại')
  return data
}

async function metricsAPI() {
  const res = await fetch('/metrics')
  if (!res.ok) throw new Error('Không tải được metrics')
  return res.json()
}

// ── Small components ──────────────────────────────────────────────────────────

function MethodSelect({ value, onChange, id }) {
  return (
    <select id={id} value={value} onChange={e => onChange(e.target.value)} className="method-select">
      {METHODS.map(m => (
        <option key={m.id} value={m.id}>{m.name}</option>
      ))}
    </select>
  )
}

function AnswerCard({ result, title }) {
  const [showSources, setShowSources] = useState(false)
  if (!result) return null
  const mInfo = METHODS.find(m => m.id === result.method)
  return (
    <div className="answer-card">
      {title && <div className="answer-title">{title}</div>}
      <div className="answer-badge">{result.method?.toUpperCase()} — {mInfo?.desc}</div>
      <div className="answer-text">{result.answer}</div>
      <div className="answer-meta">
        <span className="latency">⏱ {result.latency_ms} ms</span>
        <button
          onClick={() => setShowSources(s => !s)}
          className="sources-toggle"
        >
          {showSources ? 'Ẩn nguồn' : `Xem nguồn (${result.contexts?.length || 0} đoạn)`}
        </button>
      </div>
      {showSources && (
        <div className="sources-list">
          {result.contexts?.map((ctx, i) => (
            <div key={i} className="source-chunk">
              <div className="source-num">Đoạn {i + 1}</div>
              <div className="source-text">{ctx}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Tabs ─────────────────────────────────────────────────────────────────────

function QATab() {
  const [question, setQuestion] = useState('')
  const [method, setMethod] = useState('m1')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleAsk = async () => {
    if (!question.trim()) return
    setLoading(true)
    setError('')
    try {
      const data = await askAPI(question.trim(), method)
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const mInfo = METHODS.find(m => m.id === method)

  return (
    <div className="tab-content">
      <textarea
        className="question-input"
        placeholder="Nhập câu hỏi tiếng Việt... (Ctrl+Enter để gửi)"
        value={question}
        onChange={e => setQuestion(e.target.value)}
        onKeyDown={e => { if (e.ctrlKey && e.key === 'Enter') handleAsk() }}
        rows={3}
      />
      <div className="controls">
        <div className="method-wrap">
          <MethodSelect value={method} onChange={setMethod} id="qa-method" />
          {mInfo && <span className="method-hint">{mInfo.desc}</span>}
        </div>
        <button
          onClick={handleAsk}
          disabled={loading || !question.trim()}
          className="ask-btn"
        >
          {loading ? '⏳ Đang hỏi...' : 'Hỏi'}
        </button>
      </div>
      {error && <div className="error-msg">⚠ {error}</div>}
      <AnswerCard result={result} />
    </div>
  )
}

function CompareTab() {
  const [question, setQuestion] = useState('')
  const [method1, setMethod1] = useState('m1')
  const [method2, setMethod2] = useState('m2')
  const [results, setResults] = useState([null, null])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleCompare = async () => {
    if (!question.trim()) return
    setLoading(true)
    setError('')
    try {
      const [r1, r2] = await Promise.all([
        askAPI(question.trim(), method1),
        askAPI(question.trim(), method2),
      ])
      setResults([r1, r2])
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="tab-content">
      <textarea
        className="question-input"
        placeholder="Nhập câu hỏi để so sánh 2 phương pháp..."
        value={question}
        onChange={e => setQuestion(e.target.value)}
        rows={3}
      />
      <div className="controls compare-controls">
        <MethodSelect value={method1} onChange={setMethod1} id="cmp-m1" />
        <span className="vs-badge">VS</span>
        <MethodSelect value={method2} onChange={setMethod2} id="cmp-m2" />
        <button
          onClick={handleCompare}
          disabled={loading || !question.trim()}
          className="ask-btn"
        >
          {loading ? '⏳ Đang so sánh...' : 'So sánh'}
        </button>
      </div>
      {error && <div className="error-msg">⚠ {error}</div>}
      <div className="compare-grid">
        <AnswerCard result={results[0]} title={METHODS.find(m => m.id === method1)?.name} />
        <AnswerCard result={results[1]} title={METHODS.find(m => m.id === method2)?.name} />
      </div>
    </div>
  )
}

function MetricsTab() {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    metricsAPI()
      .then(setMetrics)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="tab-content loading">Đang tải metrics...</div>
  if (error) return <div className="tab-content error-msg">⚠ {error}</div>
  if (!metrics || Object.keys(metrics).length === 0)
    return <div className="tab-content">Chưa có dữ liệu metrics.</div>

  const present = ['m1', 'm2', 'm3', 'm4', 'm5'].filter(m => metrics[m])

  return (
    <div className="tab-content">
      <div className="metrics-header">
        <h3>Kết quả đánh giá M1–M5</h3>
        <p className="metrics-note">Local metrics: n=400 | RAGAS: n=50 mẫu, judge = llama-3.3-70b</p>
      </div>
      <div className="metrics-table-wrap">
        <table className="metrics-table">
          <thead>
            <tr>
              <th className="metric-col">Metric</th>
              {present.map(m => (
                <th key={m}>{m.toUpperCase()}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {METRIC_LABELS.map(({ key, label }) => {
              const values = present.map(m => metrics[m]?.[key])
              const maxVal = Math.max(...values.filter(v => v != null && !isNaN(v)))
              return (
                <tr key={key}>
                  <td className="metric-col metric-label">{label}</td>
                  {present.map((m, i) => {
                    const v = values[i]
                    const isBest = v != null && Math.abs(v - maxVal) < 1e-6
                    return (
                      <td key={m} className={isBest ? 'metric-best' : ''}>
                        {v != null ? v.toFixed(4) : '—'}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <div className="metrics-legend">
        <span className="legend-best">Xanh = cao nhất</span>
        &nbsp;|&nbsp; RAGAS chạy trên 50 mẫu ngẫu nhiên từ ViQuAD 2.0
      </div>
    </div>
  )
}

function UploadSection() {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [status, setStatus] = useState(null) // {ok, msg}
  const [open, setOpen] = useState(false)

  const handleUpload = async () => {
    if (!file) return
    setUploading(true)
    setStatus(null)
    try {
      const data = await uploadAPI(file)
      const action = data.status === 'updated'
        ? `✓ Đã cập nhật "${data.filename}" — xóa ${data.chunks_removed} chunk cũ, thêm ${data.chunks_added} chunk mới`
        : `✓ Đã index "${data.filename}" — thêm ${data.chunks_added} chunks`
      setStatus({ ok: true, msg: action })
      setFile(null)
    } catch (e) {
      setStatus({ ok: false, msg: `✗ Lỗi: ${e.message}` })
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="upload-section">
      <button className="upload-toggle" onClick={() => setOpen(o => !o)}>
        📄 Tải lên PDF tiếng Việt {open ? '▲' : '▼'}
      </button>
      {open && (
        <div className="upload-body">
          <div className="upload-row">
            <input
              type="file"
              accept=".pdf"
              id="pdf-input"
              className="file-input"
              onChange={e => { setFile(e.target.files[0]); setStatus(null) }}
            />
            <label htmlFor="pdf-input" className="file-label">
              {file ? `📎 ${file.name}` : 'Chọn file PDF...'}
            </label>
            <button
              onClick={handleUpload}
              disabled={!file || uploading}
              className="upload-btn"
            >
              {uploading ? '⏳ Đang xử lý...' : 'Tải lên & Index'}
            </button>
          </div>
          {status && (
            <div className={`upload-status ${status.ok ? 'success' : 'error'}`}>
              {status.msg}
            </div>
          )}
          <p className="upload-hint">
            PDF sau khi index sẽ được thêm vào ChromaDB và có thể hỏi ngay.
          </p>
        </div>
      )}
    </div>
  )
}

// ── Root app ─────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState('qa')

  const TABS = [
    { id: 'qa',      label: '💬 Q&A' },
    { id: 'compare', label: '⚖ So sánh' },
    { id: 'metrics', label: '📊 Metrics' },
  ]

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <h1>RAG Demo — Vietnamese Q&amp;A</h1>
          <p>So sánh 5 phương pháp RAG trên văn bản tiếng Việt · UIT-ViQuAD 2.0</p>
        </div>
        <div className="method-pills">
          {METHODS.map(m => (
            <span key={m.id} className="method-pill">{m.id.toUpperCase()}</span>
          ))}
        </div>
      </header>

      <UploadSection />

      <nav className="tab-nav">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`tab-btn ${tab === t.id ? 'active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {tab === 'qa'      && <QATab />}
      {tab === 'compare' && <CompareTab />}
      {tab === 'metrics' && <MetricsTab />}
    </div>
  )
}
