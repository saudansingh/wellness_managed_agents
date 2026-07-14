import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import './index.css';

// Configurable API base URL with safe optional chaining
const API_BASE_URL = import.meta.env?.VITE_API_URL || 'http://127.0.0.1:8000';

export default function App() {
  const [screen, setScreen] = useState('onboarding');
  const [profile, setProfile] = useState({ userId: '', age: '', weight: '', injuries: 'None', goals: '' });
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('Thinking...');
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleOnboardingSubmit = async (e) => {
    e.preventDefault();
    if (!profile.userId || !profile.age || !profile.weight) {
      alert("Please fill out the required fields!");
      return;
    }
    setIsLoading(true);
    setLoadingStatus("Synchronizing profile vectors...");
    try {
      const response = await fetch(`${API_BASE_URL}/api/save-profile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(profile),
      });

      // Explicitly check response status (new code logic)
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.detail || `Server responded with ${response.status}`);
      }

      setMessages([{ sender: 'ai', text: `Welcome **${profile.userId}**! Your profile is loaded. Ask me anything about your fitness or diet targets!` }]);
      setScreen('chat');
    } catch (error) {
      console.error("Profile save failed:", error);
      setMessages([{ sender: 'ai', text: `⚠️ **Couldn't save your profile.** (${error.message}). You can still explore the interface, but responses may not work until this is fixed.` }]);
      setScreen('chat');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input;
    setInput('');
    setMessages(prev => [...prev, { sender: 'user', text: userMessage }]);
    setIsLoading(true);
    setLoadingStatus('Thinking...'); // Clean, generic loading state

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: profile.userId,
          user_message: userMessage
        }),
      });

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        const detail = errBody.detail || `Server responded with ${response.status}`;
        setMessages(prev => [...prev, { sender: 'ai', text: `❌ ${detail}` }]);
        return;
      }

      const data = await response.json();
      setMessages(prev => [...prev, { sender: 'ai', text: data.plan_markdown }]);
    } catch (error) {
      setMessages(prev => [...prev, { sender: 'ai', text: "❌ Connection error. Is the backend reachable?" }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearHistory = () => {
    if (window.confirm("Purge conversational matrix log data?")) {
      setMessages([]);
    }
  };

  if (screen === 'onboarding') {
    return (
      <div className="matrix-container">
        <div className="wellness-card">
          <div style={{ textAlign: 'center' }}>
            <div className="header-badge">🧬</div>
            <h2 className="matrix-title">Setup Your Engine</h2>
            <p className="matrix-subtitle">Gathering your metrics to synchronize the multi-agent wellness matrix.</p>
          </div>

          <form onSubmit={handleOnboardingSubmit}>
            <div className="form-group">
              <label className="matrix-label">User ID / Name</label>
              <input type="text" required className="matrix-input" placeholder="e.g. ankur" value={profile.userId} onChange={e => setProfile({...profile, userId: e.target.value})} />
            </div>

            <div className="form-row form-group">
              <div>
                <label className="matrix-label">Age</label>
                <input type="number" required className="matrix-input" placeholder="24" value={profile.age} onChange={e => setProfile({...profile, age: e.target.value})} />
              </div>
              <div>
                <label className="matrix-label">Weight (KG)</label>
                <input type="number" required className="matrix-input" placeholder="61" value={profile.weight} onChange={e => setProfile({...profile, weight: e.target.value})} />
              </div>
            </div>

            <div className="form-group">
              <label className="matrix-label">Injuries / Medical Conditions</label>
              <input type="text" className="matrix-input" placeholder="e.g. none, lower back stiffness" value={profile.injuries} onChange={e => setProfile({...profile, injuries: e.target.value})} />
            </div>

            <div className="form-group">
              <label className="matrix-label">Primary Wellness Goals</label>
              <textarea className="matrix-textarea" placeholder="What target benchmarks are we optimizing?" value={profile.goals} onChange={e => setProfile({...profile, goals: e.target.value})} />
            </div>

            <button type="submit" className="matrix-btn">
              <span>Initialize Agent Mesh</span> <span>⚡</span>
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-layout">
      {/* Sidebar Core Context Display */}
      <aside className="chat-sidebar">
        <div className="sidebar-header">
          <div className="status-dot" />
          <h1 className="sidebar-title">A.I. Core Matrix</h1>
        </div>

        <div className="metric-box">
          <span className="metric-box-label">Active Subject</span>
          <span className="metric-box-value active-user">{profile.userId}</span>
        </div>

        <div className="form-row">
          <div className="metric-box">
            <span className="metric-box-label">Age Vector</span>
            <span className="metric-box-value">{profile.age} yrs</span>
          </div>
          <div className="metric-box">
            <span className="metric-box-label">Mass Index</span>
            <span className="metric-box-value">{profile.weight} KG</span>
          </div>
        </div>

        <div className="metric-box">
          <span className="metric-box-label">Risk Factors</span>
          <span className="metric-box-value" style={{ color: '#fbbf24', fontSize: '12px' }}>{profile.injuries}</span>
        </div>

        <div className="metric-box" style={{ flex: 1, minHeight: '80px' }}>
          <span className="metric-box-label">Directives</span>
          <span className="metric-box-value" style={{ fontSize: '12px', color: '#cbd5e1' }}>{profile.goals || "None."}</span>
        </div>

        <button onClick={handleClearHistory} className="purge-btn">
          Purge Chat Memory
        </button>
      </aside>

      {/* Main Terminal Chat Canvas */}
      <div className="chat-main">
        <header className="chat-header">
          <button onClick={() => setScreen('onboarding')} className="back-btn">
            ← Modify Profile
          </button>
          <span style={{ fontSize: '11px', color: '#64748b', fontFamily: 'monospace' }}>
            NODE: <span style={{ color: '#10b981', fontWeight: 'bold' }}>ONLINE</span>
          </span>
        </header>

        <main className="chat-messages-container">
          {messages.map((msg, index) => (
            <div key={index} className={`message-row ${msg.sender === 'user' ? 'user' : 'ai'}`}>
              <div className="bubble">
                {msg.sender === 'user' ? (
                  <span style={{ fontWeight: '600' }}>{msg.text}</span>
                ) : (
                  <div className="prose">
                    <ReactMarkdown>{msg.text}</ReactMarkdown>
                  </div>
                )}
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="message-row ai">
              <div className="loading-bubble">
                <span className="status-dot animate-pulse" />
                <span>{loadingStatus}</span>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </main>

        <footer className="chat-footer">
          <form onSubmit={handleSendMessage} className="input-form">
            <input
              type="text"
              value={input}
              disabled={isLoading}
              onChange={e => setInput(e.target.value)}
              placeholder="Request micro-adjustments, workout protocols, or dietary splits..."
              className="chat-input"
            />
            <button type="submit" disabled={isLoading || !input.trim()} className="transmit-btn">
              Transmit
            </button>
          </form>
        </footer>
      </div>
    </div>
  );
}
