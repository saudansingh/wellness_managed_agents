import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import './index.css';

export default function App() {
  // Production URL pointing explicitly to your Cloud Run Instance
  const BACKEND_URL = import.meta.env.VITE_API_URL || 'https://wellness-managed-agents-git-436702918308.asia-south1.run.app';
  
  const [screen, setScreen] = useState(() => localStorage.getItem('wellness_screen') || 'onboarding');
  const [profile, setProfile] = useState(() => {
    const saved = localStorage.getItem('wellness_profile');
    return saved ? JSON.parse(saved) : { userId: '', age: '', weight: '', injuries: 'None', goals: '' };
  });
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem('wellness_chat_history');
    return saved ? JSON.parse(saved) : [];
  });
  
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('Orchestrating agents...');
  const chatEndRef = useRef(null);

  useEffect(() => {
    localStorage.setItem('wellness_screen', screen);
    localStorage.setItem('wellness_profile', JSON.stringify(profile));
    localStorage.setItem('wellness_chat_history', JSON.stringify(messages));
  }, [screen, profile, messages]);

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
    
    // Explicitly map frontend state keys to backend Pydantic schema parameters
    const payload = {
      userId: String(profile.userId).trim(),
      age: parseInt(profile.age, 10),
      weight: parseFloat(profile.weight),
      injuries: profile.injuries || "None",
      goals: profile.goals || "None"
    };

    try {
      const response = await fetch(`${BACKEND_URL}/api/save-profile`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Server returned status code: ${response.status}`);
      }

      if (messages.length === 0) {
        setMessages([{ sender: 'ai', text: `Welcome **${profile.userId}**! Your profile is securely loaded into the cloud repository. Ask me anything about custom sports nutrition, posture mechanics, or fitness programs!` }]);
      }
      setScreen('chat');
    } catch (error) {
      console.error("Onboarding Sync Error:", error);
      if (messages.length === 0) {
        setMessages([{ sender: 'ai', text: `⚠️ **Running in preview mode.** Local connection failed to handshake with Cloud Run backend. Using offline interface shell configuration.` }]);
      }
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

    const textLower = userMessage.toLowerCase().trim();
    const casualPhrases = ['hey', 'hi', 'hello', 'got it', 'ok', 'robot', 'who are you', 'thank you', 'thanks'];
    const isCasual = casualPhrases.some(phrase => textLower.includes(phrase));

    if (isCasual) {
      setLoadingStatus('💬 Connecting to Core Matrix...');
    } else if (textLower.includes('diet') || textLower.includes('eat') || textLower.includes('calorie') || textLower.includes('recipe')) {
      setLoadingStatus('🥗 Clinical Sports Dietitian compiling macros...');
    } else if (textLower.includes('yoga') || textLower.includes('stretch') || textLower.includes('back') || textLower.includes('pain')) {
      setLoadingStatus('🧘 Yoga Therapist optimizing posture alignment...');
    } else {
      setLoadingStatus('🏋️ Personal Trainer configuring sets and reps...');
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ 
          user_id: String(profile.userId).trim(), 
          user_message: userMessage 
        }),
      });

      if (!response.ok) {
        throw new Error(`Chat terminal returned status code: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success && data.plan_markdown) {
        setMessages(prev => [...prev, { sender: 'ai', text: data.plan_markdown }]);
      } else {
        setMessages(prev => [...prev, { sender: 'ai', text: `❌ Engine failed to return structural data contents.` }]);
      }
    } catch (error) {
      console.error("Orchestrator Connection Error:", error);
      setMessages(prev => [...prev, { sender: 'ai', text: `❌ Network connection bottleneck encountered. Ensure the cloud service is up and try transmitting your request again.` }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearHistory = () => {
    if (window.confirm("Purge conversational matrix log data?")) {
      localStorage.removeItem('wellness_chat_history');
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
