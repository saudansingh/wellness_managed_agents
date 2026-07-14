import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';

// Configurable so you're not hardcoded to localhost once this deploys.
// Vite: set VITE_API_URL in your build env / .env file.
// (If you're on Create React App instead, swap this for process.env.REACT_APP_API_URL.)
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

      // fetch() does NOT throw on 4xx/5xx — check response.ok explicitly,
      // otherwise a failed save silently continues into the chat screen.
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
    // Kept intentionally generic — the backend now decides which specialist(s)
    // actually run, so guessing "Trainer/Yogi/Dietitian" here from keywords
    // alone was frequently wrong (e.g. showed the trainer label for "hi").
    setLoadingStatus('Thinking...');

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

  if (screen === 'onboarding') {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center p-6 text-white font-sans">
        <div className="bg-slate-800 p-8 rounded-2xl shadow-xl w-full max-w-md border border-slate-700">
          <h2 className="text-2xl font-bold text-center mb-2">📋 Setup Your Profile</h2>
          <p className="text-slate-400 text-sm text-center mb-6">Let's gather your metrics to tailor your responses.</p>
          <form onSubmit={handleOnboardingSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold uppercase text-slate-400 mb-1">User ID / Name</label>
              <input type="text" required className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white focus:outline-none focus:border-emerald-500" placeholder="e.g. ankur" value={profile.userId} onChange={e => setProfile({...profile, userId: e.target.value})} />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold uppercase text-slate-400 mb-1">Age</label>
                <input type="number" required className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white focus:outline-none focus:border-emerald-500" placeholder="24" value={profile.age} onChange={e => setProfile({...profile, age: e.target.value})} />
              </div>
              <div>
                <label className="block text-xs font-semibold uppercase text-slate-400 mb-1">Weight (KG)</label>
                <input type="number" required className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white focus:outline-none focus:border-emerald-500" placeholder="61" value={profile.weight} onChange={e => setProfile({...profile, weight: e.target.value})} />
              </div>
            </div>
            <div>
              <label className="block text-xs font-semibold uppercase text-slate-400 mb-1">Injuries / Conditions</label>
              <input type="text" className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white focus:outline-none focus:border-emerald-500" placeholder="e.g. none, knee pain" value={profile.injuries} onChange={e => setProfile({...profile, injuries: e.target.value})} />
            </div>
            <div>
              <label className="block text-xs font-semibold uppercase text-slate-400 mb-1">Goals</label>
              <textarea className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white h-20 resize-none focus:outline-none focus:border-emerald-500" placeholder="What are you trying to achieve?" value={profile.goals} onChange={e => setProfile({...profile, goals: e.target.value})} />
            </div>
            <button type="submit" className="w-full bg-emerald-600 hover:bg-emerald-500 transition-colors text-white font-semibold p-3 rounded-lg shadow-md mt-2">
              Start Conversation
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 font-sans">
      <header className="bg-slate-900 border-b border-slate-800 p-4 flex justify-between items-center shadow-md">
        <div className="flex items-center space-x-3">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
          <h1 className="font-bold text-base tracking-wide">Wellness Assistant</h1>
        </div>
        <span className="text-xs bg-slate-800 px-3 py-1.5 rounded-full border border-slate-700 text-slate-400">
          User Context: <b className="text-emerald-400">{profile.userId}</b>
        </span>
      </header>

      <main className="flex-1 overflow-y-auto p-6 space-y-4 max-w-3xl w-full mx-auto">

{messages.map((msg, index) => (
  <div key={index} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
    <div className={`max-w-xl rounded-2xl px-4 py-3 shadow-sm ${
      msg.sender === 'user'
        ? 'bg-emerald-600 text-white rounded-br-none shadow-emerald-900/20'
        : 'bg-slate-900 border border-slate-800 text-slate-200 rounded-bl-none'
    }`}>
      <div className="markdown-content text-sm space-y-2 leading-relaxed">
        {msg.sender === 'user' ? (
          <strong className="block text-base font-extrabold tracking-wide text-white">
            {msg.text}
          </strong>
        ) : (
          <ReactMarkdown>{msg.text}</ReactMarkdown>
        )}
      </div>
    </div>
  </div>
))}

        {isLoading && (
          <div className="flex justify-start transition-all duration-300 ease-in-out">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl rounded-bl-none px-4 py-3 text-slate-300 flex items-center space-x-3 shadow-md border-dashed border-emerald-500/30">
              <div className="flex space-x-1 items-center">
                <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
              <span className="text-xs font-medium text-slate-400 tracking-wide animate-pulse">{loadingStatus}</span>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </main>

      <footer className="bg-slate-900 border-t border-slate-800 p-4">
        <form onSubmit={handleSendMessage} className="max-w-3xl mx-auto flex items-center space-x-3">
          <input
            type="text"
            value={input}
            disabled={isLoading}
            onChange={e => setInput(e.target.value)}
            placeholder="Type your wellness or diet query here..."
            className="flex-1 bg-slate-950 border border-slate-800 rounded-xl p-3.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 disabled:opacity-50"
          />
          <button type="submit" disabled={isLoading || !input.trim()} className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 transition-colors px-5 py-3.5 rounded-xl font-semibold text-sm text-white">
            Send
          </button>
        </form>
      </footer>
    </div>
  );
}
