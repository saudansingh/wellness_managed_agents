import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import './index.css';

export default function App() {
  const BACKEND_URL = import.meta.env.VITE_API_URL || 'https://wellness-managed-agents-git-436702918308.asia-south1.run.app';
  
  // State Initialization with LocalStorage Persistence
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

  // Sync state mutations to local storage arrays
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
    try {
      await fetch(`${BACKEND_URL}/api/save-profile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(profile),
      });

      // Only initialize welcome message if conversation stream history is clear
      if (messages.length === 0) {
        setMessages([{ sender: 'ai', text: `Welcome **${profile.userId}**! Your multi-agent profile has been initialized securely. Ask me anything about building performance prescriptions, aligning posture mechanics, or managing target sports nutrition!` }]);
      }
      setScreen('chat');
    } catch (error) {
      console.error("Backend unreachable, entering preview mode.", error);
      if (messages.length === 0) {
        setMessages([{ sender: 'ai', text: `⚠️ **Running in local bypass mode.** (Backend at ${BACKEND_URL} was offline). Feel free to test the interface layout!` }]);
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

    const textLower = userMessage.toLowerCase();
    if (textLower.includes('diet') || textLower.includes('eat') || textLower.includes('weight') || textLower.includes('calorie')) {
      setLoadingStatus('🥗 Clinical Sports Dietitian is analyzing macro architecture...');
    } else if (textLower.includes('yoga') || textLower.includes('stretch') || textLower.includes('asana')) {
      setLoadingStatus('🧘 Yoga Therapist is aligning posture variations & sequencing...');
    } else {
      setLoadingStatus('🏋️ Personal Trainer is building performance prescription metrics...');
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: profile.userId,
          user_message: userMessage
        }),
      });
      const data = await response.json();
      setMessages(prev => [...prev, { sender: 'ai', text: data.plan_markdown }]);
    } catch (error) {
      setMessages(prev => [...prev, { sender: 'ai', text: `❌ Connection error. Failed connection attempt to endpoint: ${BACKEND_URL}/api/chat` }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Explicit action reset to clear active tracking logs
  const handleClearHistory = () => {
    if (window.confirm("Are you sure you want to completely erase the agent memory cache?")) {
      localStorage.removeItem('wellness_chat_history');
      setMessages([]);
    }
  };

  if (screen === 'onboarding') {
    return (
      <div className="min-h-screen bg-slate-950 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black flex items-center justify-center p-4 text-white font-sans selection:bg-emerald-500/30">
        <style>{`
          @keyframes shimmer { 100% { transform: translateX(100%); } }
          .animate-shimmer { animation: shimmer 2s infinite; }
        `}</style>
        
        <div className="relative bg-slate-900/90 p-8 rounded-3xl shadow-2xl w-full max-w-lg border border-slate-800 transition-all duration-500">
          <div className="absolute top-0 left-1/4 right-1/4 h-[2px] bg-gradient-to-r from-transparent via-emerald-500 to-transparent" />
          
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 text-2xl mb-4">🧬</div>
            <h2 className="text-3xl font-black tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">Setup Your Engine</h2>
            <p className="text-slate-400 text-sm mt-2 leading-relaxed">Let's gather your vitals to synchronize the multi-agent wellness lattice.</p>
          </div>

          <form onSubmit={handleOnboardingSubmit} className="space-y-5">
            <div>
              <label className="block text-[11px] font-bold tracking-wider uppercase text-slate-400 mb-1.5 pl-1">User ID / Name</label>
              <input type="text" required className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-white placeholder-slate-600 focus:outline-none focus:border-emerald-500 transition-all" placeholder="e.g. ankur" value={profile.userId} onChange={e => setProfile({...profile, userId: e.target.value})} />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-[11px] font-bold tracking-wider uppercase text-slate-400 mb-1.5 pl-1">Age</label>
                <input type="number" required className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-white placeholder-slate-600 focus:outline-none focus:border-emerald-500 transition-all" placeholder="24" value={profile.age} onChange={e => setProfile({...profile, age: e.target.value})} />
              </div>
              <div>
                <label className="block text-[11px] font-bold tracking-wider uppercase text-slate-400 mb-1.5 pl-1">Weight (KG)</label>
                <input type="number" required className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-white placeholder-slate-600 focus:outline-none focus:border-emerald-500 transition-all" placeholder="61" value={profile.weight} onChange={e => setProfile({...profile, weight: e.target.value})} />
              </div>
            </div>

            <div>
              <label className="block text-[11px] font-bold tracking-wider uppercase text-slate-400 mb-1.5 pl-1">Injuries / Medical Conditions</label>
              <input type="text" className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-white placeholder-slate-600 focus:outline-none focus:border-emerald-500 transition-all" placeholder="e.g. none, lower back stiffness" value={profile.injuries} onChange={e => setProfile({...profile, injuries: e.target.value})} />
            </div>

            <div>
              <label className="block text-[11px] font-bold tracking-wider uppercase text-slate-400 mb-1.5 pl-1">Primary Wellness Goals</label>
              <textarea className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-white placeholder-slate-600 h-24 resize-none focus:outline-none focus:border-emerald-500 transition-all leading-relaxed" placeholder="What target benchmarks are we trying to optimize?" value={profile.goals} onChange={e => setProfile({...profile, goals: e.target.value})} />
            </div>

            <button type="submit" className="w-full relative group bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 transition-all text-white font-bold p-3.5 rounded-xl shadow-lg shadow-emerald-950/40 mt-4 overflow-hidden flex items-center justify-center space-x-2">
              <span className="relative z-10">Initialize Agent Mesh</span>
              <span className="relative z-10 text-lg">⚡</span>
              <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/15 to-transparent -translate-x-full group-hover:animate-shimmer" />
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      {/* Sidebar Profile Monitoring Layout */}
      <aside className="hidden md:flex flex-col w-72 bg-slate-900 border-r border-slate-800 p-6 space-y-6">
        <div className="flex items-center space-x-3 pb-4 border-b border-slate-800">
          <div className="w-3 h-3 rounded-full bg-emerald-500 shadow-lg shadow-emerald-500/50 animate-pulse" />
          <h1 className="font-black text-sm tracking-widest uppercase bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">A.I. Core Matrix</h1>
        </div>

        <div className="space-y-4 flex-1">
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3.5">
            <span className="block text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-1">Active Subject</span>
            <span className="text-sm font-bold text-emerald-400">{profile.userId}</span>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-3">
              <span className="block text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-0.5">Age Vector</span>
              <span className="text-xs font-bold text-slate-200">{profile.age} yrs</span>
            </div>
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-3">
              <span className="block text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-0.5">Mass Index</span>
              <span className="text-xs font-bold text-slate-200">{profile.weight} KG</span>
            </div>
          </div>

          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3.5">
            <span className="block text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-1">Risk Factors</span>
            <span className="text-xs font-medium text-amber-400/90 leading-relaxed max-h-16 overflow-y-auto block">{profile.injuries}</span>
          </div>

          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3.5">
            <span className="block text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-1">Target Directives</span>
            <span className="text-xs font-medium text-slate-300 leading-relaxed max-h-24 overflow-y-auto block">{profile.goals || "None declared."}</span>
          </div>
        </div>

        {/* Clear Data Trigger Action */}
        <button onClick={handleClearHistory} className="w-full text-center text-[10px] uppercase tracking-widest text-rose-400 hover:text-rose-300 transition-colors p-2 rounded-lg border border-rose-950/40 hover:bg-rose-950/20 font-bold">
          Purge Chat Memory
        </button>
      </aside>

      {/* Main Terminal Window Frame */}
      <div className="flex-1 flex flex-col min-w-0 bg-slate-950">
        <header className="bg-slate-900 border-b border-slate-800 p-4 flex justify-between items-center z-10 shadow-md">
          {/* 🛠️ BACK BUTTON CONFIGURATION */}
          <button 
            onClick={() => setScreen('onboarding')} 
            className="flex items-center space-x-2 text-xs font-bold text-slate-400 hover:text-white bg-slate-950 px-3 py-2 rounded-lg border border-slate-800 hover:border-slate-700 transition-all shadow-inner"
          >
            <span>←</span> <span>Modify Profile</span>
          </button>
          
          <span className="text-[10px] font-bold tracking-wider uppercase bg-slate-950 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-800">
            Node: <span className="text-emerald-400 font-mono font-bold">{profile.userId}</span>
          </span>
        </header>

        {/* Conversation Track Loop Stream */}
        <main className="flex-1 overflow-y-auto p-6 space-y-6 max-w-3xl w-full mx-auto">
          {messages.map((msg, index) => (
            <div key={index} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-2xl rounded-2xl p-4 shadow-md ${
                msg.sender === 'user' 
                  ? 'bg-gradient-to-br from-emerald-600 to-teal-700 text-white rounded-br-none' 
                  : 'bg-slate-900 border border-slate-800 text-slate-100 rounded-bl-none'
              }`}>
                <div className="text-sm space-y-2 leading-relaxed tracking-wide">
                  {msg.sender === 'user' ? (
                    <span className="block font-semibold">{msg.text}</span>
                  ) : (
                    <div className="prose prose-invert prose-emerald max-w-none text-slate-200 prose-headings:text-white prose-strong:text-emerald-400">
                      <ReactMarkdown>{msg.text}</ReactMarkdown>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-slate-900 border border-slate-800 rounded-2xl rounded-bl-none px-4 py-3 text-slate-300 flex items-center space-x-3 shadow-lg border-dashed border-emerald-500/20">
                <div className="flex space-x-1 items-center">
                  <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
                <span className="text-xs font-semibold text-slate-400 tracking-wide animate-pulse">{loadingStatus}</span>
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
              placeholder="Request custom metrics, routines, or nutritional plans..."
              className="flex-1 bg-slate-950 border border-slate-800 rounded-xl py-3.5 px-4 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 transition-all shadow-inner"
            />
            <button type="submit" disabled={isLoading || !input.trim()} className="bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 disabled:opacity-30 px-5 py-3.5 rounded-xl font-bold text-sm text-white shadow-md transition-all shrink-0">
              Transmit
            </button>
          </form>
        </footer>
      </div>
    </div>
  );
}
