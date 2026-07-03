import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import './index.css';

export default function App() {
  const BACKEND_URL = import.meta.env.VITE_API_URL || 'https://wellness-managed-agents-git-436702918308.asia-south1.run.app';
  const [screen, setScreen] = useState('onboarding');
  const [profile, setProfile] = useState({ userId: '', age: '', weight: '', injuries: 'None', goals: '' });
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('Orchestrating agents...');
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
      await fetch(`${BACKEND_URL}/api/save-profile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(profile),
      });

      setMessages([{ sender: 'ai', text: `Welcome **${profile.userId}**! Your multi-agent profile has been initialized securely. Ask me anything about building performance prescriptions, aligning posture mechanics, or managing target sports nutrition!` }]);
      setScreen('chat');
    } catch (error) {
      console.error("Backend unreachable, entering preview mode.", error);
      setMessages([{ sender: 'ai', text: `⚠️ **Running in local bypass mode.** (Backend at ${BACKEND_URL} was offline). Feel free to test the interface layout!` }]);
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

  if (screen === 'onboarding') {
    return (
      <div className="min-h-screen bg-slate-950 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black flex items-center justify-center p-4 text-white font-sans selection:bg-emerald-500/30">
        
        {/* CSS Animation Injector for Button Shimmer */}
        <style>{`
          @keyframes shimmer {
            100% { transform: translateX(100%); }
          }
          .animate-shimmer {
            animation: shimmer 2s infinite;
          }
        `}</style>

        {/* Container Card with subtle backdrop blur and neon emerald borders */}
        <div className="relative bg-slate-900/60 backdrop-blur-xl p-8 rounded-3xl shadow-2xl w-full max-w-lg border border-slate-800 hover:border-slate-700/80 transition-all duration-500">
          
          {/* Glow Effects */}
          <div className="absolute -top-12 -left-12 w-40 h-40 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
          <div className="absolute -bottom-12 -right-12 w-40 h-40 bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />

          {/* Decorative Top Accent Line */}
          <div className="absolute top-0 left-1/4 right-1/4 h-[2px] bg-gradient-to-r from-transparent via-emerald-500 to-transparent" />

          {/* Header Block */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 text-2xl mb-4 shadow-inner shadow-emerald-500/10">
              🧬
            </div>
            <h2 className="text-3xl font-black tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
              Setup Your Engine
            </h2>
            <p className="text-slate-400 text-sm max-w-xs mx-auto mt-2 leading-relaxed">
              Let's gather your vitals to synchronize the multi-agent wellness lattice.
            </p>
          </div>

          {/* Interactive Form */}
          <form onSubmit={handleOnboardingSubmit} className="space-y-5">
            
            {/* User ID / Name */}
            <div className="group relative">
              <label className="block text-[11px] font-bold tracking-wider uppercase text-slate-400 group-focus-within:text-emerald-400 transition-colors mb-1.5 pl-1">
                User ID / Name
              </label>
              <input 
                type="text" 
                required 
                className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-white placeholder-slate-600 focus:outline-none focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all shadow-inner" 
                placeholder="e.g. ankur" 
                value={profile.userId} 
                onChange={e => setProfile({...profile, userId: e.target.value})} 
              />
            </div>

            {/* Grid Layout for Metrics */}
            <div className="grid grid-cols-2 gap-4">
              <div className="group">
                <label className="block text-[11px] font-bold tracking-wider uppercase text-slate-400 group-focus-within:text-emerald-400 transition-colors mb-1.5 pl-1">
                  Age
                </label>
                <input 
                  type="number" 
                  required 
                  className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-white placeholder-slate-600 focus:outline-none focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all shadow-inner" 
                  placeholder="24" 
                  value={profile.age} 
                  onChange={e => setProfile({...profile, age: e.target.value})} 
                />
              </div>
              <div className="group">
                <label className="block text-[11px] font-bold tracking-wider uppercase text-slate-400 group-focus-within:text-emerald-400 transition-colors mb-1.5 pl-1">
                  Weight (KG)
                </label>
                <input 
                  type="number" 
                  required 
                  className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-white placeholder-slate-600 focus:outline-none focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all shadow-inner" 
                  placeholder="61" 
                  value={profile.weight} 
                  onChange={e => setProfile({...profile, weight: e.target.value})} 
                />
              </div>
            </div>

            {/* Injuries */}
            <div className="group">
              <label className="block text-[11px] font-bold tracking-wider uppercase text-slate-400 group-focus-within:text-emerald-400 transition-colors mb-1.5 pl-1">
                Injuries / Medical Conditions
              </label>
              <input 
                type="text" 
                className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-white placeholder-slate-600 focus:outline-none focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all shadow-inner" 
                placeholder="e.g. none, lower back stiffness" 
                value={profile.injuries} 
                onChange={e => setProfile({...profile, injuries: e.target.value})} 
              />
            </div>

            {/* Goals */}
            <div className="group">
              <label className="block text-[11px] font-bold tracking-wider uppercase text-slate-400 group-focus-within:text-emerald-400 transition-colors mb-1.5 pl-1">
                Primary Wellness Goals
              </label>
              <textarea 
                className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-white placeholder-slate-600 h-24 resize-none focus:outline-none focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all shadow-inner leading-relaxed" 
                placeholder="What target benchmarks are we trying to optimize?" 
                value={profile.goals} 
                onChange={e => setProfile({...profile, goals: e.target.value})} 
              />
            </div>

            {/* Premium Animated Call to Action Button */}
            <button 
              type="submit" 
              className="w-full relative group/btn bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 active:scale-[0.99] transition-all text-white font-bold p-3.5 rounded-xl shadow-lg shadow-emerald-950/40 mt-4 overflow-hidden flex items-center justify-center space-x-2"
            >
              <span className="relative z-10">Initialize Agent Mesh</span>
              <span className="relative z-10 text-lg group-hover/btn:translate-x-1 transition-transform duration-300">⚡</span>
              <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/15 to-transparent -translate-x-full group-hover/btn:animate-shimmer" />
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      
      {/* Left Column: Glassmorphic Core Context Metrics Sidebar */}
      <aside className="hidden md:flex flex-col w-72 bg-slate-900/40 border-r border-slate-800/60 p-6 space-y-6 backdrop-blur-md">
        <div className="flex items-center space-x-3 pb-4 border-b border-slate-800/60">
          <div className="w-3 h-3 rounded-full bg-emerald-500 shadow-lg shadow-emerald-500/50 animate-pulse" />
          <h1 className="font-black text-sm tracking-widest uppercase bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
            A.I. Core Matrix
          </h1>
        </div>

        <div className="space-y-4 flex-1">
          <div className="bg-slate-950/50 border border-slate-800/80 rounded-xl p-3.5 shadow-inner">
            <span className="block text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-1">Active Subject</span>
            <span className="text-sm font-bold text-emerald-400">{profile.userId}</span>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-950/50 border border-slate-800/80 rounded-xl p-3 shadow-inner">
              <span className="block text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-0.5">Age Vector</span>
              <span className="text-xs font-bold text-slate-200">{profile.age} yrs</span>
            </div>
            <div className="bg-slate-950/50 border border-slate-800/80 rounded-xl p-3 shadow-inner">
              <span className="block text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-0.5">Mass Index</span>
              <span className="text-xs font-bold text-slate-200">{profile.weight} KG</span>
            </div>
          </div>

          <div className="bg-slate-950/50 border border-slate-800/80 rounded-xl p-3.5 shadow-inner">
            <span className="block text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-1">Risk Factors</span>
            <span className="text-xs font-medium text-amber-400/90 leading-relaxed block max-h-16 overflow-y-auto">{profile.injuries}</span>
          </div>

          <div className="bg-slate-950/50 border border-slate-800/80 rounded-xl p-3.5 shadow-inner">
            <span className="block text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-1">Target Directives</span>
            <span className="text-xs font-medium text-slate-300 leading-relaxed block max-h-28 overflow-y-auto">{profile.goals || "No target objectives declared."}</span>
          </div>
        </div>

        <div className="text-[10px] text-slate-500 text-center tracking-wider border-t border-slate-800/60 pt-4">
          SYSTEM VERSION 2.4.0 // LIVE
        </div>
      </aside>

      {/* Right Column: Dynamic Terminal Communication Stream */}
      <div className="flex-1 flex flex-col min-w-0 bg-slate-950 relative">
        <div className="absolute top-0 right-1/4 w-80 h-80 bg-teal-500/[0.02] rounded-full blur-3xl pointer-events-none" />

        {/* Floating Top Mini Header */}
        <header className="bg-slate-900/30 backdrop-blur-md border-b border-slate-800/60 p-4 flex justify-between items-center z-10">
          <div className="flex items-center space-x-3 md:hidden">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <h1 className="font-bold text-sm text-slate-200 tracking-wide">Wellness Assistant</h1>
          </div>
          <div className="hidden md:block text-xs font-medium text-slate-400">
            Secure Endpoint Connection Pipeline: <span className="font-mono text-emerald-400 text-[11px] bg-slate-900/80 px-2 py-0.5 rounded border border-slate-800">Operational</span>
          </div>
          <span className="text-[10px] font-bold tracking-wider uppercase bg-slate-800/60 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700/50 shadow-inner">
            Context Node: <span className="text-emerald-400 font-mono font-bold">{profile.userId}</span>
          </span>
        </header>

        {/* Message Loop Render Canvas */}
        <main className="flex-1 overflow-y-auto p-6 space-y-6 max-w-3xl w-full mx-auto scrollbar-thin scrollbar-thumb-slate-800">
          {messages.map((msg, index) => (
            <div key={index} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}>
              <div className={`max-w-2xl rounded-2xl p-4 shadow-md transition-all duration-200 ${
                msg.sender === 'user' 
                  ? 'bg-gradient-to-br from-emerald-600 to-teal-700 text-white rounded-br-none shadow-emerald-950/20 font-medium' 
                  : 'bg-slate-900/90 border border-slate-800/80 text-slate-100 rounded-bl-none shadow-black/40'
              }`}>
                <div className="text-sm space-y-3 leading-relaxed tracking-wide selection:bg-white/20">
                  {msg.sender === 'user' ? (
                    <span className="block font-semibold tracking-wide">{msg.text}</span>
                  ) : (
                    <div className="prose prose-invert prose-emerald max-w-none text-slate-200 prose-headings:text-white prose-headings:font-bold prose-headings:mt-2 prose-headings:mb-1 prose-p:my-1 prose-ul:list-disc prose-ul:pl-4 prose-strong:text-emerald-400 prose-strong:font-bold">
                      <ReactMarkdown>{msg.text}</ReactMarkdown>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          
          {/* Dynamic Telemetry Status Loader */}
          {isLoading && (
            <div className="flex justify-start transition-all duration-300 ease-in-out">
              <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl rounded-bl-none px-4 py-3.5 text-slate-300 flex items-center space-x-3.5 shadow-lg border-dashed border-emerald-500/20 backdrop-blur-sm">
                <div className="flex space-x-1.5 items-center">
                  <span className="w-2 h-2 bg-gradient-to-r from-emerald-400 to-teal-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 bg-gradient-to-r from-emerald-400 to-teal-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-2 h-2 bg-gradient-to-r from-emerald-400 to-teal-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
                <span className="text-xs font-semibold text-slate-400 tracking-wide animate-pulse">{loadingStatus}</span>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </main>

        {/* Input Control Box */}
        <footer className="bg-slate-900/20 backdrop-blur-md border-t border-slate-800/60 p-4">
          <form onSubmit={handleSendMessage} className="max-w-3xl mx-auto flex items-center space-x-3">
            <div className="flex-1 relative flex items-center">
              <input
                type="text"
                value={input}
                disabled={isLoading}
                onChange={e => setInput(e.target.value)}
                placeholder="Request tailored metrics, posture routines, or micro plans..."
                className="w-full bg-slate-950/80 border border-slate-800 rounded-xl py-3.5 pl-4 pr-12 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/5 disabled:opacity-50 transition-all shadow-inner"
              />
              <span className="absolute right-4 text-slate-600 text-xs font-mono select-none hidden sm:inline">⌘K</span>
            </div>
            <button 
              type="submit" 
              disabled={isLoading || !input.trim()} 
              className="bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 disabled:opacity-30 disabled:pointer-events-none transition-all px-5 py-3.5 rounded-xl font-bold text-sm text-white shadow-md shadow-emerald-950/20 shrink-0"
            >
              Transmit
            </button>
          </form>
        </footer>
      </div>
    </div>
  );
}
