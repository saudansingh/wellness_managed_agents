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

      setMessages([{ sender: 'ai', text: `Welcome **${profile.userId}**! Your profile is loaded. Ask me anything about your fitness or diet targets!` }]);
      setScreen('chat');
    } catch (error) {
      console.error("Backend unreachable, entering preview mode.", error);
      // 🛠️ FIX: Dynamic error message string tracking target environment
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
      // 🛠️ FIX: Target endpoint switched to /api/chat instead of duplicating save-profile
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
      // 🛠️ FIX: Dynamic execution context trace parameter
      setMessages(prev => [...prev, { sender: 'ai', text: `❌ Connection error. Failed connection attempt to endpoint: ${BACKEND_URL}/api/chat` }]);
    } finally {
      setIsLoading(false);
    }
  };

  if (screen === 'onboarding') {
  return (
    <div className="min-h-screen bg-slate-950 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black flex items-center justify-center p-4 text-white font-sans selection:bg-emerald-500/30">
      
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
            <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover/btn:animate-[shimmer_1.5s_infinite]" />
          </button>
        </form>
      </div>
    </div>
  );
}
        
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
