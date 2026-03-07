"use client";

import { useState, useEffect, useRef } from "react";
import { Send, Bot, User, Link as LinkIcon, ShieldAlert } from "lucide-react";

interface Fund {
    fund_id: string;
    fund_name: string;
}

interface Message {
    role: "user" | "ai";
    content: string;
    source_url?: string;
    last_updated?: string;
    refused?: boolean;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
    const [funds, setFunds] = useState<Fund[]>([]);
    const [activeFund, setActiveFund] = useState<string>("hdfc_large_cap");

    // Track messages per fund separately
    const [history, setHistory] = useState<Record<string, Message[]>>({});
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        // Fetch available funds from FastAPI backend
        fetch(`${API_BASE_URL}/funds`)
            .then(res => res.json())
            .then(data => {
                setFunds(data);
                if (data.length > 0) setActiveFund(data[0].fund_id);
            })
            .catch(console.error);
    }, []);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [history, activeFund, isLoading]);

    const currentMessages = history[activeFund] || [];

    const handleSend = async (e?: React.FormEvent) => {
        e?.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMsg = input.trim();
        setInput("");

        // Add user message to history
        setHistory(prev => ({
            ...prev,
            [activeFund]: [...(prev[activeFund] || []), { role: "user", content: userMsg }]
        }));

        setIsLoading(true);

        try {
            const res = await fetch(`${API_BASE_URL}/ask`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ fund_id: activeFund, question: userMsg })
            });

            const data = await res.json();

            setHistory(prev => ({
                ...prev,
                [activeFund]: [
                    ...(prev[activeFund] || []),
                    {
                        role: "ai",
                        content: data.answer,
                        source_url: data.source_url,
                        last_updated: data.last_updated,
                        refused: data.refused
                    }
                ]
            }));
        } catch (error) {
            setHistory(prev => ({
                ...prev,
                [activeFund]: [
                    ...(prev[activeFund] || []),
                    {
                        role: "ai",
                        content: "Network error connecting to the backend. Please ensure the FastAPI server is running.",
                        refused: true
                    }
                ]
            }));
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col md:flex-row h-screen bg-groww-dark text-groww-text overflow-hidden">

            {/* Sidebar / Mobile Header - Fund Selector */}
            <div className="w-full md:w-80 border-b md:border-b-0 md:border-r border-groww-border/30 bg-groww-dark flex flex-col pt-4 md:pt-6 shrink-0 z-10">
                <div className="px-4 md:px-6 mb-4 md:mb-8">
                    <div className="flex items-center justify-between md:justify-start gap-3">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 shrink-0">
                                <svg width="100%" height="100%" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <mask id="circle-mask">
                                        <circle cx="50" cy="50" r="50" fill="white" />
                                    </mask>
                                    <g mask="url(#circle-mask)">
                                        <rect width="100%" height="100%" fill="#5367FF" />
                                        <path d="M0 60 L35 85 L65 40 L100 60 V100 H0 Z" fill="#00D09C" />
                                    </g>
                                </svg>
                            </div>
                            <div>
                                <h1 className="font-semibold text-base md:text-lg leading-tight tracking-tight">Mutual Funds Assistant</h1>
                                <p className="text-xs text-groww-muted hidden md:block">Groww Aesthetics</p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Mobile horizontal scroller, desktop vertical list */}
                <div className="flex md:flex-col overflow-x-auto md:overflow-y-auto px-4 md:pr-4 gap-3 md:space-y-3 pb-4 md:pb-6 scrollbar-hide" style={{ msOverflowStyle: 'none', scrollbarWidth: 'none' }}>
                    <style dangerouslySetInnerHTML={{ __html: `::-webkit-scrollbar { display: none; }` }} />
                    <p className="hidden md:block px-2 text-xs font-medium text-groww-muted tracking-wider uppercase mb-2">Select Fund Context</p>

                    {funds.map(fund => (
                        <button
                            key={fund.fund_id}
                            onClick={() => setActiveFund(fund.fund_id)}
                            className={`min-w-fit md:w-full text-left p-3 md:p-4 rounded-xl transition-all duration-300 relative overflow-hidden group whitespace-nowrap
                ${activeFund === fund.fund_id
                                    ? 'bg-groww-card border border-groww-teal/50 shadow-[0_0_15px_rgba(0,208,156,0.1)]'
                                    : 'bg-groww-card/40 border border-transparent hover:bg-groww-card hover:border-groww-border'}`}
                        >
                            <div className="flex items-center gap-3">
                                <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0
                  ${activeFund === fund.fund_id ? 'bg-groww-teal/20 text-groww-teal' : 'bg-groww-border/50 text-groww-muted'}`}>
                                    <ShieldAlert className="w-4 h-4" />
                                </div>
                                <span className={`font-medium text-sm md:text-base ${activeFund === fund.fund_id ? 'text-white' : 'text-groww-muted group-hover:text-groww-text'}`}>
                                    {fund.fund_name}
                                </span>
                            </div>
                        </button>
                    ))}
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col bg-[#0A0D14] h-full overflow-hidden">

                {/* Chat History */}
                <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6 md:py-8 space-y-4 md:space-y-6">
                    {currentMessages.length === 0 ? (
                        <div className="h-full flex flex-col items-center justify-center text-center opacity-70 px-4">
                            <Bot className="w-12 h-12 md:w-16 md:h-16 text-groww-border mb-4" />
                            <h2 className="text-lg md:text-xl font-medium mb-2 text-white">Ask facts. Not advice.</h2>
                            <p className="text-sm md:text-base text-groww-muted max-w-sm">
                                I can answer factual questions about the currently selected fund based strictly on official KIM and Scheme documents.
                            </p>
                        </div>
                    ) : (
                        currentMessages.map((msg, idx) => (
                            <div key={idx} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                                <div className={`max-w-[90%] md:max-w-[75%] rounded-2xl p-4 md:p-5 ${msg.role === "user"
                                    ? "bg-groww-teal text-[#003B2C] rounded-tr-sm font-medium"
                                    : "bg-groww-card border border-groww-border/50 text-groww-text rounded-tl-sm shadow-xl"
                                    }`}>
                                    <div className="flex items-center gap-2 mb-2">
                                        {msg.role === "user" ? <User className="w-3.5 h-3.5 md:w-4 md:h-4 opacity-70" /> : <Bot className="w-3.5 h-3.5 md:w-4 md:h-4 text-groww-teal" />}
                                        <span className="text-[10px] md:text-xs font-semibold opacity-70">
                                            {msg.role === "user" ? "You" : "Assistant"}
                                        </span>
                                    </div>

                                    <div className="text-sm md:text-base leading-relaxed whitespace-pre-wrap">{msg.content}</div>

                                    {msg.role === "ai" && !msg.refused && msg.source_url && (
                                        <div className="mt-4 pt-3 border-t border-groww-border/50 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-[10px] md:text-xs text-groww-muted">
                                            <a
                                                href={msg.source_url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="flex items-center gap-1.5 hover:text-groww-teal transition-colors"
                                            >
                                                <LinkIcon className="w-3 h-3 md:w-3.5 md:h-3.5" />
                                                Source PDF Document
                                            </a>
                                            <span>Updated: {msg.last_updated}</span>
                                        </div>
                                    )}
                                    {msg.role === "ai" && msg.refused && (
                                        <div className="mt-3 text-[10px] md:text-xs text-orange-400/80 font-medium italic">
                                            This request triggered safety guardrails (advice/PII/out-of-scope).
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))
                    )}
                    {isLoading && (
                        <div className="flex justify-start">
                            <div className="bg-groww-card border border-groww-border/50 text-groww-muted rounded-2xl rounded-tl-sm p-4 text-sm animate-pulse flex items-center gap-3">
                                <Bot className="w-4 h-4" /> Retrieving context...
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Example Questions */}
                <div className="px-4 md:px-8 pb-2 opacity-80 overflow-x-auto scrollbar-hide flex gap-2">
                    {[
                        "What is the expense ratio?",
                        "What are the exit load charges?",
                        "Tell me about the investment objective",
                        "Who is the fund manager?",
                        "What is the minimum SIP amount?",
                        "Explain the riskometer and benchmark",
                        "What is the lock-in period for ELSS?",
                        "How can I request my account statement?"
                    ].map((q) => (
                        <button
                            key={q}
                            onClick={() => setInput(q)}
                            className="whitespace-nowrap bg-groww-card/60 border border-groww-border/30 px-3 py-1.5 rounded-full text-[11px] md:text-xs text-groww-muted hover:text-groww-teal hover:border-groww-teal/40 transition-all"
                        >
                            {q}
                        </button>
                    ))}
                </div>

                {/* Input Bar */}
                <div className="p-4 md:p-6 bg-gradient-to-t from-[#0A0D14] to-transparent shrink-0">
                    <form onSubmit={handleSend} className="max-w-4xl mx-auto relative">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder={`Ask a question...`}
                            disabled={isLoading}
                            className="w-full text-sm md:text-base bg-groww-card/80 border border-groww-border/80 focus:border-groww-teal/60 focus:ring-1 focus:ring-groww-teal/60 rounded-full py-3 md:py-4 pl-4 md:pl-6 pr-14 md:pr-16 text-groww-text placeholder-groww-muted outline-none transition-all shadow-lg disabled:opacity-50"
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || isLoading}
                            className="absolute right-2 top-1/2 -translate-y-1/2 bg-groww-teal text-[#003B2C] p-2.5 rounded-full hover:bg-groww-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <Send className="w-5 h-5" />
                        </button>
                    </form>
                    <div className="text-center mt-3 text-xs text-groww-muted/80">
                        AI can make mistakes. Always verify with official fund documents before investing.
                    </div>
                </div>

            </div>
        </div>
    );
}
