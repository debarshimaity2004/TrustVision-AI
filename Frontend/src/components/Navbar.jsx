import { Link } from 'react-router-dom';
import { Shield, Menu, X } from 'lucide-react';
import { useState } from 'react';

export default function Navbar() {
    const [isOpen, setIsOpen] = useState(false);

    return (
        <nav className="fixed w-full z-50 bg-slate-900/80 backdrop-blur-md border-b border-white/10">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    <div className="flex items-center gap-2">
                        <Shield className="h-8 w-8 text-emerald-400" />
                        <Link to="/" className="text-xl font-bold tracking-tight text-white">
                            TrustVision<span className="text-emerald-400">.ai</span>
                        </Link>
                    </div>

                    <div className="hidden md:block">
                        <div className="ml-10 flex items-baseline space-x-8">
                
                            <Link
                                to="/analyze"
                                className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/50 hover:bg-emerald-500 hover:text-white transition-all px-4 py-2 rounded-full text-sm font-medium shadow-[0_0_15px_rgba(16,185,129,0.2)]"
                            >
                                Launch Dashboard
                            </Link>
                        </div>
                    </div>

                    <div className="-mr-2 flex md:hidden">
                        <button
                            onClick={() => setIsOpen(!isOpen)}
                            className="inline-flex items-center justify-center p-2 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 focus:outline-none"
                        >
                            {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
                        </button>
                    </div>
                </div>
            </div>

            {/* Mobile menu */}
            {isOpen && (
                <div className="md:hidden bg-slate-900 border-b border-white/10">
                    <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
                        <a href="#platform" className="text-gray-300 hover:text-white block px-3 py-2 rounded-md text-base font-medium">Platform</a>
                        <a href="#technology" className="text-gray-300 hover:text-white block px-3 py-2 rounded-md text-base font-medium">Technology</a>
                        <Link to="/analyze" className="text-emerald-400 font-bold block px-3 py-2 rounded-md text-base">Launch Dashboard</Link>
                    </div>
                </div>
            )}
        </nav>
    );
}
