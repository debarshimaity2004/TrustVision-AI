import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Zap, Menu, X, ArrowLeft } from 'lucide-react';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function Navbar() {
    const [isOpen, setIsOpen] = useState(false);
    const location = useLocation();
    const navigate = useNavigate();

    return (
        <nav className="fixed w-full z-50 top-0 left-0 pt-4 px-4 sm:px-6 lg:px-8">
            <motion.div 
                initial={{ y: -50, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ type: "spring", stiffness: 200, damping: 20 }}
                className="max-w-7xl mx-auto glass-effect rounded-full shadow-lg border border-white/60 bg-white/70 backdrop-blur-xl"
            >
                <div className="flex items-center justify-between h-16 px-6">
                    {/* Logo */}
                    <motion.div
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="flex items-center gap-3"
                    >
                        <div className="relative flex items-center justify-center w-10 h-10 rounded-full bg-gradient-to-tr from-sky-400 to-emerald-400 shadow-md">
                            <div className="absolute inset-0 rounded-full animate-pulse-ring bg-emerald-400/30"></div>
                            <Zap className="h-5 w-5 text-white z-10" fill="currentColor" />
                        </div>
                        <Link to="/" className="font-bold text-xl tracking-tight text-slate-800">
                            Trust<span className="gradient-text font-black">Vision</span>
                        </Link>
                    </motion.div>

                    {/* Desktop Menu */}
                    <div className="hidden md:flex items-center gap-1">
                        {location.pathname === '/' ? (
                            <Link to="/analyze">
                                <motion.div
                                    whileHover={{ scale: 1.05 }}
                                    whileTap={{ scale: 0.95 }}
                                    className="relative group px-6 py-2.5 rounded-full bg-slate-900 border border-slate-800 shadow-[0_4px_14px_0_rgb(0,0,0,10%)] hover:shadow-[0_6px_20px_rgba(15,23,42,23%)] transition-all flex items-center gap-2 overflow-hidden"
                                >
                                    <div className="absolute inset-0 bg-gradient-to-r from-sky-500 to-emerald-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                                    <span className="relative text-white font-medium text-sm flex items-center gap-2">
                                        <span className="flex h-2 w-2 rounded-full bg-emerald-400"></span>
                                        Launch Detector
                                    </span>
                                </motion.div>
                            </Link>
                        ) : (
                            <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={() => navigate(-1)}
                                className="px-6 py-2.5 rounded-full bg-white/80 hover:bg-white border text-slate-700 shadow-sm transition-all flex items-center gap-2 font-medium text-sm"
                            >
                                <ArrowLeft className="w-4 h-4" />
                                Go Back
                            </motion.button>
                        )}
                    </div>

                    {/* Mobile Menu Button */}
                    <div className="flex md:hidden">
                        <motion.button
                            whileTap={{ scale: 0.95 }}
                            onClick={() => setIsOpen(!isOpen)}
                            className="p-2 rounded-full text-slate-600 hover:text-slate-900 hover:bg-slate-100 focus:outline-none transition-colors"
                        >
                            {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
                        </motion.button>
                    </div>
                </div>
            </motion.div>

            {/* Mobile Menu Dropdown */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -20, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -20, scale: 0.95 }}
                        transition={{ duration: 0.2 }}
                        className="md:hidden absolute top-24 left-4 right-4 glass-card rounded-2xl shadow-xl overflow-hidden border border-white"
                    >
                        <div className="p-4 space-y-2">
                            {location.pathname === '/' ? (
                                <Link 
                                    to="/analyze" 
                                    onClick={() => setIsOpen(false)}
                                    className="flex justify-center w-full py-3 px-4 rounded-xl font-bold bg-gradient-to-r from-sky-500 to-emerald-500 text-white shadow-md text-center"
                                >
                                    Launch Detector
                                </Link>
                            ) : (
                                <button 
                                    onClick={() => { setIsOpen(false); navigate(-1); }} 
                                    className="flex items-center justify-center gap-2 w-full py-3 px-4 rounded-xl font-bold bg-white text-slate-800 shadow-sm border text-center"
                                >
                                    <ArrowLeft className="w-5 h-5" />
                                    Go Back
                                </button>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </nav>
    );
}
