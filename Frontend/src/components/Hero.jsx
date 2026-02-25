import { ArrowRight, ShieldCheck, Zap, Video } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Hero() {
    return (
        <div className="relative pt-32 pb-20 sm:pt-40 sm:pb-24 overflow-hidden">
            {/* Background Glow */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-emerald-500/20 rounded-full blur-[120px] pointer-events-none" />

            <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 text-sm font-medium mb-8">
                    <span className="flex h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    Real-time Deepfake Detection Engine
                </div>

                <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight text-white mb-8">
                    Trust in Digital Media <br className="hidden sm:block" />
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-500">
                        Restored by AI.
                    </span>
                </h1>

                <p className="mt-4 max-w-2xl text-xl text-slate-400 mx-auto mb-10">
                    Enterprise-grade media authenticity verification platform. Detect manipulated images and videos with explainable AI and state-of-the-art accuracy.
                </p>

                <div className="flex flex-col sm:flex-row justify-center gap-4">
                    <Link
                        to="/analyze"
                        className="inline-flex items-center justify-center gap-2 px-8 py-4 text-base font-semibold text-white bg-emerald-600 hover:bg-emerald-500 rounded-full transition-all shadow-[0_0_20px_rgba(16,185,129,0.3)] hover:shadow-[0_0_30px_rgba(16,185,129,0.5)]"
                    >
                        Start Analyzing <ArrowRight className="w-5 h-5" />
                    </Link>
                    <button className="inline-flex items-center justify-center gap-2 px-8 py-4 text-base font-semibold text-slate-300 bg-slate-800/50 hover:bg-slate-800 border border-slate-700 rounded-full transition-all">
                        View Live Demo
                    </button>
                </div>

                {/* Floating Features Row */}
                <div className="mt-20 grid grid-cols-1 gap-6 sm:grid-cols-3 md:gap-8 border-t border-white/5 pt-12">
                    <div className="flex flex-col items-center">
                        <div className="flex items-center justify-center h-12 w-12 rounded-xl bg-blue-500/10 text-blue-400 mb-4">
                            <Zap className="h-6 w-6" />
                        </div>
                        <h3 className="text-lg font-medium text-white">Millisecond Analysis</h3>
                        <p className="mt-2 text-sm text-slate-400">Instant inference times powered by PyTorch.</p>
                    </div>
                    <div className="flex flex-col items-center">
                        <div className="flex items-center justify-center h-12 w-12 rounded-xl bg-emerald-500/10 text-emerald-400 mb-4">
                            <ShieldCheck className="h-6 w-6" />
                        </div>
                        <h3 className="text-lg font-medium text-white">Explainable AI</h3>
                        <p className="mt-2 text-sm text-slate-400">Grad-CAM heatmaps highlight manipulations.</p>
                    </div>
                    <div className="flex flex-col items-center">
                        <div className="flex items-center justify-center h-12 w-12 rounded-xl bg-purple-500/10 text-purple-400 mb-4">
                            <Video className="h-6 w-6" />
                        </div>
                        <h3 className="text-lg font-medium text-white">Video & Frame Extraction</h3>
                        <p className="mt-2 text-sm text-slate-400">Deep scan videos across every single frame.</p>
                    </div>
                </div>
            </div>
        </div>
    );
}
