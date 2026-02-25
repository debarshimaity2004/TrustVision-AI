import Navbar from '../components/Navbar';
import Hero from '../components/Hero';

export default function Home() {
    return (
        <div className="min-h-screen bg-slate-950 selection:bg-emerald-500/30">
            <Navbar />
            <main>
                <Hero />
            </main>

            {/* Decorative Gradient Line */}
            <div className="h-px w-full bg-gradient-to-r from-transparent via-emerald-500/50 to-transparent opacity-50"></div>

            {/* Footer minimal */}
            <footer className="py-8 text-center text-sm text-slate-500">
                <p>© {new Date().getFullYear()} TrustVision AI. Built for Digital Trust.</p>
            </footer>
        </div>
    );
}
