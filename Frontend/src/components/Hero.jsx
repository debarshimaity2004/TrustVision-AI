import { ArrowRight, Cpu, Shield, Sparkles, Zap, Image, Video } from 'lucide-react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

export default function Hero() {
    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                staggerChildren: 0.1,
                delayChildren: 0.2,
            },
        },
    };

    const itemVariants = {
        hidden: { opacity: 0, y: 30 },
        visible: {
            opacity: 1,
            y: 0,
            transition: { type: "spring", stiffness: 100, damping: 20 },
        },
    };

    return (
        <div className="relative pt-32 pb-20 sm:pt-48 sm:pb-32 overflow-hidden bg-transparent">
            {/* Minimalist Floating Background Orbs */}
            <div className="absolute top-20 left-[10%] w-96 h-96 bg-sky-200/50 rounded-full blur-[100px] animate-float-slow -z-10" />
            <div className="absolute bottom-20 right-[10%] w-[500px] h-[500px] bg-emerald-100/60 rounded-full blur-[120px] animate-float -z-10" style={{ animationDelay: '2s' }} />
            
            <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
                    
                    {/* Left Column - Text Content */}
                    <motion.div
                        variants={containerVariants}
                        initial="hidden"
                        animate="visible"
                        className="text-left"
                    >
                        {/* Premium Badge */}
                        <motion.div
                            variants={itemVariants}
                            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white shadow-sm border border-slate-200 text-slate-600 text-sm font-semibold mb-8"
                        >
                            <span className="flex h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
                            TrustVision 2.0 Live
                        </motion.div>

                        <motion.h1
                            variants={itemVariants}
                            className="text-5xl md:text-6xl lg:text-7xl font-black tracking-tight text-slate-900 mb-6 leading-[1.1]"
                        >
                            Expose <span className="gradient-text">Deepfakes</span><br/>
                            With Absolute Clarity.
                        </motion.h1>

                        <motion.p
                            variants={itemVariants}
                            className="max-w-xl text-lg md:text-xl text-slate-600 font-medium leading-relaxed mb-10"
                        >
                            State-of-the-art AI verification designed for modern security. Analyze images and video feeds in real-time to detect synthetic manipulation with unparalleled accuracy.
                        </motion.p>

                        {/* CTA Buttons */}
                        <motion.div
                            variants={itemVariants}
                            className="flex flex-col sm:flex-row gap-4"
                        >
                            <Link to="/analyze">
                                <motion.button
                                    whileHover={{ scale: 1.03, y: -2 }}
                                    whileTap={{ scale: 0.97 }}
                                    className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-8 py-4 text-base font-bold text-white bg-slate-900 hover:bg-slate-800 rounded-2xl shadow-[0_10px_30px_-10px_rgba(15,23,42,0.4)] transition-all"
                                >
                                    <Sparkles className="w-5 h-5 text-emerald-400" />
                                    Upload Media
                                </motion.button>
                            </Link>

                            <Link to="/analyze?mode=live">
                                <motion.button
                                    whileHover={{ scale: 1.03, y: -2 }}
                                    whileTap={{ scale: 0.97 }}
                                    className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-8 py-4 text-base font-bold text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 shadow-sm rounded-2xl transition-all"
                                >
                                    <Zap className="w-5 h-5 gradient-icon" />
                                    Live Webcam Feed
                                </motion.button>
                            </Link>
                        </motion.div>
                    </motion.div>

                    {/* Right Column - Visual Abstract Display */}
                    <motion.div 
                        initial={{ opacity: 0, scale: 0.9, rotateY: 15 }}
                        animate={{ opacity: 1, scale: 1, rotateY: 0 }}
                        transition={{ duration: 1, delay: 0.4, type: "spring" }}
                        className="relative hidden lg:block"
                    >
                        {/* Premium Glass Panel Mockup */}
                        <div className="relative z-10 p-2 rounded-3xl glass-card backdrop-blur-3xl shadow-2xl overflow-hidden aspect-square flex flex-col justify-between">
                            <div className="p-8 pb-0">
                                <div className="flex justify-between items-start mb-10">
                                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-sky-100 to-emerald-100 flex items-center justify-center text-emerald-600 shadow-inner">
                                        <Shield className="w-6 h-6" />
                                    </div>
                                    <div className="px-3 py-1 bg-emerald-50 text-emerald-600 font-bold text-xs rounded-full border border-emerald-100 uppercase tracking-widest flex items-center gap-2">
                                        <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                                        Secure
                                    </div>
                                </div>

                                <div className="space-y-4">
                                    <div className="h-4 w-1/3 bg-slate-200 rounded-md animate-pulse"></div>
                                    <div className="h-10 w-3/4 bg-slate-100 rounded-xl relative overflow-hidden">
                                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/50 to-transparent animate-shimmer"></div>
                                    </div>
                                </div>
                            </div>
                            
                            {/* Abstract Verification UI */}
                            <div className="relative h-64 mt-8 mx-auto w-full max-w-sm rounded-t-3xl bg-slate-50 border-t border-x border-slate-200 shadow-inner overflow-hidden">
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <div className="w-48 h-48 rounded-full border-[1.5px] border-dashed border-slate-300 animate-[spin_20s_linear_infinite]" />
                                    <div className="absolute w-32 h-32 rounded-full border-[1.5px] border-emerald-200 animate-[spin_10s_linear_infinite_reverse]" />
                                    
                                    <div className="absolute w-16 h-16 rounded-full bg-gradient-to-tr from-sky-400 to-emerald-400 shadow-[0_0_40px_rgba(52,211,153,0.4)] flex items-center justify-center pulse-ring" />
                                    <div className="absolute w-16 h-16 rounded-full bg-gradient-to-tr from-sky-500 to-emerald-500 flex items-center justify-center z-10 shadow-lg">
                                        <CheckCircle className="w-8 h-8 text-white" />
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Floating elements behind */}
                        <div className="absolute -top-6 -right-6 w-24 h-24 bg-white rounded-2xl shadow-xl flex items-center justify-center animate-float" style={{animationDelay: '1s'}}>
                            <Image className="w-8 h-8 text-sky-500" />
                        </div>
                        <div className="absolute -bottom-10 -left-10 w-28 h-28 bg-white rounded-full shadow-xl flex items-center justify-center animate-float-slow">
                            <Video className="w-10 h-10 text-emerald-500" />
                        </div>
                    </motion.div>
                </div>

                {/* Features Grid below */}
                <motion.div
                    variants={containerVariants}
                    initial="hidden"
                    animate="visible"
                    className="mt-32 grid grid-cols-1 md:grid-cols-3 gap-8"
                >
                    {[
                        {
                            icon: Cpu,
                            title: "Microsecond Analysis",
                            desc: "Real-time edge processing guarantees immediate authenticity verification.",
                            color: "sky"
                        },
                        {
                            icon: Shield,
                            title: "Cryptographic Explainability",
                            desc: "Transparent heatmaps point out exact anomalous noise structures.",
                            color: "emerald"
                        },
                        {
                            icon: Zap,
                            title: "Multi-Modal Architecture",
                            desc: "Hybrid ViT and frequency CNN models catch the most advanced fakes.",
                            color: "blue"
                        }
                    ].map((feature, index) => (
                        <motion.div
                            key={index}
                            variants={itemVariants}
                            whileHover={{ y: -8, scale: 1.02 }}
                            className="p-8 rounded-[2rem] bg-white border border-slate-100 shadow-[0_8px_30px_rgb(0,0,0,0.04)] transition-all cursor-pointer group hover:shadow-[0_20px_40px_-15px_rgba(15,23,42,0.1)]"
                        >
                            <div className={`w-14 h-14 rounded-2xl bg-${feature.color}-50 text-${feature.color}-600 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform shadow-sm`}>
                                <feature.icon className="w-7 h-7" />
                            </div>
                            <h3 className="text-xl font-bold text-slate-900 mb-3">{feature.title}</h3>
                            <p className="text-slate-500 leading-relaxed font-medium">{feature.desc}</p>
                        </motion.div>
                    ))}
                </motion.div>
            </div>
        </div>
    );
}

// Icon definition for CheckCircle since it wasn't imported above
function CheckCircle(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  )
}
