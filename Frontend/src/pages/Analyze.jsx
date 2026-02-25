import { useState } from 'react';
import Navbar from '../components/Navbar';
import { UploadCloud, FileType2, X, Activity, CheckCircle, AlertTriangle, ShieldAlert } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';

export default function Analyze() {
    const [file, setFile] = useState(null);
    const [isDragActive, setIsDragActive] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [result, setResult] = useState(null);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") setIsDragActive(true);
        else if (e.type === "dragleave") setIsDragActive(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const uploadedFile = e.dataTransfer.files[0];
            if (uploadedFile.size > 50 * 1024 * 1024) {
                toast.error("File size exceeds 50MB limit");
                return;
            }
            setFile(uploadedFile);
            toast.success("File uploaded successfully");
        }
    };

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            const uploadedFile = e.target.files[0];
            if (uploadedFile.size > 50 * 1024 * 1024) {
                toast.error("File size exceeds 50MB limit");
                return;
            }
            setFile(uploadedFile);
            toast.success("File uploaded successfully");
        }
    };

    const removeFile = () => {
        setFile(null);
        setResult(null);
    };

    const startAnalysis = () => {
        if (!file) {
            toast.error("Please upload a file first");
            return;
        }
        setIsAnalyzing(true);
        setResult(null);
        toast.loading("Scanning artifacts...", { id: "analysis-toast" });

        // Mocking an inference API delay
        setTimeout(() => {
            setIsAnalyzing(false);
            toast.success("Analysis complete!", { id: "analysis-toast" });
            // Mock Results
            setResult({
                score: 87.4,
                prediction: "REAL",
                risk: "LOW",
                time: "1.2s",
                model: "ResNet-50 v2"
            });
        }, 2500);
    };

    return (
        <div className="min-h-screen bg-slate-950 font-sans selection:bg-emerald-500/30 pb-20">
            <Navbar />

            <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 pt-32">
                <div className="text-center mb-12">
                    <h1 className="text-4xl font-extrabold text-white mb-4">Media Authenticity Scanner</h1>
                    <p className="text-slate-400">Upload an image or video to determine if it has been AI-generated or manipulated.</p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

                    {/* Left Column: Upload Area */}
                    <div className="flex flex-col gap-6">
                        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
                            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                                <FileType2 className="w-5 h-5 text-emerald-400" />
                                Upload Media
                            </h2>

                            {!file ? (
                                <div
                                    className={`border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center transition-all ${isDragActive ? 'border-emerald-500 bg-emerald-500/5' : 'border-slate-700 bg-slate-800/50 hover:border-slate-500 hover:bg-slate-800'}`}
                                    onDragEnter={handleDrag}
                                    onDragLeave={handleDrag}
                                    onDragOver={handleDrag}
                                    onDrop={handleDrop}
                                >
                                    <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mb-4 text-emerald-400">
                                        <UploadCloud className="w-8 h-8" />
                                    </div>
                                    <p className="text-slate-300 font-medium mb-1">Drag and drop your file here</p>
                                    <p className="text-slate-500 text-sm mb-6">Supports JPG, PNG, MP4 (Max 50MB)</p>

                                    <input type="file" id="file-upload" className="hidden" onChange={handleChange} accept="image/*,video/*" />
                                    <label htmlFor="file-upload" className="px-6 py-2.5 bg-slate-700 hover:bg-slate-600 text-white font-medium rounded-lg cursor-pointer transition-colors">
                                        Browse Files
                                    </label>
                                </div>
                            ) : (
                                <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 flex items-center justify-between">
                                    <div className="flex items-center gap-4 overflow-hidden">
                                        <div className="h-12 w-12 bg-emerald-500/20 text-emerald-400 rounded-lg flex items-center justify-center shrink-0">
                                            <FileType2 className="w-6 h-6" />
                                        </div>
                                        <div className="truncate">
                                            <p className="text-white font-medium truncate">{file.name}</p>
                                            <p className="text-slate-400 text-xs">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                                        </div>
                                    </div>
                                    <button onClick={removeFile} className="p-2 text-slate-400 hover:text-red-400 transition-colors">
                                        <X className="w-5 h-5" />
                                    </button>
                                </div>
                            )}

                            <button
                                onClick={startAnalysis}
                                disabled={!file || isAnalyzing}
                                className={`mt-6 w-full py-4 rounded-xl flex items-center justify-center gap-2 font-bold transition-all shadow-lg ${!file ? 'bg-slate-800 text-slate-500 cursor-not-allowed' : isAnalyzing ? 'bg-emerald-600/50 text-emerald-200 cursor-wait' : 'bg-emerald-600 text-white hover:bg-emerald-500 hover:shadow-emerald-900/50'}`}
                            >
                                {isAnalyzing ? (
                                    <>
                                        <Activity className="w-5 h-5 animate-pulse" /> Analyzing Media...
                                    </>
                                ) : (
                                    'Run Deepfake Scan'
                                )}
                            </button>
                        </div>
                    </div>

                    {/* Right Column: Results Area */}
                    <div className="flex flex-col gap-6">
                        <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl h-full flex flex-col">
                            <div className="p-6 border-b border-slate-800 bg-slate-900/50">
                                <h2 className="text-xl font-semibold text-white flex items-center gap-2">
                                    <ShieldAlert className="w-5 h-5 text-emerald-400" />
                                    Analysis Report
                                </h2>
                            </div>

                            <div className="p-6 flex-1 flex flex-col">
                                {!isAnalyzing && !result ? (
                                    <div className="m-auto flex flex-col items-center justify-center text-center opacity-40">
                                        <ShieldAlert className="w-16 h-16 mb-4 text-slate-500" />
                                        <p className="text-slate-400">Awaiting media upload.<br />Upload a file to view analysis results.</p>
                                    </div>
                                ) : null}

                                {/* Processing State */}
                                {isAnalyzing && (
                                    <div className="m-auto flex flex-col items-center justify-center w-full">
                                        <div className="relative w-24 h-24 mb-6">
                                            <div className="absolute inset-0 border-4 border-slate-800 rounded-full"></div>
                                            <div className="absolute inset-0 border-4 border-emerald-500 rounded-full border-t-transparent animate-spin"></div>
                                            <Activity className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 text-emerald-400 animate-pulse" />
                                        </div>
                                        <h3 className="text-xl text-white font-medium mb-2">Scanning Artifacts...</h3>
                                        <p className="text-slate-400 text-sm">Running visual models across frames.</p>
                                    </div>
                                )}

                                {/* Mock Results Dashboard */}
                                <AnimatePresence>
                                    {result && (
                                        <motion.div
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            className="flex flex-col gap-6 h-full"
                                        >
                                            {/* Top Metric Cards */}
                                            <div className="grid grid-cols-2 gap-4">
                                                <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
                                                    <p className="text-slate-400 text-sm font-medium mb-1">Authenticity Score</p>
                                                    <div className="flex items-end gap-2">
                                                        <span className="text-4xl font-bold text-emerald-400">{result.score}%</span>
                                                    </div>
                                                </div>
                                                <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
                                                    <p className="text-slate-400 text-sm font-medium mb-1">Prediction</p>
                                                    <div className="flex items-center gap-2 h-full">
                                                        <CheckCircle className="w-6 h-6 text-emerald-400" />
                                                        <span className="text-2xl font-bold text-white">{result.prediction}</span>
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Detail Metrics */}
                                            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50 space-y-4">
                                                <div className="flex justify-between items-center border-b border-slate-700 pb-3">
                                                    <span className="text-slate-400">Risk Level</span>
                                                    <span className="px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-xs font-bold border border-emerald-500/20">{result.risk}</span>
                                                </div>
                                                <div className="flex justify-between items-center border-b border-slate-700 pb-3">
                                                    <span className="text-slate-400">Inference Time</span>
                                                    <span className="text-white font-mono">{result.time}</span>
                                                </div>
                                                <div className="flex justify-between items-center">
                                                    <span className="text-slate-400">Model Framework</span>
                                                    <span className="text-white text-sm">{result.model}</span>
                                                </div>
                                            </div>

                                            {/* Mock Image Grad-CAM Preview */}
                                            {file && file.type.startsWith("image/") && (
                                                <div className="mt-auto bg-slate-800 rounded-xl overflow-hidden border border-slate-700">
                                                    <div className="bg-slate-700/50 px-4 py-2 border-b border-slate-700 flex items-center justify-between">
                                                        <span className="text-xs text-slate-300 font-medium tracking-wide">GRAD-CAM HEATMAP (SIMULATED)</span>
                                                    </div>
                                                    <div className="w-full h-40 bg-slate-900 relative">
                                                        {/* We can use an object URL of the uploaded image as the background, overlaid with a tinted gradient to mock a heatmap */}
                                                        <div className="absolute inset-0 bg-cover bg-center opacity-40 mix-blend-luminosity" style={{ backgroundImage: `url(${URL.createObjectURL(file)})` }}></div>
                                                        <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-emerald-500/10 to-red-500/20 mix-blend-overlay"></div>
                                                        <div className="absolute inset-0 flex items-center justify-center">
                                                            <span className="text-slate-400/50 text-sm px-4 text-center">No anomalies detected in spatial frequencies.</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            )}

                                        </motion.div>
                                    )}
                                </AnimatePresence>

                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}
