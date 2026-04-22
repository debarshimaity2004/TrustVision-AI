import { useEffect, useRef, useState } from 'react';
import Navbar from '../components/Navbar';
import {
    Activity,
    Camera,
    CheckCircle,
    FileDown,
    Upload,
    AlertCircle,
    X,
    ShieldCheck,
    Video,
    Image as ImageIcon
} from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { useSearchParams } from 'react-router-dom';

const API_BASE_URL = 'http://localhost:8000';
const API_URL = `${API_BASE_URL}/scan`;
const LIVE_SCAN_INTERVAL_MS = 2500;

function formatScanResult(data, modelName) {
    return {
        scanId: data.id,
        filename: data.filename,
        score: data.authenticity_score !== undefined ? parseFloat(data.authenticity_score).toFixed(2) : 0,
        prediction: data.prediction,
        risk: data.risk_level,
        confidence: data.confidence !== undefined ? parseFloat(data.confidence).toFixed(2) : 0,
        heatmap_base64: data.heatmap_base64,
        model: data.model_version || modelName,
        timestamp: data.timestamp,
        reportUrl: data.report_url ? `${API_BASE_URL}${data.report_url}` : null,
    };
}

function resultAccent(prediction) {
    return prediction === 'FAKE'
        ? {
            badge: 'bg-rose-100 text-rose-700 border-rose-200',
            icon: 'text-rose-500',
            text: 'text-rose-600',
            meter: 'from-rose-500 via-red-400 to-orange-400',
            bg: 'bg-rose-50/50',
            ring: 'ring-rose-200/50'
        }
        : {
            badge: 'bg-emerald-100 text-emerald-700 border-emerald-200',
            icon: 'text-emerald-500',
            text: 'text-emerald-600',
            meter: 'from-emerald-400 via-emerald-500 to-teal-400',
            bg: 'bg-emerald-50/50',
            ring: 'ring-emerald-200/50'
        };
}

export default function Analyze() {
    const [searchParams] = useSearchParams();
    const [mode, setMode] = useState(() => (searchParams.get('mode') === 'live' ? 'live' : 'upload'));
    const [file, setFile] = useState(null);
    const [isDragActive, setIsDragActive] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [uploadResult, setUploadResult] = useState(null);
    const [liveResult, setLiveResult] = useState(null);
    const [isCameraReady, setIsCameraReady] = useState(false);
    const [isLiveActive, setIsLiveActive] = useState(false);
    const [isLiveProcessing, setIsLiveProcessing] = useState(false);
    const [liveStatus, setLiveStatus] = useState('Camera offline');

    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const streamRef = useRef(null);
    const liveIntervalRef = useRef(null);
    const liveRequestInFlightRef = useRef(false);

    const activeResult = mode === 'live' ? liveResult : uploadResult;
    const accent = resultAccent(activeResult?.prediction);

    useEffect(() => {
        return () => {
            stopLiveDetection();
        };
    }, []);

    useEffect(() => {
        if (mode !== 'live') {
            stopLiveDetection();
        }
    }, [mode]);

    useEffect(() => {
        setMode(searchParams.get('mode') === 'live' ? 'live' : 'upload');
    }, [searchParams]);

    const scanMedia = async (blob, filename, modelName) => {
        const formData = new FormData();
        formData.append('file', blob, filename);

        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || 'API scan failed');
        }

        return formatScanResult(data, modelName);
    };

    const stopStreamTracks = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
        }
    };

    const clearLiveInterval = () => {
        if (liveIntervalRef.current) {
            window.clearInterval(liveIntervalRef.current);
            liveIntervalRef.current = null;
        }
    };

    const stopLiveDetection = () => {
        clearLiveInterval();
        stopStreamTracks();
        liveRequestInFlightRef.current = false;
        setIsLiveActive(false);
        setIsLiveProcessing(false);
        setIsCameraReady(false);
        setLiveStatus('Camera offline');

        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
    };

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setIsDragActive(true);
        } else if (e.type === 'dragleave') {
            setIsDragActive(false);
        }
    };

    const assignUploadedFile = (uploadedFile) => {
        if (uploadedFile.size > 50 * 1024 * 1024) {
            toast.error('File size exceeds 50MB limit');
            return;
        }

        setFile(uploadedFile);
        setUploadResult(null);
        toast.success('File attached');
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            assignUploadedFile(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            assignUploadedFile(e.target.files[0]);
        }
    };

    const removeFile = () => {
        setFile(null);
        setUploadResult(null);
    };

    const startAnalysis = async () => {
        if (!file) {
            toast.error('Please attach a media file first');
            return;
        }

        setIsAnalyzing(true);
        setUploadResult(null);
        toast.loading('Analyzing metadata and pixels...', { id: 'analysis' });

        try {
            const result = await scanMedia(file, file.name, 'TrustVision Upload Scan');
            setUploadResult(result);
            toast.success('Analysis complete!', { id: 'analysis' });
        } catch (error) {
            console.error('Analysis Error:', error);
            toast.error(`Error: ${error.message}`, { id: 'analysis' });
        } finally {
            setIsAnalyzing(false);
        }
    };

    const downloadPdfReport = async () => {
        if (!activeResult?.reportUrl) {
            toast.error('No report available.');
            return;
        }

        try {
            toast.loading('Compiling report...', { id: 'pdf' });
            const response = await fetch(activeResult.reportUrl);
            if (!response.ok) throw new Error('PDF Generation Failed');

            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = `TrustVision_Report_${activeResult.scanId || 'scan'}.pdf`;
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(downloadUrl);
            toast.success('Report downloaded', { id: 'pdf' });
        } catch (error) {
            console.error('PDF error:', error);
            toast.error(`Report error: ${error.message}`, { id: 'pdf' });
        }
    };

    const analyzeCurrentFrame = async () => {
        if (
            !videoRef.current ||
            !canvasRef.current ||
            videoRef.current.videoWidth === 0 ||
            liveRequestInFlightRef.current
        ) {
            return;
        }

        const canvas = canvasRef.current;
        const video = videoRef.current;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        setIsLiveProcessing(true);
        setLiveStatus('Analyzing frame...');
        liveRequestInFlightRef.current = true;

        try {
            const blob = await new Promise((resolve) => {
                canvas.toBlob(resolve, 'image/jpeg', 0.9);
            });

            if (!blob) throw new Error('Capture failed');

            const result = await scanMedia(blob, 'webcam-frame.jpg', 'TrustVision Live Array');
            setLiveResult(result);
            setLiveStatus(result.prediction === 'FAKE' ? 'Manipulated frame detected' : 'Authentic frame verified');
        } catch (error) {
            console.error('Live scan failed:', error);
            setLiveStatus('Analysis failed');
        } finally {
            liveRequestInFlightRef.current = false;
            setIsLiveProcessing(false);
        }
    };

    const startLiveDetection = async () => {
        if (!navigator.mediaDevices?.getUserMedia) {
            toast.error('Webcam not supported');
            return;
        }

        try {
            stopLiveDetection();
            setLiveResult(null);
            setLiveStatus('Initializing camera...');

            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                },
                audio: false,
            });

            streamRef.current = stream;
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
            }

            setIsCameraReady(true);
            setIsLiveActive(true);
            setLiveStatus('Scanning live stream (2.5s intervals)');
            await analyzeCurrentFrame();

            liveIntervalRef.current = window.setInterval(() => {
                analyzeCurrentFrame();
            }, LIVE_SCAN_INTERVAL_MS);
        } catch (error) {
            console.error('Webcam access failed:', error);
            stopLiveDetection();
            toast.error(`Unable to start webcam: ${error.message}`);
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 font-sans text-slate-800 pb-20 pt-32 selection:bg-emerald-200">
            <Navbar />

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                {/* Minimal Header */}
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col items-center text-center mb-10"
                >
                    <div className="w-16 h-16 rounded-3xl bg-white shadow-sm border border-slate-100 flex items-center justify-center mb-6">
                        <ShieldCheck className="w-8 h-8 text-sky-500" />
                    </div>
                    <h1 className="text-4xl md:text-5xl font-black text-slate-900 mb-4 tracking-tight">Trust Analyzer</h1>
                    <p className="text-slate-500 font-medium max-w-xl mx-auto">
                        Securely verify media authenticity using our cryptographically-backed neural verification arrays.
                    </p>
                </motion.div>

                {/* Mode Selector */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="mb-8 flex justify-center"
                >
                    <div className="inline-flex rounded-full bg-white border border-slate-200 p-1.5 shadow-[0_4px_16px_rgba(0,0,0,0.04)]">
                        {['upload', 'live'].map((m) => (
                            <button
                                key={m}
                                onClick={() => setMode(m)}
                                className={`px-6 py-2.5 rounded-full text-sm font-bold transition-all flex items-center gap-2 ${
                                    mode === m
                                        ? 'bg-slate-900 text-white shadow-md'
                                        : 'text-slate-500 hover:text-slate-900 hover:bg-slate-50'
                                }`}
                            >
                                {m === 'upload' ? <ImageIcon className="w-4 h-4" /> : <Video className="w-4 h-4" />}
                                {m === 'upload' ? 'File Analysis' : 'Live Stream'}
                            </button>
                        ))}
                    </div>
                </motion.div>

                <div className="grid grid-cols-1 lg:grid-cols-[1.5fr_1fr] gap-8 mt-4">
                    {/* Input/Analysis Section */}
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.1 }}
                        className="bg-white rounded-[2rem] border border-slate-100 p-8 shadow-[0_10px_40px_-15px_rgba(0,0,0,0.05)] h-full"
                    >
                        {mode === 'upload' ? (
                            <div className="flex flex-col h-full">
                                <h2 className="text-xl font-bold text-slate-900 mb-6 flex items-center gap-2">
                                    <Upload className="w-5 h-5 text-slate-400" /> Upload Source Media
                                </h2>

                                {!file ? (
                                    <motion.div
                                        whileHover={{ borderColor: '#0ea5e9' }}
                                        className={`flex-1 border-2 border-dashed rounded-3xl p-12 flex flex-col items-center justify-center transition-all cursor-pointer min-h-[300px] ${
                                            isDragActive
                                                ? 'border-sky-500 bg-sky-50'
                                                : 'border-slate-200 bg-slate-50/50 hover:bg-slate-50'
                                        }`}
                                        onDragEnter={handleDrag}
                                        onDragLeave={handleDrag}
                                        onDragOver={handleDrag}
                                        onDrop={handleDrop}
                                    >
                                        <div className="w-16 h-16 bg-white shadow-sm rounded-full flex items-center justify-center mb-4">
                                            <Upload className="w-8 h-8 text-sky-500" />
                                        </div>
                                        <p className="text-slate-800 font-bold mb-1">Drag files here to upload</p>
                                        <p className="text-slate-400 text-sm font-medium mb-8">PNG, JPG up to 50MB</p>

                                        <input type="file" id="file-upload" className="hidden" onChange={handleChange} accept="image/*" />
                                        <label
                                            htmlFor="file-upload"
                                            className="px-6 py-2.5 bg-white border border-slate-200 shadow-sm text-slate-700 font-bold rounded-xl cursor-pointer hover:bg-slate-50 transition-all"
                                        >
                                            Select from computer
                                        </label>
                                    </motion.div>
                                ) : (
                                    <motion.div
                                        initial={{ opacity: 0, scale: 0.95 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        className="bg-slate-50 border border-slate-200 rounded-2xl p-6 flex items-center justify-between mb-auto"
                                    >
                                        <div className="flex items-center gap-4 truncate">
                                            <div className="w-12 h-12 bg-white rounded-xl shadow-sm border border-slate-100 flex items-center justify-center shrink-0">
                                                <ImageIcon className="w-6 h-6 text-emerald-500" />
                                            </div>
                                            <div className="truncate">
                                                <p className="text-slate-800 font-bold truncate text-sm">{file.name}</p>
                                                <p className="text-slate-500 text-xs font-medium">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                                            </div>
                                        </div>
                                        <button
                                            onClick={removeFile}
                                            className="w-8 h-8 flex items-center justify-center rounded-full bg-slate-200/50 text-slate-500 hover:text-slate-800 hover:bg-slate-200 transition-colors"
                                        >
                                            <X className="w-4 h-4" />
                                        </button>
                                    </motion.div>
                                )}

                                <motion.button
                                    whileHover={file && !isAnalyzing ? { scale: 1.01 } : {}}
                                    whileTap={file && !isAnalyzing ? { scale: 0.98 } : {}}
                                    onClick={startAnalysis}
                                    disabled={!file || isAnalyzing}
                                    className={`w-full py-4 mt-6 rounded-2xl font-bold transition-all flex items-center justify-center gap-2 ${
                                        !file
                                            ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                                            : isAnalyzing
                                            ? 'bg-gradient-to-r from-sky-500 to-emerald-500 text-white cursor-wait opacity-90'
                                            : 'bg-slate-900 text-white shadow-lg hover:shadow-slate-300'
                                    }`}
                                >
                                    {isAnalyzing ? (
                                        <><Activity className="w-5 h-5 animate-spin" /> Verifying Signatures...</>
                                    ) : (
                                        <><Activity className="w-5 h-5" /> Execute Verification</>
                                    )}
                                </motion.button>
                            </div>
                        ) : (
                            <div className="flex flex-col h-full">
                                <h2 className="text-xl font-bold text-slate-900 mb-6 flex items-center gap-2">
                                    <Video className="w-5 h-5 text-slate-400" /> Live Stream Analysis
                                </h2>
                                
                                <div className="relative rounded-3xl overflow-hidden bg-slate-900 mb-6 aspect-video shadow-inner">
                                    <video
                                        ref={videoRef}
                                        className="w-full h-full object-cover"
                                        muted
                                        playsInline
                                        autoPlay
                                    />
                                    
                                    {!isCameraReady && (
                                        <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-100">
                                            <div className="w-16 h-16 bg-white rounded-full shadow-sm flex items-center justify-center mb-4">
                                                <Camera className="w-8 h-8 text-slate-400" />
                                            </div>
                                            <p className="text-slate-600 font-bold">Camera Disconnected</p>
                                        </div>
                                    )}
                                    
                                    <canvas ref={canvasRef} className="hidden" />
                                    
                                    {isLiveActive && (
                                        <div className="absolute inset-0 border-[3px] border-emerald-400/50 rounded-3xl pointer-events-none">
                                            {isLiveProcessing && (
                                                <div className="absolute w-full h-[2px] bg-emerald-400/80 shadow-[0_0_15px_rgba(52,211,153,1)] animate-[scan-line_2.5s_linear_infinite]" />
                                            )}
                                        </div>
                                    )}
                                    
                                    {isLiveActive && (
                                        <div className="absolute top-4 left-4 bg-white/90 backdrop-blur-md px-3 py-1.5 rounded-full flex items-center gap-2 shadow-sm font-semibold text-xs text-slate-700">
                                            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                                            Live
                                        </div>
                                    )}
                                </div>

                                <div className="bg-slate-50 border border-slate-200 rounded-2xl p-4 flex justify-between items-center mb-6">
                                    <div>
                                        <p className="text-xs text-slate-500 font-bold uppercase tracking-wide">Stream Status</p>
                                        <p className="font-semibold text-slate-900">{liveStatus}</p>
                                    </div>
                                    {isLiveProcessing && <Activity className="w-5 h-5 text-emerald-500 animate-[spin_2s_linear_infinite]" />}
                                </div>

                                <div className="flex gap-4 mt-auto">
                                    <motion.button
                                        whileHover={!isLiveActive ? { scale: 1.02 } : {}}
                                        whileTap={!isLiveActive ? { scale: 0.98 } : {}}
                                        onClick={startLiveDetection}
                                        disabled={isLiveActive}
                                        className={`flex-1 py-4 rounded-2xl font-bold transition-all ${
                                            isLiveActive
                                                ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                                                : 'bg-slate-900 text-white shadow-md'
                                        }`}
                                    >
                                        Initiate Feed
                                    </motion.button>
                                    <motion.button
                                        whileHover={isLiveActive || isCameraReady ? { scale: 1.02 } : {}}
                                        whileTap={isLiveActive || isCameraReady ? { scale: 0.98 } : {}}
                                        onClick={stopLiveDetection}
                                        disabled={!isLiveActive && !isCameraReady}
                                        className={`flex-1 py-4 rounded-2xl font-bold transition-all border ${
                                            !isLiveActive && !isCameraReady
                                                ? 'bg-slate-50 border-slate-200 text-slate-400 cursor-not-allowed'
                                                : 'bg-white border-slate-200 text-slate-700 shadow-sm hover:bg-slate-50'
                                        }`}
                                    >
                                        Terminate Signal
                                    </motion.button>
                                </div>
                            </div>
                        )}
                    </motion.div>

                    {/* Report Output Panel */}
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.2 }}
                        className={`bg-white rounded-[2rem] border p-8 shadow-[0_10px_40px_-15px_rgba(0,0,0,0.05)] transition-all flex flex-col ${
                            activeResult ? `border-px ${accent.ring} shadow-lg ring-4` : 'border-slate-100'
                        }`}
                    >
                        <h2 className="text-xl font-bold text-slate-900 mb-6 flex items-center gap-2">
                            <ShieldCheck className="w-5 h-5 text-slate-400" />
                            Diagnostic Output
                        </h2>

                        <div className="flex-1">
                            <AnimatePresence mode="wait">
                                {!isAnalyzing && !isLiveProcessing && !activeResult ? (
                                    <motion.div
                                        key="empty"
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0 }}
                                        className="h-full flex flex-col items-center justify-center text-center text-slate-500 min-h-[400px]"
                                    >
                                        <div className="w-16 h-16 rounded-full border-2 border-slate-100 bg-slate-50 flex items-center justify-center mb-4">
                                            <Activity className="w-6 h-6 text-slate-300" />
                                        </div>
                                        <p className="font-semibold text-slate-600">Awaiting Data Feed</p>
                                        <p className="text-sm">Initiate an upload or live stream to begin.</p>
                                    </motion.div>
                                ) : (isAnalyzing || isLiveProcessing) && !activeResult ? (
                                    <motion.div
                                        key="loading"
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0 }}
                                        className="h-full flex flex-col items-center justify-center text-center min-h-[400px]"
                                    >
                                        <motion.div
                                            animate={{ rotate: 360 }}
                                            transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
                                            className="w-12 h-12 rounded-full border-[3px] border-slate-100 border-t-sky-500 mb-6"
                                        />
                                        <p className="font-bold text-slate-900 mb-1">Verifying Data</p>
                                        <p className="text-sm text-slate-500">Cross-referencing neural patterns...</p>
                                    </motion.div>
                                ) : activeResult ? (
                                    <motion.div
                                        key="result"
                                        initial={{ opacity: 0, y: 10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="space-y-6"
                                    >
                                        {/* Main Result Hero */}
                                        <div className={`p-6 rounded-3xl ${accent.bg} border ${accent.badge} flex flex-col items-center justify-center text-center`}>
                                            <p className="text-xs font-bold uppercase tracking-widest mb-1 opacity-70">Authenticity Rating</p>
                                            <div className="flex items-end gap-1 mb-2">
                                                <span className={`text-6xl font-black tracking-tighter ${accent.text}`}>{activeResult.score}</span>
                                                <span className={`text-xl font-bold pb-2 ${accent.text}`}>%</span>
                                            </div>
                                            <div className={`px-4 py-1.5 rounded-full bg-white/50 border font-bold text-sm inline-flex items-center gap-2 ${accent.badge}`}>
                                                <CheckCircle className="w-4 h-4" /> {activeResult.prediction} DETECTED
                                            </div>
                                        </div>

                                        {/* Stat Grid */}
                                        <div className="grid grid-cols-2 gap-4">
                                            <div className="bg-slate-50 border border-slate-100 p-4 rounded-2xl">
                                                <p className="text-xs text-slate-500 font-bold uppercase mb-1">Confidence</p>
                                                <p className="text-xl font-black text-slate-900">{activeResult.confidence}%</p>
                                            </div>
                                            <div className="bg-slate-50 border border-slate-100 p-4 rounded-2xl">
                                                <p className="text-xs text-slate-500 font-bold uppercase mb-1">Risk Index</p>
                                                <p className="text-xl font-black text-slate-900 capitalize">{activeResult.risk}</p>
                                            </div>
                                        </div>

                                        {/* Advanced Heatmap */}
                                        {activeResult.heatmap_base64 && (
                                            <div className="rounded-2xl border border-slate-200 overflow-hidden bg-white shadow-sm">
                                                <div className="bg-slate-50 px-4 py-2 border-b border-slate-200 flex justify-between items-center">
                                                    <span className="text-xs font-bold text-slate-500">RESNET-VIT HEATMAP</span>
                                                    <span className="w-2 h-2 rounded-full bg-sky-500" />
                                                </div>
                                                <img src={activeResult.heatmap_base64} alt="Heatmap" className="w-full h-auto" />
                                            </div>
                                        )}

                                        {/* Download Report */}
                                        {activeResult.reportUrl && (
                                            <motion.button
                                                whileHover={{ scale: 1.02 }}
                                                whileTap={{ scale: 0.98 }}
                                                onClick={downloadPdfReport}
                                                className="w-full py-4 rounded-2xl border-2 border-slate-900 text-slate-900 font-bold hover:bg-slate-900 hover:text-white transition-colors flex items-center justify-center gap-2"
                                            >
                                                <FileDown className="w-5 h-5" /> Download Full Diagnostic PDF
                                            </motion.button>
                                        )}
                                    </motion.div>
                                ) : null}
                            </AnimatePresence>
                        </div>
                    </motion.div>
                </div>
            </main>
        </div>
    );
}
