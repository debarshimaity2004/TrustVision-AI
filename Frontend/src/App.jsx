import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import Home from "./pages/Home";
import Analyze from "./pages/Analyze";

function App() {
  return (
    <BrowserRouter>
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: 'linear-gradient(135deg, #1a1a2e, #16213e)',
            color: '#f8fafc',
            border: '1px solid #7c3aed',
            borderRadius: '10px',
            boxShadow: '0 0 30px rgba(168, 85, 247, 0.2)',
            backdropFilter: 'blur(10px)',
          },
          success: {
            style: {
              borderColor: '#6366f1',
              boxShadow: '0 0 30px rgba(99, 102, 241, 0.3)',
            },
            iconTheme: {
              primary: '#6366f1',
              secondary: '#1a1a2e',
            },
          },
          error: {
            style: {
              borderColor: '#ef4444',
              boxShadow: '0 0 30px rgba(239, 68, 68, 0.3)',
            },
            iconTheme: {
              primary: '#ef4444',
              secondary: '#1a1a2e',
            },
          },
        }}
      />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/analyze" element={<Analyze />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;