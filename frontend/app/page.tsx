'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Upload, RefreshCw, Layers, Server, Activity } from 'lucide-react';

interface Detection {
  class_name: string;
  class_id: number;
  confidence: number;
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
}

interface ApiResponse {
  filename: string;
  width: number;
  height: number;
  detections_count: number;
  detections: Detection[];
}

const getClassColor = (className: string) => {
  switch (className.toLowerCase()) {
    case 'table': return '#3b82f6'; 
    case 'table column header': return '#10b981'; 
    case 'table projected row header': return '#f59e0b'; 
    default: return '#8b5cf6'; 
  }
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!results || !previewUrl) return;

    const img = new Image();
    img.src = previewUrl;
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const displayWidth = canvas.parentElement?.clientWidth || img.width;
      const scaleFactor = displayWidth / results.width;
      const displayHeight = results.height * scaleFactor;

      canvas.width = displayWidth;
      canvas.height = displayHeight;

      ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

      results.detections.forEach((det) => {
        const color = getClassColor(det.class_name);
        const { x1, y1, x2, y2 } = det.bbox;

        const rx1 = x1 * scaleFactor;
        const ry1 = y1 * scaleFactor;
        const rx2 = x2 * scaleFactor;
        const ry2 = y2 * scaleFactor;
        const rw = rx2 - rx1;
        const rh = ry2 - ry1;

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(rx1, ry1, rw, rh);

        ctx.fillStyle = color;
        ctx.font = 'bold 11px sans-serif';
        const labelText = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
        const textWidth = ctx.measureText(labelText).width;
        
        ctx.fillRect(rx1, ry1 - 18 > 0 ? ry1 - 18 : ry1, textWidth + 10, 18);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(labelText, rx1 + 5, (ry1 - 18 > 0 ? ry1 - 5 : ry1 + 13));
      });
    };
  }, [results, previewUrl]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setResults(null);
      setError(null);
    }
  };

  const handleUploadSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed processing document image data.');
      }

      const data: ApiResponse = await response.json();
      setResults(data);
    } catch (err: any) {
      setError(err.message || 'Network connection interface failure.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-900 text-slate-100 p-8">
      <header className="max-w-7xl mx-auto mb-8 pb-4 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Layers className="text-blue-500 w-8 h-8" />
          <div>
            <h1 className="text-2xl font-bold tracking-tight">YOLOv26 Document Table Layout Analyzer</h1>
            <p className="text-xs text-slate-400">Deep-learning structural segmentation parser engine</p>
          </div>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono bg-slate-950 px-4 py-2 rounded-lg border border-slate-800">
          <div className="flex items-center gap-2">
            <Server className="text-emerald-400 w-4 h-4" />
            <span>FastAPI: Online</span>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-slate-950 p-6 rounded-xl border border-slate-800 shadow-xl">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Upload className="w-5 h-5 text-blue-500" /> Source Document Ingestion
            </h2>
            
            <label className="border-2 border-dashed border-slate-700 hover:border-blue-500 transition-colors rounded-xl p-8 flex flex-col items-center justify-center gap-3 cursor-pointer bg-slate-900/50">
              <input type="file" accept=".png,.jpg,.jpeg" className="hidden" onChange={handleFileChange} />
              <Upload className="w-10 h-10 text-slate-500" />
              <div className="text-center">
                <span className="text-sm font-medium text-slate-300 block">Click to browse filesystem</span>
              </div>
            </label>

            {file && (
              <div className="mt-4 p-3 bg-slate-900 rounded-lg border border-slate-800 flex items-center justify-between text-xs">
                <span className="font-mono text-slate-300 truncate max-w-[250px]">{file.name}</span>
                <span className="text-slate-500">{(file.size / 1024).toFixed(1)} KB</span>
              </div>
            )}

            <button
              onClick={handleUploadSubmit}
              disabled={!file || loading}
              className="w-full mt-6 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 text-white font-medium py-3 px-4 rounded-xl transition-all shadow-lg flex items-center justify-center gap-2"
            >
              {loading ? <RefreshCw className="w-5 h-5 animate-spin" /> : 'Run Layout Analysis'}
            </button>
          </div>

          {error && (
            <div className="p-4 bg-red-950/40 border border-red-900 text-red-200 text-sm rounded-xl">
              <p className="font-semibold">Execution Blocked</p>
              <p className="text-xs mt-1 text-red-400">{error}</p>
            </div>
          )}

          {results && (
            <div className="bg-slate-950 p-6 rounded-xl border border-slate-800 shadow-xl space-y-4">
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                <Activity className="w-4 h-4 text-emerald-400" /> Pipeline Analytical Data
              </h3>
              <div className="grid grid-cols-2 gap-4 text-center">
                <div className="bg-slate-900 p-3 rounded-lg border border-slate-800">
                  <span className="text-xs text-slate-500 block">Targets Located</span>
                  <span className="text-2xl font-mono font-bold text-blue-400">{results.detections_count}</span>
                </div>
                <div className="bg-slate-900 p-3 rounded-lg border border-slate-800">
                  <span className="text-xs text-slate-500 block">Image Geometry</span>
                  <span className="text-sm font-mono font-semibold block mt-1">{results.width} x {results.height} px</span>
                </div>
              </div>

              <div className="max-h-[220px] overflow-y-auto space-y-2 pr-1">
                {results.detections.map((det, index) => {
                  const labelColor = getClassColor(det.class_name);
                  return (
                    <div key={index} className="flex items-center justify-between p-3 bg-slate-900 rounded-lg border border-slate-800/60 text-xs">
                      <div className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: labelColor }} />
                        <span className="font-semibold text-slate-200 capitalize">{det.class_name}</span>
                      </div>
                      <div className="flex items-center gap-3 font-mono">
                        <span className="text-slate-400">Conf: <b className="text-slate-200">{(det.confidence * 100).toFixed(1)}%</b></span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        <div className="lg:col-span-7">
          <div className="bg-slate-950 p-6 rounded-xl border border-slate-800 shadow-xl flex flex-col min-h-[500px]">
            <h2 className="text-lg font-semibold mb-4">Inference Visualizer Output Canvas</h2>
            <div className="flex-1 bg-slate-900 rounded-xl border border-slate-800 flex items-center justify-center p-4 relative overflow-hidden">
              {previewUrl ? (
                <div className="relative w-full shadow-2xl rounded-lg overflow-hidden border border-slate-800">
                  <canvas ref={canvasRef} className="block max-w-full mx-auto" />
                </div>
              ) : (
                <div className="text-center text-slate-600 max-w-sm">
                  <Layers className="w-12 h-12 mx-auto mb-3 opacity-20" />
                  <p className="text-sm">Ingest a page layout graphic from the control card deck panel to trace anchor coordinate arrays.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}