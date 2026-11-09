'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Image as ImageIcon, Loader2, X, CheckCircle2, AlertTriangle, Sparkles, PanelRightClose } from 'lucide-react';
import { useEditorStore } from '@/lib/store';

export default function AIPromptPanel() {
  const [prompt, setPrompt] = useState('');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number>(0);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const {
    setCsgTree,
    setStlData,
    setCadqueryCode,
    attachedImage,
    attachedImagePreview,
    setAttachedImage,
    isGenerating,
    setIsGenerating,
    generationTime,
    setGenerationTime,
    isPromptCollapsed,
    setIsPromptCollapsed,
  } = useEditorStore();

  // Timer while generating
  useEffect(() => {
    if (isGenerating) {
      setElapsed(0);
      const start = Date.now();
      timerRef.current = setInterval(() => {
        setElapsed(Math.round((Date.now() - start) / 100) / 10);
      }, 100);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isGenerating]);

  const handleImageSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setErrorMsg('Please upload a valid image file (PNG/JPG).');
      return;
    }
    setErrorMsg(null);
    const previewUrl = URL.createObjectURL(file);
    setAttachedImage(file, previewUrl);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageSelect(e.dataTransfer.files[0]);
    }
  };

  const handleRemoveImage = () => {
    if (attachedImagePreview) {
      URL.revokeObjectURL(attachedImagePreview);
    }
    setAttachedImage(null, null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() && !attachedImage) return;

    setErrorMsg(null);
    setIsGenerating(true);
    const startTime = Date.now();

    try {
      let tree: any = null;
      let code: string | null = null;

      if (attachedImage) {
        // Image-to-CAD flow via Enhanced Hybrid endpoint
        const formData = new FormData();
        formData.append('file', attachedImage);
        if (prompt.trim()) {
          formData.append('prompt', prompt.trim());
        }

        const genRes = await fetch('http://localhost:8000/api/generate-from-image', {
          method: 'POST',
          body: formData,
        });

        if (!genRes.ok) {
          let msg = `Image CAD generation failed (${genRes.status})`;
          try {
            const errJson = await genRes.json();
            if (errJson.detail) msg += `: ${errJson.detail}`;
          } catch {
            const errText = await genRes.text();
            if (errText) msg += `: ${errText}`;
          }
          setErrorMsg(msg);
          return;
        }

        const genData = await genRes.json();
        tree = genData.tree;
        code = genData.code;
      } else {
        // Text-to-CAD flow
        const genRes = await fetch('http://localhost:8000/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: prompt.trim() }),
        });

        if (!genRes.ok) {
          let msg = `Text CAD generation failed (${genRes.status})`;
          try {
            const errJson = await genRes.json();
            if (errJson.detail) msg += `: ${errJson.detail}`;
          } catch {
            const errText = await genRes.text();
            if (errText) msg += `: ${errText}`;
          }
          setErrorMsg(msg);
          return;
        }

        const genData = await genRes.json();
        tree = genData.tree;
        code = genData.code;
      }

      const steps = tree?.steps || [];
      setCsgTree(steps);
      if (code) setCadqueryCode(code);

      // Compile CSG to STL
      const compRes = await fetch('http://localhost:8000/api/compile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tree),
      });

      if (!compRes.ok) {
        let msg = `CadQuery kernel compilation failed (${compRes.status})`;
        try {
          const errJson = await compRes.json();
          if (errJson.detail) msg += `: ${errJson.detail}`;
        } catch {
          const errText = await compRes.text();
          if (errText) msg += `: ${errText}`;
        }
        setErrorMsg(msg);
        return;
      }

      const compData = await compRes.json();
      setStlData(compData.data);
      if (compData.code) setCadqueryCode(compData.code);

      const totalSec = Math.round((Date.now() - startTime) / 100) / 10;
      setGenerationTime(totalSec);
    } catch (err: any) {
      console.warn(err);
      setErrorMsg(err.message || 'Generation failed. Please try a different prompt or reference image.');
    } finally {
      setIsGenerating(false);
    }
  };

  const setSamplePrompt = (text: string) => {
    setPrompt(text);
  };

  return (
    <div
      className="glass-panel"
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        position: 'relative',
        border: isDragOver ? '2px dashed var(--accent-primary)' : undefined,
      }}
      onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={handleDrop}
    >
      {/* Header */}
      <div style={{ padding: '16px', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button
            type="button"
            className="button secondary"
            style={{
              padding: '6px',
              borderRadius: '6px',
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--accent-primary)',
              cursor: 'pointer',
              background: 'rgba(59, 130, 246, 0.1)',
              border: '1px solid rgba(59, 130, 246, 0.25)',
            }}
            title="Toggle AI Panel"
            onClick={() => setIsPromptCollapsed(!isPromptCollapsed)}
          >
            <PanelRightClose size={16} />
          </button>
          <div>
            <h2 style={{ fontSize: '1rem', fontWeight: 600 }}>
              Ask AIMDEA
            </h2>
          </div>
        </div>
      </div>

      {/* Content Area */}
      <div style={{ flex: 1, padding: '16px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {/* Error Banner */}
        {errorMsg && (
          <div style={{
            background: 'rgba(239, 68, 68, 0.12)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            color: '#f87171',
            borderRadius: '8px',
            padding: '10px 12px',
            fontSize: '0.8rem',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '8px',
          }}>
            <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: '2px' }} />
            <div style={{ flex: 1 }}>{errorMsg}</div>
            <button onClick={() => setErrorMsg(null)} style={{ background: 'transparent', border: 'none', color: 'inherit', cursor: 'pointer' }}>
              <X size={14} />
            </button>
          </div>
        )}

        {/* Success Status Banner */}
        {generationTime !== null && !isGenerating && !errorMsg && (
          <div style={{
            background: 'rgba(34, 197, 94, 0.12)',
            border: '1px solid rgba(34, 197, 94, 0.3)',
            color: '#4ade80',
            borderRadius: '8px',
            padding: '10px 12px',
            fontSize: '0.8rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}>
            <CheckCircle2 size={16} />
            <span>Valid Solid Compiled in {generationTime}s</span>
          </div>
        )}

        {/* Active Timer Indicator */}
        {isGenerating && (
          <div style={{
            background: 'rgba(59, 130, 246, 0.12)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            color: 'var(--accent-hover)',
            borderRadius: '8px',
            padding: '12px',
            fontSize: '0.85rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 500 }}>
              <Loader2 size={16} className="animate-spin" />
              <span>Generating 3D CAD Geometry ({elapsed.toFixed(1)}s)...</span>
            </div>
            <span style={{ fontSize: '0.75rem', opacity: 0.8 }}>
              {attachedImage ? 'DINOv2 retrieval → Pro spatial reasoning → Flash repair' : 'Synthesizing CSG construction sequence'}
            </span>
          </div>
        )}

        {/* Prompt suggestion shortcuts */}
        <div>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '8px' }}>
            Suggestions
          </span>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {[
              'A flanged cylinder with a central bore hole',
              'A rectangular base plate with four mounting holes',
              'A stepped shaft with chamfered edges',
              'A hollow cylindrical bearing housing',
            ].map((preset, i) => (
              <button
                key={i}
                type="button"
                className="button secondary"
                onClick={() => setSamplePrompt(preset)}
                style={{
                  fontSize: '0.75rem',
                  padding: '6px 10px',
                  justifyContent: 'flex-start',
                  textAlign: 'left',
                  borderRadius: '6px',
                }}
              >
                &ldquo;{preset}&rdquo;
              </button>
            ))}
          </div>
        </div>

        {/* Attached Image Preview */}
        {attachedImagePreview && (
          <div style={{
            position: 'relative',
            borderRadius: '8px',
            overflow: 'hidden',
            border: '1px solid var(--border-color)',
            background: 'rgba(0,0,0,0.2)',
            padding: '8px',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                REFERENCE IMAGE ATTACHED
              </span>
              <button
                type="button"
                onClick={handleRemoveImage}
                style={{
                  background: 'rgba(0,0,0,0.4)',
                  border: 'none',
                  color: 'white',
                  borderRadius: '50%',
                  padding: '4px',
                  cursor: 'pointer',
                  display: 'flex',
                }}
              >
                <X size={14} />
              </button>
            </div>
            <img
              src={attachedImagePreview}
              alt="Reference Preview"
              style={{
                width: '100%',
                maxHeight: '160px',
                objectFit: 'contain',
                borderRadius: '4px',
              }}
            />
          </div>
        )}
      </div>

      {/* Input Form */}
      <div style={{ padding: '16px', borderTop: '1px solid var(--border-color)' }}>
        <form onSubmit={handleGenerate} style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <input
            type="file"
            accept="image/*"
            ref={fileInputRef}
            style={{ display: 'none' }}
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleImageSelect(e.target.files[0]);
              }
            }}
          />

          <textarea
            className="input-field"
            placeholder={attachedImage ? 'Add instructions or guidance for this image (optional)...' : 'Describe the CAD part you want to build...'}
            rows={3}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            style={{ resize: 'none' }}
            disabled={isGenerating}
          />

          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              type="button"
              className="button secondary"
              style={{
                padding: '8px 14px',
                border: attachedImage ? '1px solid var(--accent-primary)' : undefined,
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
              title={attachedImage ? 'Replace Reference Image' : 'Attach Reference Image (or drag and drop)'}
              onClick={() => fileInputRef.current?.click()}
              disabled={isGenerating}
            >
              <ImageIcon size={18} style={{ color: attachedImage ? 'var(--accent-primary)' : undefined }} />
            </button>

            <button
              type="submit"
              className="button"
              style={{ flex: 1 }}
              disabled={isGenerating || (!prompt.trim() && !attachedImage)}
            >
              {isGenerating ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
              {isGenerating ? `Generating (${elapsed.toFixed(1)}s)...` : 'Generate CAD'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
