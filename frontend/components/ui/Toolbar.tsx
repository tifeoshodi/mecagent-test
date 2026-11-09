'use client';

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import {
  Download,
  Share2,
  Undo2,
  Redo2,
  Sun,
  Moon,
  Sunset,
  FileCode,
  FileBox,
  Copy,
  Check,
  BarChart3,
  Home,
  ChevronDown
} from 'lucide-react';
import { useEditorStore } from '@/lib/store';

export default function Toolbar() {
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [copied, setCopied] = useState(false);
  const [isExportingStep, setIsExportingStep] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const {
    theme,
    setTheme,
    isPromptCollapsed,
    setIsPromptCollapsed,
    stlData,
    csgTree,
    cadqueryCode,
  } = useEditorStore();

  const cycleTheme = () => {
    if (theme === 'light') setTheme('dim');
    else if (theme === 'dim') setTheme('dark');
    else setTheme('light');
  };

  const ThemeIcon = theme === 'light' ? Sun : theme === 'dim' ? Sunset : Moon;

  // Close export menu on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setShowExportMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleDownloadStl = () => {
    if (!stlData) {
      alert('No 3D solid generated yet to export.');
      return;
    }
    const byteCharacters = atob(stlData);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'model/stl' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cad_model_${Date.now()}.stl`;
    a.click();
    URL.revokeObjectURL(url);
    setShowExportMenu(false);
  };

  const handleDownloadStep = async () => {
    if (!cadqueryCode && csgTree.length === 0) {
      alert('No 3D solid generated yet to export.');
      return;
    }
    setIsExportingStep(true);
    try {
      const res = await fetch('http://localhost:8000/api/export/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code: cadqueryCode,
          tree: { steps: csgTree },
        }),
      });

      if (!res.ok) {
        throw new Error('STEP export failed');
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `cad_model_${Date.now()}.step`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e: any) {
      alert(`Export STEP failed: ${e.message}`);
    } finally {
      setIsExportingStep(false);
      setShowExportMenu(false);
    }
  };

  const handleCopyCode = async () => {
    if (!cadqueryCode) {
      alert('No CadQuery code available. Generate a model first.');
      return;
    }
    try {
      await navigator.clipboard.writeText(cadqueryCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      setShowExportMenu(false);
    } catch (e) {
      console.error('Failed to copy', e);
    }
  };

  return (
    <div
      className="glass-panel"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '6px 12px',
        borderRadius: '30px',
        boxShadow: 'var(--shadow-md)',
      }}
    >
      {/* Navigation Links */}
      <Link
        href="/"
        className="button secondary"
        style={{ padding: '8px', borderRadius: '50%' }}
        title="Home"
      >
        <Home size={16} />
      </Link>

      <div style={{ width: '1px', height: '18px', background: 'var(--border-color)', margin: '0 2px' }} />

      <button className="button secondary" style={{ padding: '8px', borderRadius: '50%' }} title="Undo">
        <Undo2 size={16} />
      </button>
      <button className="button secondary" style={{ padding: '8px', borderRadius: '50%' }} title="Redo">
        <Redo2 size={16} />
      </button>

      <div style={{ width: '1px', height: '18px', background: 'var(--border-color)', margin: '0 2px' }} />


      <button
        className="button secondary"
        style={{ padding: '8px', borderRadius: '50%' }}
        title="Toggle Theme"
        onClick={cycleTheme}
      >
        <ThemeIcon size={16} />
      </button>

      {/* Export Dropdown */}
      <div style={{ position: 'relative' }} ref={menuRef}>
        <button
          className="button"
          style={{ padding: '7px 16px', borderRadius: '20px', fontSize: '0.8rem', gap: '6px' }}
          onClick={() => setShowExportMenu(!showExportMenu)}
        >
          <Download size={15} />
          <span>Export</span>
          <ChevronDown size={13} />
        </button>

        {showExportMenu && (
          <div
            className="glass-panel"
            style={{
              position: 'absolute',
              top: 'calc(100% + 8px)',
              right: 0,
              minWidth: '200px',
              padding: '6px',
              display: 'flex',
              flexDirection: 'column',
              gap: '4px',
              zIndex: 100,
              boxShadow: 'var(--shadow-lg)',
            }}
          >
            <button
              onClick={handleDownloadStl}
              className="button secondary"
              style={{
                justifyContent: 'flex-start',
                width: '100%',
                padding: '8px 12px',
                fontSize: '0.8rem',
                border: 'none',
                borderRadius: '6px',
              }}
            >
              <span>Download STL (.stl)</span>
            </button>

            <button
              onClick={handleDownloadStep}
              disabled={isExportingStep}
              className="button secondary"
              style={{
                justifyContent: 'flex-start',
                width: '100%',
                padding: '8px 12px',
                fontSize: '0.8rem',
                border: 'none',
                borderRadius: '6px',
              }}
            >
              <span>{isExportingStep ? 'Exporting STEP...' : 'Download STEP (.step)'}</span>
            </button>

            <div style={{ height: '1px', background: 'var(--border-color)', margin: '4px 0' }} />

            <button
              onClick={handleCopyCode}
              className="button secondary"
              style={{
                justifyContent: 'flex-start',
                width: '100%',
                padding: '8px 12px',
                fontSize: '0.8rem',
                border: 'none',
                borderRadius: '6px',
              }}
            >
              {copied ? <Check size={16} style={{ color: '#10b981' }} /> : <Copy size={16} />}
              <span>{copied ? 'Code Copied!' : 'Copy CadQuery Code'}</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
