'use client';

import CADViewport from '@/components/viewport/CADViewport';
import AIPromptPanel from '@/components/panels/AIPromptPanel';
import CSGTreePanel from '@/components/panels/CSGTreePanel';
import Toolbar from '@/components/ui/Toolbar';
import { PanelRightOpen } from 'lucide-react';
import { useEditorStore } from '@/lib/store';

export default function EditorPage() {
  const isPromptCollapsed = useEditorStore(state => state.isPromptCollapsed);
  const setIsPromptCollapsed = useEditorStore(state => state.setIsPromptCollapsed);
  
  return (
    <div className="app-container">
      {/* Top Toolbar */}
      <div className="toolbar-top">
        <Toolbar />
      </div>

      {/* Center Viewport: React Three Fiber Canvas */}
      <div className="viewport-container">
        <CADViewport />
      </div>

      {/* Floating Re-Open Button when AI Panel is Collapsed */}
      {isPromptCollapsed && (
        <button
          className="button secondary glass-panel animate-fade-in"
          style={{
            position: 'absolute',
            top: '16px',
            right: '16px',
            zIndex: 20,
            padding: '8px 14px',
            borderRadius: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            background: 'var(--bg-panel)',
            border: '1px solid var(--border-color)',
            boxShadow: 'var(--shadow-md)',
            color: 'var(--text-primary)',
            cursor: 'pointer',
          }}
          title="Open Ask AIMDEA Panel"
          onClick={() => setIsPromptCollapsed(false)}
        >
          <PanelRightOpen size={16} style={{ color: 'var(--accent-primary)' }} />
          <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>Ask AIMDEA</span>
        </button>
      )}

      {/* Right Sidebar: AI Generation & Tools */}
      <div className="panel-right animate-fade-in" style={{ animationDelay: '0.1s', right: isPromptCollapsed ? '-340px' : '16px' }}>
        <AIPromptPanel />
      </div>

      {/* Bottom Panel: CSG Tree Timeline */}
      <div className="panel-bottom animate-fade-in" style={{ animationDelay: '0.2s', right: isPromptCollapsed ? '16px' : '352px' }}>
        <CSGTreePanel />
      </div>
    </div>
  );
}
