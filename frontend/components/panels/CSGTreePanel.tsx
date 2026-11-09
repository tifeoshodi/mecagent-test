'use client';

import { Box, Circle, Plus, Settings2, Hammer, Droplet, ArrowRightLeft } from 'lucide-react';
import { useEditorStore } from '@/lib/store';

export default function CSGTreePanel() {
  const { csgTree } = useEditorStore();

  const getIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'box': return Box;
      case 'cylinder': return Circle;
      case 'sphere': return Circle;
      case 'fillet': return Settings2;
      case 'extrude': return Hammer;
      case 'hole': return Droplet;
      case 'mirror': return ArrowRightLeft;
      default: return Box;
    }
  };

  return (
    <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ padding: '16px', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ fontSize: '1rem', fontWeight: 600 }}>Construction Tree</h2>
        <button className="button secondary" style={{ padding: '4px', border: 'none' }}>
          <Plus size={16} />
        </button>
      </div>
      
      <div style={{ flex: 1, overflowX: 'auto', overflowY: 'hidden', padding: '12px', display: 'flex' }}>
        {csgTree.length === 0 ? (
          <div style={{ width: '100%', textAlign: 'center', padding: '2rem 0', color: 'var(--text-secondary)' }}>
            No construction steps yet.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'row', gap: '12px', height: '100%', alignItems: 'center' }}>
            {csgTree.map((step, index) => {
              const Icon = getIcon(step.operation);
              return (
                <div 
                  key={index} 
                  style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '12px',
                    padding: '12px',
                    background: 'rgba(150, 150, 150, 0.05)',
                    borderRadius: '8px',
                    border: '1px solid var(--border-color)',
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                    minWidth: '220px',
                    flexShrink: 0,
                    height: '100%'
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.borderColor = 'var(--accent-primary)';
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = 'var(--shadow-md)';
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.borderColor = 'var(--border-color)';
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  <div style={{ color: 'var(--accent-hover)', background: 'rgba(150, 150, 150, 0.1)', padding: '10px', borderRadius: '50%' }}>
                    <Icon size={20} />
                  </div>
                  <div style={{ flex: 1, overflow: 'hidden' }}>
                    <div style={{ fontSize: '0.875rem', fontWeight: 600, textTransform: 'capitalize', color: 'var(--text-primary)' }}>
                      {index + 1}. {step.operation}
                    </div>
                    <div style={{ 
                      fontSize: '0.75rem', 
                      color: 'var(--text-secondary)',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      marginTop: '4px'
                    }}>
                      {step.params && Object.entries(step.params)
                        .filter(([k,v]) => v !== null)
                        .map(([k, v]) => `${k}:${v}`)
                        .join(', ')}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
