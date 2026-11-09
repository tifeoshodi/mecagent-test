'use client';

import { CheckCircle2, TrendingUp, Award, Layers, FileDown, AlertCircle } from 'lucide-react';

interface MetricsSummary {
  experiment_name: string;
  total_samples: number;
  vsr: number;
  vsr_percentage: number;
  vsr_successful: number;
  vsr_failed: number;
  mean_iou_best: number;
  iou_valid_count: number;
  perfect_count: number;
  high_count: number;
  partial_count: number;
  low_count: number;
  step_export_count: number;
}

export default function MetricsDashboard({ summary }: { summary: MetricsSummary }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '16px', marginBottom: '24px' }}>
      {/* VSR Card */}
      <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
            VALID SOLID RATE (VSR)
          </span>
          <CheckCircle2 size={18} style={{ color: '#10b981' }} />
        </div>
        <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          {summary.vsr_percentage}%
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
          {summary.vsr_successful} / {summary.total_samples} parts compiled to watertight B-Rep
        </div>
        <div style={{ height: '4px', background: 'rgba(150,150,150,0.2)', borderRadius: '2px', overflow: 'hidden', marginTop: '4px' }}>
          <div style={{ height: '100%', width: `${summary.vsr_percentage}%`, background: '#10b981' }} />
        </div>
      </div>

      {/* Mean IoU Card */}
      <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
            MEAN BEST 3D IoU
          </span>
          <TrendingUp size={18} style={{ color: 'var(--accent-primary)' }} />
        </div>
        <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-hover)' }}>
          {summary.mean_iou_best.toFixed(4)}
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
          Evaluated across 86 valid models (38,400 voxel grid)
        </div>
        <div style={{ height: '4px', background: 'rgba(150,150,150,0.2)', borderRadius: '2px', overflow: 'hidden', marginTop: '4px' }}>
          <div style={{ height: '100%', width: `${Math.min(100, summary.mean_iou_best * 100)}%`, background: 'var(--accent-primary)' }} />
        </div>
      </div>

      {/* High Fidelity / Perfect Matches */}
      <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
            HIGH FIDELITY (≥ 0.50)
          </span>
          <Award size={18} style={{ color: '#f59e0b' }} />
        </div>
        <div style={{ fontSize: '2rem', fontWeight: 700, color: '#f59e0b' }}>
          {summary.high_count} <span style={{ fontSize: '1rem', fontWeight: 500, color: 'var(--text-secondary)' }}>({summary.perfect_count} perfect)</span>
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
          10 parts achieved &gt;0.95 IoU with perfect CAD topology
        </div>
        <div style={{ height: '4px', background: 'rgba(150,150,150,0.2)', borderRadius: '2px', overflow: 'hidden', marginTop: '4px' }}>
          <div style={{ height: '100%', width: `${(summary.high_count / summary.total_samples) * 100}%`, background: '#f59e0b' }} />
        </div>
      </div>

      {/* STEP Exports Card */}
      <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
            CAD STEP EXPORTS
          </span>
          <FileDown size={18} style={{ color: '#8b5cf6' }} />
        </div>
        <div style={{ fontSize: '2rem', fontWeight: 700, color: '#a78bfa' }}>
          {summary.step_export_count}
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
          Ready for SolidWorks, Fusion 360, and FreeCAD
        </div>
        <div style={{ height: '4px', background: 'rgba(150,150,150,0.2)', borderRadius: '2px', overflow: 'hidden', marginTop: '4px' }}>
          <div style={{ height: '100%', width: `${(summary.step_export_count / summary.total_samples) * 100}%`, background: '#8b5cf6' }} />
        </div>
      </div>
    </div>
  );
}
