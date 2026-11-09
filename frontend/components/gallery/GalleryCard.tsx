'use client';

import { useState } from 'react';
import { Eye, Download, Box, CheckCircle2, XCircle } from 'lucide-react';

export interface BenchmarkPart {
  part_id: string;
  clean_id: string;
  iou: number;
  status: 'valid' | 'failed';
  category: 'perfect' | 'high' | 'partial' | 'low' | 'failed';
  has_stl: boolean;
  has_step: boolean;
  has_image: boolean;
  code: string;
}

export default function GalleryCard({
  part,
  onSelect,
}: {
  part: BenchmarkPart;
  onSelect: (part: BenchmarkPart) => void;
}) {
  const [isHovered, setIsHovered] = useState(false);

  const getBadgeColor = (category: string, iou: number) => {
    if (category === 'failed') return { bg: 'rgba(239, 68, 68, 0.15)', text: '#f87171', border: 'rgba(239, 68, 68, 0.3)' };
    if (iou >= 0.80) return { bg: 'rgba(16, 185, 129, 0.15)', text: '#34d399', border: 'rgba(16, 185, 129, 0.3)' };
    if (iou >= 0.50) return { bg: 'rgba(245, 158, 11, 0.15)', text: '#fbbf24', border: 'rgba(245, 158, 11, 0.3)' };
    if (iou >= 0.20) return { bg: 'rgba(249, 115, 22, 0.15)', text: '#fb923c', border: 'rgba(249, 115, 22, 0.3)' };
    return { bg: 'rgba(244, 63, 94, 0.15)', text: '#fb7185', border: 'rgba(244, 63, 94, 0.3)' };
  };

  const badge = getBadgeColor(part.category, part.iou);

  return (
    <div
      className="glass-panel"
      style={{
        display: 'flex',
        flexDirection: 'column',
        borderRadius: '12px',
        overflow: 'hidden',
        cursor: 'pointer',
        transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
        transform: isHovered ? 'translateY(-4px)' : 'none',
        boxShadow: isHovered ? 'var(--shadow-lg)' : 'var(--shadow-sm)',
        borderColor: isHovered ? 'var(--accent-primary)' : 'var(--border-color)',
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={() => onSelect(part)}
    >
      {/* Target Image Preview Area */}
      <div style={{ position: 'relative', width: '100%', aspectRatio: '1 / 1', background: '#ffffff', overflow: 'hidden' }}>
        {part.has_image ? (
          <img
            src={`http://localhost:8000/api/benchmark/part/${part.clean_id}/image?v=hf_2d`}
            alt={`Target ${part.part_id}`}
            loading="lazy"
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              padding: '12px',
              transition: 'transform 0.3s ease',
              transform: isHovered ? 'scale(1.05)' : 'scale(1)',
            }}
          />
        ) : (
          <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
            No Preview
          </div>
        )}

        {/* Status Tag Overlay */}
        <div style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          display: 'flex',
          gap: '6px',
        }}>
          <span style={{
            fontSize: '0.7rem',
            fontWeight: 700,
            padding: '3px 8px',
            borderRadius: '12px',
            background: badge.bg,
            color: badge.text,
            border: `1px solid ${badge.border}`,
            backdropFilter: 'var(--glass-blur)',
          }}>
            {part.status === 'failed' ? 'FAILED' : `IoU: ${part.iou.toFixed(3)}`}
          </span>
        </div>

        {/* Hover Action Pill */}
        <div style={{
          position: 'absolute',
          bottom: '10px',
          right: '10px',
          opacity: isHovered ? 1 : 0,
          transition: 'opacity 0.2s ease',
          background: 'var(--accent-primary)',
          color: '#fff',
          borderRadius: '20px',
          padding: '4px 10px',
          fontSize: '0.75rem',
          fontWeight: 500,
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          boxShadow: 'var(--shadow-md)',
        }}>
          <Eye size={13} />
          <span>Inspect 3D</span>
        </div>
      </div>

      {/* Card Footer Details */}
      <div style={{ padding: '12px 14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid var(--border-color)' }}>
        <div>
          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            {part.clean_id}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            {part.status === 'valid' ? 'Watertight Solid' : 'Kernel Error'}
          </div>
        </div>

        <div style={{ display: 'flex', gap: '6px' }} onClick={(e) => e.stopPropagation()}>
          {part.has_step && (
            <a
              href={`http://localhost:8000/api/benchmark/part/${part.clean_id}/step`}
              download={`${part.clean_id}.step`}
              className="button secondary"
              style={{ padding: '6px 8px', fontSize: '0.7rem', borderRadius: '6px' }}
              title="Download STEP file"
            >
              <Download size={13} />
              <span>STEP</span>
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
