'use client';

import { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import {
  Wrench,
  Home,
  Search,
  SlidersHorizontal,
  Loader2,
  AlertTriangle,
  RefreshCw,
  Sun,
  Moon,
  Sunset,
  Layers,
  Box,
} from 'lucide-react';
import GalleryCard, { BenchmarkPart } from '@/components/gallery/GalleryCard';
import ComparisonModal from '@/components/gallery/ComparisonModal';
import LaserGridBackground from '@/components/ui/LaserGridBackground';
import { useEditorStore } from '@/lib/store';

export default function GalleryPage() {
  const [data, setData] = useState<{ summary: any; parts: BenchmarkPart[] } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filterCategory, setFilterCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'iou-desc' | 'iou-asc' | 'id'>('iou-desc');
  const [selectedPart, setSelectedPart] = useState<BenchmarkPart | null>(null);

  const { theme, setTheme } = useEditorStore();

  const cycleTheme = () => {
    if (theme === 'light') setTheme('dim');
    else if (theme === 'dim') setTheme('dark');
    else setTheme('light');
  };

  const ThemeIcon = theme === 'light' ? Sun : theme === 'dim' ? Sunset : Moon;

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:8000/api/benchmark/results');
      if (!res.ok) {
        throw new Error(`Failed to load benchmark results (${res.status})`);
      }
      const json = await res.json();
      setData(json);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Could not connect to backend server. Make sure server is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const filteredParts = useMemo(() => {
    if (!data) return [];
    let parts = [...data.parts];

    // Filter by Category
    if (filterCategory === 'perfect') {
      parts = parts.filter((p) => p.category === 'perfect');
    } else if (filterCategory === 'high') {
      parts = parts.filter((p) => p.category === 'perfect' || p.category === 'high');
    } else if (filterCategory === 'partial') {
      parts = parts.filter((p) => p.category === 'partial');
    } else if (filterCategory === 'low') {
      parts = parts.filter((p) => p.category === 'low');
    } else if (filterCategory === 'failed') {
      parts = parts.filter((p) => p.status === 'failed');
    }

    // Filter by Search Query
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      parts = parts.filter(
        (p) => p.part_id.toLowerCase().includes(q) || p.clean_id.toLowerCase().includes(q)
      );
    }

    // Sort
    if (sortBy === 'iou-desc') {
      parts.sort((a, b) => b.iou - a.iou);
    } else if (sortBy === 'iou-asc') {
      parts.sort((a, b) => a.iou - b.iou);
    } else if (sortBy === 'id') {
      parts.sort((a, b) => a.clean_id.localeCompare(b.clean_id));
    }

    return parts;
  }, [data, filterCategory, searchQuery, sortBy]);

  return (
    <div
      style={{
        height: '100vh',
        width: '100vw',
        overflowY: 'auto',
        position: 'relative',
        background: 'radial-gradient(ellipse at 50% 0%, rgba(59, 130, 246, 0.12) 0%, var(--bg-color) 70%)',
        color: 'var(--text-primary)',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Dynamic CAD Blueprint Grid & Moving Laser Scanner Background */}
      <LaserGridBackground />

      {/* Top Navbar */}
      <header
        className="glass-panel"
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 50,
          borderRadius: 0,
          borderLeft: 'none',
          borderRight: 'none',
          borderTop: 'none',
          padding: '14px 32px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <Link href="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Box size={24} style={{ color: 'var(--accent-primary)' }} />
            <span style={{ fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.02em', color: 'var(--text-primary)' }}>
              AIMDEA
            </span>
          </Link>
          <span style={{ color: 'var(--border-color)' }}>|</span>
          <span style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
            Gallery
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Link href="/" className="button secondary" style={{ padding: '8px 14px', borderRadius: '20px', gap: '6px', textDecoration: 'none' }}>
            <Home size={15} />
            <span>Home</span>
          </Link>
          <Link href="/editor" className="button" style={{ padding: '8px 16px', borderRadius: '20px', gap: '6px', textDecoration: 'none' }}>
            <span>Launch Editor</span>
          </Link>
          <button
            className="button secondary"
            style={{ padding: '8px', borderRadius: '50%' }}
            onClick={cycleTheme}
            title="Toggle Theme"
          >
            <ThemeIcon size={16} />
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main style={{ maxWidth: '1400px', width: '100%', margin: '0 auto', padding: '32px 24px 64px 24px', position: 'relative', zIndex: 10 }}>
        {/* Page Title */}
        <div style={{ marginBottom: '24px' }}>
          <h1 style={{ fontSize: '2rem', fontWeight: 800, letterSpacing: '-0.02em', margin: 0 }}>
            Enhanced Hybrid Benchmark Results
          </h1>
        </div>

        {/* Loading State */}
        {loading && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '80px 0', gap: '16px' }}>
            <Loader2 size={36} className="animate-spin" style={{ color: 'var(--accent-primary)' }} />
            <span style={{ color: 'var(--text-secondary)' }}>Loading benchmark metrics and 100 evaluation parts...</span>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="glass-panel" style={{ padding: '32px', textAlign: 'center', margin: '40px 0', borderColor: 'rgba(239, 68, 68, 0.4)' }}>
            <AlertTriangle size={36} style={{ color: '#f87171', margin: '0 auto 12px auto' }} />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '8px' }}>Failed to Load Benchmark Data</h3>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '16px', maxWidth: '500px', margin: '0 auto 16px auto' }}>
              {error}
            </p>
            <button onClick={fetchData} className="button" style={{ gap: '6px' }}>
              <RefreshCw size={15} />
              <span>Retry Connection</span>
            </button>
          </div>
        )}

        {/* Loaded Content */}
        {!loading && !error && data && (
          <>
            {/* Controls Bar: Filters & Search */}
            <div
              className="glass-panel"
              style={{
                padding: '16px 20px',
                marginBottom: '24px',
                display: 'flex',
                flexWrap: 'wrap',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: '16px',
              }}
            >
              {/* Category Filter Pills */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {[
                  { id: 'all', label: `All (${data.summary.total_samples})` },
                  { id: 'perfect', label: `Perfect ≥0.95 (${data.summary.perfect_count})` },
                  { id: 'high', label: `High ≥0.50 (${data.summary.high_count})` },
                  { id: 'partial', label: `Partial 0.2-0.5 (${data.summary.partial_count})` },
                  { id: 'low', label: `Low <0.20 (${data.summary.low_count})` },
                  { id: 'failed', label: `Failed (${data.summary.vsr_failed})` },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setFilterCategory(tab.id)}
                    className={`button secondary ${filterCategory === tab.id ? 'active' : ''}`}
                    style={{
                      padding: '6px 14px',
                      borderRadius: '20px',
                      fontSize: '0.8rem',
                      background: filterCategory === tab.id ? 'var(--accent-primary)' : 'transparent',
                      color: filterCategory === tab.id ? '#fff' : 'var(--text-primary)',
                      borderColor: filterCategory === tab.id ? 'var(--accent-primary)' : 'var(--border-color)',
                    }}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Search & Sort Controls */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: '1', minWidth: '280px', justifyContent: 'flex-end' }}>
                <div style={{ position: 'relative', width: '220px' }}>
                  <Search
                    size={15}
                    style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }}
                  />
                  <input
                    type="text"
                    placeholder="Search part ID..."
                    className="input-field"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    style={{ paddingLeft: '32px', paddingRight: '12px', fontSize: '0.8rem', borderRadius: '20px' }}
                  />
                </div>

                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as any)}
                  className="input-field"
                  style={{
                    width: '210px',
                    fontSize: '0.8rem',
                    borderRadius: '20px',
                    cursor: 'pointer',
                    backgroundColor: 'var(--bg-panel)',
                    color: 'var(--text-primary)',
                    border: '1px solid var(--border-color)',
                    fontWeight: 500,
                  }}
                >
                  <option value="iou-desc" style={{ backgroundColor: 'var(--bg-color)', color: 'var(--text-primary)' }}>
                    Sort By: Highest IoU First
                  </option>
                  <option value="iou-asc" style={{ backgroundColor: 'var(--bg-color)', color: 'var(--text-primary)' }}>
                    Sort By: Lowest IoU First
                  </option>
                  <option value="id" style={{ backgroundColor: 'var(--bg-color)', color: 'var(--text-primary)' }}>
                    Sort By: Part ID
                  </option>
                </select>
              </div>
            </div>

            {/* Results Count & Subtitle */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', padding: '0 4px' }}>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Showing <strong>{filteredParts.length}</strong> of {data.summary.total_samples} parts
              </span>
            </div>

            {/* Parts Grid */}
            {filteredParts.length === 0 ? (
              <div className="glass-panel" style={{ padding: '60px 20px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                No parts match your current filter or search criteria.
              </div>
            ) : (
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
                  gap: '20px',
                }}
              >
                {filteredParts.map((part) => (
                  <GalleryCard key={part.clean_id} part={part} onSelect={(p) => setSelectedPart(p)} />
                ))}
              </div>
            )}
          </>
        )}
      </main>

      {/* Comparison Modal */}
      {selectedPart && (
        <ComparisonModal part={selectedPart} onClose={() => setSelectedPart(null)} />
      )}
    </div>
  );
}
