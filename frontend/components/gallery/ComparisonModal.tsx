'use client';

import { useState, useEffect, useMemo, Suspense, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { Canvas } from '@react-three/fiber';
import { TrackballControls, Grid, Environment, ContactShadows, Center } from '@react-three/drei';
import * as THREE from 'three';
import { STLLoader } from 'three-stdlib';
import { X, Download, Copy, Check, ExternalLink, Layers, Loader2, RotateCcw, Hand } from 'lucide-react';
import { BenchmarkPart } from './GalleryCard';
import { useEditorStore } from '@/lib/store';

function RemoteSTLModel({ url, wireframe }: { url: string; wireframe: boolean }) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    setLoading(true);
    const loader = new STLLoader();
    loader.load(
      url,
      (geom) => {
        if (active) {
          geom.computeVertexNormals();
          geom.computeBoundingBox();
          const box = geom.boundingBox;
          if (box) {
            const size = new THREE.Vector3();
            box.getSize(size);
            const maxDim = Math.max(size.x, size.y, size.z);
            if (maxDim > 1e-4) {
              const scale = 30 / maxDim;
              geom.scale(scale, scale, scale);
              geom.computeBoundingBox();
            }
          }
          setGeometry(geom);
          setLoading(false);
        }
      },
      undefined,
      (err) => {
        console.error('Error loading STL:', err);
        if (active) setLoading(false);
      }
    );
    return () => {
      active = false;
    };
  }, [url]);

  if (loading || !geometry) {
    return (
      <Center>
        <mesh position={[0, 10, 0]}>
          <octahedronGeometry args={[12, 2]} />
          <meshStandardMaterial color="#60a5fa" wireframe transparent opacity={0.6} />
        </mesh>
      </Center>
    );
  }

  return (
    <Center top>
      <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]} castShadow receiveShadow>
        <meshStandardMaterial
          color="#3b82f6"
          metalness={0.5}
          roughness={0.25}
          side={THREE.DoubleSide}
          wireframe={wireframe}
        />
      </mesh>
    </Center>
  );
}

function ModalControlsManager({
  controlsRef,
  isPanMode,
}: {
  controlsRef: React.RefObject<any>;
  isPanMode: boolean;
}) {
  const isPanModeRef = useRef(isPanMode);
  isPanModeRef.current = isPanMode;

  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls) return;

    controls.noRotate = isPanMode;
    controls.noPan = false;
    controls.handleResize?.();
    if (controls.domElement) {
      controls.domElement.style.touchAction = 'none';
      controls.domElement.style.userSelect = 'none';
    }

    if (!controls.__customPanBound) {
      controls.__customPanBound = true;

      controls.onPointerDown = (event: PointerEvent) => {
        if (controls.enabled === false) return;
        try {
          controls.domElement?.setPointerCapture?.(event.pointerId);
        } catch {}
        controls.onMouseDown(event);
      };

      controls.onPointerMove = (event: PointerEvent) => {
        if (controls.enabled === false) return;
        controls.onMouseMove(event);
      };

      controls.onPointerUp = (event: PointerEvent) => {
        if (controls.enabled === false) return;
        try {
          if (controls.domElement?.hasPointerCapture?.(event.pointerId)) {
            controls.domElement.releasePointerCapture(event.pointerId);
          }
        } catch {}
        controls.onMouseUp();
      };

      controls.onMouseDown = (event: MouseEvent | PointerEvent) => {
        if (!controls.domElement || controls.enabled === false) return;
        event.preventDefault?.();

        if (controls._state === controls.STATE.NONE) {
          if (isPanModeRef.current && (event.button === 0 || event.button === 2)) {
            controls._state = controls.STATE.PAN;
          } else {
            switch (event.button) {
              case controls.mouseButtons.LEFT:
                controls._state = controls.STATE.ROTATE;
                break;
              case controls.mouseButtons.MIDDLE:
                controls._state = controls.STATE.ZOOM;
                break;
              case controls.mouseButtons.RIGHT:
                controls._state = controls.STATE.PAN;
                break;
            }
          }
        }

        const state = controls._keyState !== controls.STATE.NONE ? controls._keyState : controls._state;

        if (state === controls.STATE.ROTATE && !controls.noRotate) {
          controls._moveCurr.copy(controls.getMouseOnCircle(event.pageX, event.pageY));
          controls._movePrev.copy(controls._moveCurr);
        } else if (state === controls.STATE.ZOOM && !controls.noZoom) {
          controls._zoomStart.copy(controls.getMouseOnScreen(event.pageX, event.pageY));
          controls._zoomEnd.copy(controls._zoomStart);
        } else if (state === controls.STATE.PAN && !controls.noPan) {
          controls._panStart.copy(controls.getMouseOnScreen(event.pageX, event.pageY));
          controls._panEnd.copy(controls._panStart);
        }

        controls.domElement.ownerDocument.addEventListener("pointermove", controls.onPointerMove);
        controls.domElement.ownerDocument.addEventListener("pointerup", controls.onPointerUp);
        controls.dispatchEvent(controls.startEvent);
      };
    }
  }, [isPanMode]);

  return null;
}

export default function ComparisonModal({
  part,
  onClose,
}: {
  part: BenchmarkPart;
  onClose: () => void;
}) {
  const router = useRouter();
  const [wireframe, setWireframe] = useState(false);
  const [isPanMode, setIsPanMode] = useState(false);
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<'3d' | 'code'>('3d');
  const { setStlData, setCadqueryCode, theme } = useEditorStore();
  const controlsRef = useRef<any>(null);

  const handleResetView = () => {
    if (controlsRef.current) {
      if (typeof controlsRef.current.reset === 'function') {
        controlsRef.current.reset();
      }
      if (controlsRef.current.object) {
        controlsRef.current.object.position.set(45, 45, 45);
        controlsRef.current.object.up.set(0, 1, 0);
        controlsRef.current.object.lookAt(0, 15, 0);
        controlsRef.current.object.updateProjectionMatrix();
      }
      if (controlsRef.current.target) {
        controlsRef.current.target.set(0, 15, 0);
      }
      controlsRef.current.update?.();
    }
  };

  const handleCopyCode = async () => {
    if (!part.code) return;
    try {
      await navigator.clipboard.writeText(part.code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error(e);
    }
  };

  const handleOpenInEditor = async () => {
    // If STL exists, fetch base64 to put into editor store
    if (part.has_stl) {
      try {
        const res = await fetch(`http://localhost:8000/api/benchmark/part/${part.clean_id}/stl`);
        const blob = await res.blob();
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64data = reader.result as string;
          const cleanB64 = base64data.split(',')[1] || base64data;
          setStlData(cleanB64);
          if (part.code) setCadqueryCode(part.code);
          router.push('/editor');
        };
        reader.readAsDataURL(blob);
        return;
      } catch (e) {
        console.error('Failed to load into editor:', e);
      }
    }
    if (part.code) setCadqueryCode(part.code);
    router.push('/editor');
  };

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 1000,
        backgroundColor: 'rgba(0, 0, 0, 0.75)',
        backdropFilter: 'blur(8px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '24px',
      }}
      onClick={onClose}
    >
      <div
        className="glass-panel"
        style={{
          width: '100%',
          maxWidth: '1100px',
          height: '88vh',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          borderRadius: '16px',
          boxShadow: 'var(--shadow-lg)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div style={{
          padding: '16px 24px',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 700 }}>
              Sample: {part.clean_id}
            </h2>
            <span style={{
              fontSize: '0.75rem',
              fontWeight: 700,
              padding: '4px 10px',
              borderRadius: '12px',
              background: part.status === 'valid' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
              color: part.status === 'valid' ? '#34d399' : '#f87171',
              border: '1px solid var(--border-color)',
            }}>
              IoU: {part.iou.toFixed(4)} ({part.category.toUpperCase()})
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button
              onClick={handleOpenInEditor}
              className="button secondary"
              style={{ padding: '6px 12px', fontSize: '0.8rem', gap: '6px' }}
              title="Load into 3D Workspace"
            >
              <ExternalLink size={14} />
              <span>Open in Editor</span>
            </button>
            <button
              onClick={onClose}
              className="button secondary"
              style={{ padding: '8px', borderRadius: '50%' }}
              title="Close modal"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Modal Body: Two Columns */}
        <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', overflow: 'hidden' }}>
          {/* Left Column: Target Reference Image */}
          <div style={{
            padding: '24px',
            borderRight: '1px solid var(--border-color)',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            background: '#ffffff',
            position: 'relative',
          }}>
            <div style={{
              position: 'absolute',
              top: '16px',
              left: '16px',
              fontSize: '0.75rem',
              fontWeight: 700,
              color: '#334155',
              letterSpacing: '0.05em',
            }}>
              [2D REFERENCE DRAWING (HUGGING FACE)]
            </div>

            {part.has_image ? (
              <img
                src={`http://localhost:8000/api/benchmark/part/${part.clean_id}/image?v=hf_2d`}
                alt={`Target ${part.clean_id}`}
                style={{
                  maxWidth: '100%',
                  maxHeight: '400px',
                  objectFit: 'contain',
                  borderRadius: '8px',
                  boxShadow: 'var(--shadow-md)',
                }}
              />
            ) : (
              <div style={{ color: 'var(--text-secondary)' }}>No target image available</div>
            )}
          </div>

          {/* Right Column: 3D Viewport or CadQuery Code */}
          <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', height: '100%' }}>
            {/* View Toggle */}
            <div style={{
              position: 'absolute',
              top: '16px',
              left: '16px',
              zIndex: 10,
              display: 'flex',
              gap: '6px',
            }}>
              <button
                className={`button secondary ${activeTab === '3d' ? 'active' : ''}`}
                style={{
                  padding: '6px 12px',
                  fontSize: '0.75rem',
                  borderRadius: '20px',
                  background: activeTab === '3d' ? 'var(--accent-primary)' : 'var(--bg-panel)',
                  color: activeTab === '3d' ? '#fff' : 'var(--text-primary)',
                  backdropFilter: 'var(--glass-blur)',
                }}
                onClick={() => setActiveTab('3d')}
              >
                3D Model Preview
              </button>
              <button
                className={`button secondary ${activeTab === 'code' ? 'active' : ''}`}
                style={{
                  padding: '6px 12px',
                  fontSize: '0.75rem',
                  borderRadius: '20px',
                  background: activeTab === 'code' ? 'var(--accent-primary)' : 'var(--bg-panel)',
                  color: activeTab === 'code' ? '#fff' : 'var(--text-primary)',
                  backdropFilter: 'var(--glass-blur)',
                }}
                onClick={() => setActiveTab('code')}
              >
                CadQuery Code
              </button>
            </div>

            {/* Wireframe & Reset View toggles for 3D */}
            {activeTab === '3d' && (
              <div style={{ position: 'absolute', top: '16px', right: '16px', zIndex: 10, display: 'flex', gap: '8px' }}>
                <button
                  className={`button secondary ${isPanMode ? 'active' : ''}`}
                  style={{
                    padding: '6px 12px',
                    fontSize: '0.75rem',
                    borderRadius: '20px',
                    background: isPanMode ? 'var(--accent-primary)' : 'var(--bg-panel)',
                    color: isPanMode ? '#fff' : 'var(--text-primary)',
                    border: isPanMode ? '1px solid var(--accent-primary)' : '1px solid var(--border-color)',
                    backdropFilter: 'var(--glass-blur)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '5px',
                  }}
                  onClick={() => setIsPanMode(!isPanMode)}
                  title={isPanMode ? 'Pan Mode Active: Drag to pan workspace without rotating' : 'Orbit Mode: Drag to rotate 360°'}
                >
                  <Hand size={13} />
                  <span>{isPanMode ? 'Pan Mode' : 'Orbit'}</span>
                </button>

                <button
                  className="button secondary"
                  style={{
                    padding: '6px 12px',
                    fontSize: '0.75rem',
                    borderRadius: '20px',
                    background: 'var(--bg-panel)',
                    color: 'var(--text-primary)',
                    backdropFilter: 'var(--glass-blur)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '5px',
                  }}
                  onClick={handleResetView}
                  title="Reset Orbit Rotation to Centered Default View"
                >
                  <RotateCcw size={13} />
                  <span>Reset View</span>
                </button>

                <button
                  className="button secondary"
                  style={{
                    padding: '6px 12px',
                    fontSize: '0.75rem',
                    borderRadius: '20px',
                    background: wireframe ? 'var(--accent-primary)' : 'var(--bg-panel)',
                    color: wireframe ? '#fff' : 'var(--text-primary)',
                    backdropFilter: 'var(--glass-blur)',
                  }}
                  onClick={() => setWireframe(!wireframe)}
                  title="Toggle Wireframe"
                >
                  <Layers size={13} />
                  <span>{wireframe ? 'Shaded' : 'Wireframe'}</span>
                </button>
              </div>
            )}

            {activeTab === '3d' ? (
              <div style={{
                width: '100%',
                height: '100%',
                background: 'radial-gradient(circle at center, var(--bg-panel) 0%, var(--bg-color) 100%)',
                cursor: isPanMode ? 'grab' : 'default',
              }}>
                {part.has_stl ? (
                  <Canvas key={part.clean_id} camera={{ position: [45, 45, 45], fov: 45, near: 0.05, far: 10000 }}>
                    <ambientLight intensity={0.6} />
                    <directionalLight position={[20, 30, 20]} intensity={1.2} castShadow />
                    <directionalLight position={[-20, 20, -20]} intensity={0.5} />
                    <Environment preset="city" />

                    {/* CAD Grid Floor positioned at y = 0 */}
                    <Grid
                      position={[0, 0, 0]}
                      args={[100, 100]}
                      cellSize={2}
                      cellThickness={0.7}
                      cellColor={theme === 'light' ? '#cbd5e1' : '#1e293b'}
                      sectionSize={10}
                      sectionThickness={1.2}
                      sectionColor={theme === 'light' ? '#94a3b8' : '#38bdf8'}
                      fadeDistance={100}
                      fadeStrength={1.2}
                      infiniteGrid
                    />

                    <Suspense fallback={null}>
                      <RemoteSTLModel
                        url={`http://localhost:8000/api/benchmark/part/${part.clean_id}/stl`}
                        wireframe={wireframe}
                      />
                    </Suspense>

                    <ModalControlsManager
                      controlsRef={controlsRef}
                      isPanMode={isPanMode}
                    />

                    <TrackballControls
                      ref={controlsRef}
                      makeDefault
                      rotateSpeed={2.5}
                      zoomSpeed={1.2}
                      panSpeed={1.0}
                      dynamicDampingFactor={0.2}
                      minDistance={0.5}
                      maxDistance={1500}
                    />
                  </Canvas>
                ) : (
                  <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: '#f87171' }}>
                    Solid generation failed (Kernel syntax or topological invalidity)
                  </div>
                )}
              </div>
            ) : (
              <div style={{ flex: 1, padding: '54px 20px 20px 20px', overflowY: 'auto', background: 'rgba(0,0,0,0.2)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    PYTHON CADQUERY SCRIPT
                  </span>
                  <button
                    onClick={handleCopyCode}
                    className="button secondary"
                    style={{ padding: '4px 8px', fontSize: '0.75rem', gap: '4px' }}
                  >
                    {copied ? <Check size={13} style={{ color: '#10b981' }} /> : <Copy size={13} />}
                    <span>{copied ? 'Copied' : 'Copy'}</span>
                  </button>
                </div>
                <pre style={{
                  background: 'rgba(0,0,0,0.4)',
                  padding: '16px',
                  borderRadius: '8px',
                  fontSize: '0.8rem',
                  fontFamily: 'monospace',
                  color: '#93c5fd',
                  overflowX: 'auto',
                  lineHeight: '1.5',
                }}>
                  {part.code || '# No CadQuery prediction code recorded for this sample.'}
                </pre>
              </div>
            )}
          </div>
        </div>

        {/* Modal Footer Controls */}
        <div style={{
          padding: '16px 24px',
          borderTop: '1px solid var(--border-color)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'var(--bg-panel)',
        }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Benchmark Run: Enhanced Hybrid (Gemini 3.1 Pro + Flash Repair + DINOv2)
          </div>

          <div style={{ display: 'flex', gap: '10px' }}>
            {part.has_stl && (
              <a
                href={`http://localhost:8000/api/benchmark/part/${part.clean_id}/stl`}
                download={`${part.clean_id}.stl`}
                className="button secondary"
                style={{ fontSize: '0.8rem', gap: '6px' }}
              >
                <Download size={14} />
                <span>Download STL</span>
              </a>
            )}

            {part.has_step && (
              <a
                href={`http://localhost:8000/api/benchmark/part/${part.clean_id}/step`}
                download={`${part.clean_id}.step`}
                className="button"
                style={{ fontSize: '0.8rem', gap: '6px', background: '#10b981' }}
              >
                <Download size={14} />
                <span>Download STEP (.step)</span>
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
