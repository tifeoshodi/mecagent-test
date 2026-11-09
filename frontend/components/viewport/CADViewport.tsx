'use client';

import { Canvas } from '@react-three/fiber';
import { TrackballControls, Grid, Environment, Center } from '@react-three/drei';
import { Suspense, useMemo, useRef, useState, useEffect } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { STLLoader } from 'three-stdlib';
import { useEditorStore } from '@/lib/store';
import { Layers, Loader2, RotateCcw, Hand, Move, ZoomIn, ZoomOut } from 'lucide-react';

function STLModel({
  base64Data,
  wireframe,
  onCenterComputed,
}: {
  base64Data: string;
  wireframe: boolean;
  onCenterComputed?: (height: number) => void;
}) {
  const geometry = useMemo(() => {
    try {
      const binaryString = window.atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      const loader = new STLLoader();
      const geom = loader.parse(bytes.buffer);
      geom.computeVertexNormals();
      geom.computeBoundingBox();
      const box = geom.boundingBox;
      if (box) {
        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        if (maxDim > 1e-4) {
          const scale = 16 / maxDim;
          geom.scale(scale, scale, scale);
          geom.computeBoundingBox();
        }
      }
      return geom;
    } catch (e) {
      console.error('Failed to parse STL', e);
      return null;
    }
  }, [base64Data]);

  if (!geometry) return null;

  return (
    <Center
      top
      onCentered={(info) => {
        if (info && info.height) {
          onCenterComputed?.(info.height / 2);
        }
      }}
    >
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

function LoadingMesh() {
  return (
    <Center top>
      <mesh position={[0, 7, 0]}>
        <octahedronGeometry args={[12, 2]} />
        <meshStandardMaterial color="#60a5fa" wireframe transparent opacity={0.6} />
      </mesh>
    </Center>
  );
}

function SceneController({
  controlsRef,
  targetY,
  isPanMode,
  onZoomChange,
}: {
  controlsRef: React.RefObject<any>;
  targetY: number;
  isPanMode: boolean;
  onZoomChange: (pct: number) => void;
}) {
  const { camera } = useThree();
  const initialized = useRef(false);
  const lastPct = useRef(100);

  // Initialize camera and controls target centered at object
  useEffect(() => {
    if (controlsRef.current) {
      if (typeof window !== 'undefined') {
        (window as any).__cadControls = controlsRef.current;
      }
      controlsRef.current.target.set(0, targetY, 0);
      if (!initialized.current) {
        camera.position.set(38, targetY + 24, 38);
        camera.up.set(0, 1, 0);
        camera.lookAt(0, targetY, 0);
        camera.updateProjectionMatrix();
        initialized.current = true;
      }
      controlsRef.current.update?.();
    }
  }, [camera, targetY]);

  const isPanModeRef = useRef(isPanMode);
  isPanModeRef.current = isPanMode;

  // Sync pan mode and handle reliable left-click, right-click, and touch panning
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
            // Left click (0) or Right click (2) in Pan Mode: directly set PAN state
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

  // Track zoom level live
  useFrame(() => {
    if (controlsRef.current && controlsRef.current.object) {
      const cam = controlsRef.current.object;
      const target = controlsRef.current.target || new THREE.Vector3(0, targetY, 0);
      const dist = cam.position.distanceTo(target);
      const d0 = 58.85; // baseline 100% distance from (38, targetY+24, 38) to (0, targetY, 0)
      const pct = Math.round((d0 / Math.max(dist, 0.001)) * 100);
      if (Math.abs(pct - lastPct.current) >= 1) {
        lastPct.current = pct;
        onZoomChange(pct);
      }
    }
  });

  return null;
}

export default function CADViewport() {
  const { stlData, isGenerating, wireframe, setWireframe, theme } = useEditorStore();
  const controlsRef = useRef<any>(null);
  const [isPanMode, setIsPanMode] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [zoomPct, setZoomPct] = useState(100);
  const [targetY, setTargetY] = useState(8);
  const targetYRef = useRef(8);

  targetYRef.current = targetY;

  // Logarithmic slider value (0 to 100) mapped to 10% to 6400%
  const sliderVal = useMemo(() => {
    const clamped = Math.max(10, Math.min(6400, zoomPct));
    return (Math.log(clamped / 10) / Math.log(640)) * 100;
  }, [zoomPct]);

  // Spacebar shortcut to temporarily engage Pan mode (CAD standard)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (e.code === 'Space' && !e.repeat && tag !== 'INPUT' && tag !== 'TEXTAREA') {
        e.preventDefault();
        setIsPanMode(true);
      }
    };
    const handleKeyUp = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (e.code === 'Space' && tag !== 'INPUT' && tag !== 'TEXTAREA') {
        e.preventDefault();
        setIsPanMode(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  const handleResetView = () => {
    if (controlsRef.current) {
      if (typeof controlsRef.current.reset === 'function') {
        controlsRef.current.reset();
      }
      const cy = targetYRef.current;
      controlsRef.current.target.set(0, cy, 0);
      if (controlsRef.current.object) {
        controlsRef.current.object.position.set(38, cy + 24, 38);
        controlsRef.current.object.up.set(0, 1, 0);
        controlsRef.current.object.lookAt(0, cy, 0);
        controlsRef.current.object.updateProjectionMatrix();
      }
      controlsRef.current.update?.();
      setZoomPct(100);
    }
  };

  const applyZoom = (targetPercentage: number) => {
    if (controlsRef.current && controlsRef.current.object) {
      const cam = controlsRef.current.object;
      const target = controlsRef.current.target || new THREE.Vector3(0, targetYRef.current, 0);
      const dir = new THREE.Vector3().subVectors(cam.position, target).normalize();
      const d0 = 58.85;
      const desiredDist = d0 / (targetPercentage / 100);
      const clampedDist = Math.max(0.5, Math.min(1500, desiredDist));
      cam.position.copy(target).addScaledVector(dir, clampedDist);
      cam.updateProjectionMatrix();
      controlsRef.current.update?.();
      setZoomPct(targetPercentage);
    }
  };

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        background: 'radial-gradient(circle at center, var(--bg-panel) 0%, var(--bg-color) 100%)',
        cursor: isPanMode ? (isDragging ? 'grabbing' : 'grab') : 'default',
        userSelect: 'none',
        WebkitUserSelect: 'none',
        touchAction: 'none',
      }}
      onPointerDown={() => setIsDragging(true)}
      onPointerUp={() => setIsDragging(false)}
      onPointerLeave={() => setIsDragging(false)}
    >
      {/* Top Left Viewport Overlay Controls */}
      <div style={{
        position: 'absolute',
        top: '16px',
        left: '16px',
        zIndex: 10,
        display: 'flex',
        gap: '8px',
        flexWrap: 'wrap',
      }}>
        {/* Wireframe / Shaded Mode Toggle */}
        <button
          className={`button secondary ${wireframe ? 'active' : ''}`}
          style={{
            padding: '8px 12px',
            fontSize: '0.75rem',
            background: wireframe ? 'var(--accent-primary)' : 'var(--bg-panel)',
            color: wireframe ? '#fff' : 'var(--text-primary)',
            backdropFilter: 'var(--glass-blur)',
            borderRadius: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
          onClick={() => setWireframe(!wireframe)}
          title="Toggle Wireframe Rendering"
        >
          <Layers size={14} />
          <span>{wireframe ? 'Shaded' : 'Wireframe'}</span>
        </button>

        {/* Pan / Orbit Mode Toggle Button */}
        <button
          className={`button secondary ${isPanMode ? 'active' : ''}`}
          style={{
            padding: '8px 12px',
            fontSize: '0.75rem',
            background: isPanMode ? 'var(--accent-primary)' : 'var(--bg-panel)',
            color: isPanMode ? '#fff' : 'var(--text-primary)',
            border: isPanMode ? '1px solid var(--accent-primary)' : '1px solid var(--border-color)',
            backdropFilter: 'var(--glass-blur)',
            borderRadius: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            boxShadow: isPanMode ? '0 0 10px rgba(59, 130, 246, 0.4)' : 'none',
            transition: 'all 0.2s ease',
          }}
          onClick={() => setIsPanMode(!isPanMode)}
          title={isPanMode ? 'Pan Mode Active: Click & drag to pan without rotating. Click to return to Orbit.' : 'Orbit Mode: Click & drag to rotate 360°. Click to activate Pan.'}
        >
          {isPanMode ? <Hand size={14} /> : <Move size={14} />}
          <span>{isPanMode ? 'Pan Mode' : 'Orbit'}</span>
        </button>

        {/* Reset View Button */}
        <button
          className="button secondary"
          style={{
            padding: '8px 12px',
            fontSize: '0.75rem',
            background: 'var(--bg-panel)',
            color: 'var(--text-primary)',
            backdropFilter: 'var(--glass-blur)',
            borderRadius: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
          onClick={handleResetView}
          title="Reset Orbit Rotation and Pan to Default Centered View"
        >
          <RotateCcw size={14} />
          <span>Reset View</span>
        </button>

        {/* Separator */}
        <div style={{ width: '1px', height: '20px', background: 'var(--border-color)', margin: '0 2px', alignSelf: 'center' }} />

        {/* Zoom Slider & Live Percentage */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          background: 'var(--bg-panel)',
          backdropFilter: 'var(--glass-blur)',
          border: '1px solid var(--border-color)',
          borderRadius: '20px',
          padding: '4px 12px',
        }}>
          <button
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              padding: '2px',
            }}
            onClick={() => applyZoom(10)}
            title="Minimum Zoom (10%)"
          >
            <ZoomOut size={13} />
          </button>

          <input
            type="range"
            min="0"
            max="100"
            step="0.2"
            value={sliderVal}
            onChange={(e) => {
              const s = parseFloat(e.target.value);
              const targetPct = Math.round(10 * Math.pow(640, s / 100));
              applyZoom(targetPct);
            }}
            style={{
              width: '100px',
              height: '4px',
              accentColor: 'var(--accent-primary)',
              cursor: 'pointer',
              background: 'rgba(255, 255, 255, 0.15)',
              borderRadius: '2px',
            }}
            title={`Zoom: ${zoomPct}% (Range: 10% - 6400%)`}
          />

          <button
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              padding: '2px',
            }}
            onClick={() => applyZoom(6400)}
            title="Maximum Zoom (6400%)"
          >
            <ZoomIn size={13} />
          </button>

          <span
            style={{
              fontSize: '0.75rem',
              fontVariantNumeric: 'tabular-nums',
              fontWeight: 600,
              color: 'var(--text-primary)',
              minWidth: '50px',
              textAlign: 'right',
              cursor: 'pointer',
            }}
            onClick={() => applyZoom(100)}
            title="Current Zoom (Click to reset to 100%)"
          >
            {zoomPct}%
          </span>
        </div>

        {isGenerating && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '8px 14px',
            borderRadius: '20px',
            background: 'var(--bg-panel)',
            backdropFilter: 'var(--glass-blur)',
            fontSize: '0.75rem',
            color: 'var(--accent-hover)',
            border: '1px solid var(--border-color)',
          }}>
            <Loader2 size={14} className="animate-spin" />
            <span>Kernel Synthesizing Solid...</span>
          </div>
        )}
      </div>

      <Canvas
        camera={{ position: [38, 32, 38], fov: 45, near: 0.05, far: 10000 }}
        style={{ touchAction: 'none' }}
      >
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
          fadeDistance={120}
          fadeStrength={1.2}
          infiniteGrid
        />

        <Suspense fallback={null}>
          {isGenerating ? (
            <LoadingMesh />
          ) : stlData ? (
            <STLModel
              base64Data={stlData}
              wireframe={wireframe}
              onCenterComputed={(y) => setTargetY(y)}
            />
          ) : null}
        </Suspense>

        <SceneController
          controlsRef={controlsRef}
          targetY={targetY}
          isPanMode={isPanMode}
          onZoomChange={(pct) => setZoomPct(pct)}
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
    </div>
  );
}
