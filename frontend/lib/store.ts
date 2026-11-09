import { create } from 'zustand';

interface EditorState {
  csgTree: any[];
  stlData: string | null;
  cadqueryCode: string | null;
  attachedImage: File | null;
  attachedImagePreview: string | null;
  isGenerating: boolean;
  generationTime: number | null;
  wireframe: boolean;
  theme: 'light' | 'dim' | 'dark';
  isPromptCollapsed: boolean;

  setCsgTree: (tree: any[]) => void;
  setStlData: (base64Str: string | null) => void;
  setCadqueryCode: (code: string | null) => void;
  setAttachedImage: (file: File | null, previewUrl?: string | null) => void;
  setIsGenerating: (status: boolean) => void;
  setGenerationTime: (time: number | null) => void;
  setWireframe: (wf: boolean) => void;
  setTheme: (theme: 'light' | 'dim' | 'dark') => void;
  setIsPromptCollapsed: (status: boolean) => void;
}

export const useEditorStore = create<EditorState>((set) => ({
  csgTree: [],
  stlData: null,
  cadqueryCode: null,
  attachedImage: null,
  attachedImagePreview: null,
  isGenerating: false,
  generationTime: null,
  wireframe: false,
  theme: 'dark',
  isPromptCollapsed: false,

  setCsgTree: (tree) => set({ csgTree: tree }),
  setStlData: (base64Str) => set({ stlData: base64Str }),
  setCadqueryCode: (code) => set({ cadqueryCode: code }),
  setAttachedImage: (file, previewUrl = null) => set({ attachedImage: file, attachedImagePreview: previewUrl }),
  setIsGenerating: (status) => set({ isGenerating: status }),
  setGenerationTime: (time) => set({ generationTime: time }),
  setWireframe: (wf) => set({ wireframe: wf }),
  setTheme: (theme) => {
    if (typeof document !== 'undefined') {
      document.documentElement.setAttribute('data-theme', theme);
    }
    set({ theme });
  },
  setIsPromptCollapsed: (status) => set({ isPromptCollapsed: status }),
}));
