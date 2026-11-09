**PartMode** (the free, open-source, browser-based parametric CAD tool announced and demoed in the @protosphinx post) was built rapidly from scratch by Sphinx (@protosphinx) over roughly one month in mid-2026.



**Origins and motivation**

Sphinx (founder of Deskera, with a maths background from IITK) was already developing open tools around manufacturing and BOMs, including BOMwiki. In mid-July 2026 he added browser-based CAD capabilities to BOMwiki so people could design real parts in the browser and contribute to a library of open CAD designs. This evolved into a standalone product, PartMode (partmode.com), designed from the outset as both a human-usable SolidWorks-style tool **and** a native platform for AI agents.



The goal was a free alternative that supports headless/API-driven operation (no need for UI interaction or screen scraping), exact geometry, and agent workflows for automated design, iteration, and integration with hardware pipelines.



**How the technical foundation was assembled**

Rather than writing a geometry kernel from zero (which is extremely hard), the project composes mature open-source components and adds a custom application layer:



\- Geometry kernel: OpenCascade Technology (OCCT), a long-established industrial B-rep (boundary representation) solid modelling kernel, compiled to WebAssembly (WASM). This enables exact, high-fidelity solid modelling that runs entirely in the browser.

\- Modelling API: Replicad (a JavaScript library) provides a convenient, fluent interface on top of the OCCT WASM runtime for common operations (sketches, extrude, revolve, cut, fillet, patterns, etc.).

\- Visualisation: Three.js for interactive 3D rendering in the browser.

\- Application / UX layer (custom TypeScript/JavaScript): Implements a SolidWorks-style parametric workflow — constrained 2D sketches, editable feature history/tree with rebuild, multi-body parts, structured assemblies with mates, parameters/equations, configurations, inspection/mass properties, production drawings (leveraging OCCT hidden-line removal), and standard exchange formats (STEP import/export, STL, AMF, 3MF, SVG, DXF, PDF). Projects are local-first (stored in the browser).

\- Agent-native design: Built in from the beginning. There is a dual interface that shares the same canonical document model and exact kernel:

&#x20; - Human GUI.

&#x20; - Typed operations for agents (via MCP tools, skills, preview-then-commit with revision checks, permissions, and optional headless/server sessions). Agents issue precise CAD commands and receive validated geometry/evidence rather than relying on UI automation.



Architecture (simplified): Human UI or agent client → canonical project document (schema-5) → worker process running replicad + OCCT WASM → exact B-rep evidence, validation, and exports. Browser-approved agent sessions require explicit user approval; headless mode is a separate, deliberate option.



**Timeline and release**

\- Mid-July 2026: Early browser CAD integration into BOMwiki.

\- \~1 month of intensive development (Sphinx noted the released version was “just a month old (from scratch)”).

\- 1 August 2026: Public announcement and demo video (the one in the original post), showing the template library, parametric editing of parts (plates, flanges, etc.), feature history, and UI.

\- 7 August 2026: Version 0.1 open-sourced under the GNU Affero General Public License v3.0 on GitHub (BOMWiki/partmode). Later iterations continued (UI references V8).



The demo video simply records the running browser application performing typical parametric modelling tasks. The system validates results and rejects broken geometry where possible rather than silently producing invalid models.



In short, it was enabled by modern web technology (WASM for a real industrial kernel) plus a deliberate focus on agent APIs and parametric history from day one, allowing a capable free browser CAD to appear unusually quickly compared with traditional CAD development. The source, architecture notes, and site are public for anyone who wants to inspect or contribute further.



**Here are the direct links for PartMode:**



\- Main site: https://partmode.com/

(Browser-based CAD app)



\- Source code (GitHub repository): https://github.com/BOMWiki/partmode

Open-source under the GNU Affero General Public License v3.0 (AGPL-3.0). Includes the full application source, tests, and documentation.



\- Architectural notes: https://github.com/BOMWiki/partmode/blob/main/docs/architecture.md

Explains the dual human/agent interfaces, canonical document model, kernel worker (OpenCascade WASM + replicad), and related boundaries.



Additional useful pages on the site:

\- About: https://partmode.com/about

\- Help / documentation (including agent/MCP details): https://partmode.com/help



You can clone the repo and run it locally with Node.js (see the README for `npm ci`, `npm run build`, `npm start`).



\*\*Distinction and similarity between MecAgent and PartMode\*\*



\### Core similarity

Both are responses to the same underlying shift: AI agents are becoming capable of participating in mechanical design, but traditional CAD tools were built for human mouse-and-menu interaction.  



\- Both aim to produce \*\*real, parametric, editable CAD\*\* (feature history, constraints, rebuildable models) rather than just meshes or static geometry.  

\- Both treat agents as first-class users.  

\- Both sit in the Drawing-to-CAD / Text-to-CAD / agentic modelling space that has become active in 2025–2026.  

\- The original X post that started this thread (MecAgent’s Fable 5 + Drawing2CAD demo in SolidWorks) is exactly the kind of capability PartMode was positioned against (“you don’t need SolidWorks…”).



\### Key distinctions



| Aspect              | MecAgent                                      | PartMode                                              |

|---------------------|-----------------------------------------------|-------------------------------------------------------|

| \*\*Architecture\*\*    | AI agent / copilot that \*drives\* existing commercial CAD (primarily SolidWorks, also Inventor; more planned) via macros, APIs, and automation | Standalone, browser-based parametric CAD system built from scratch |

| \*\*Geometry kernel\*\* | Relies on the host CAD software’s kernel (Parasolid in SolidWorks, etc.) | OpenCascade (OCCT) compiled to WebAssembly + Replicad |

| \*\*Business model\*\*  | Commercial product (copilot, credits, enterprise features) | Free \& open-source (AGPL-3.0) |

| \*\*Access model\*\*    | Requires a SolidWorks (or similar) licence + MecAgent installation | Free in any modern browser; local-first; no account needed for core modelling |

| \*\*Agent interface\*\* | Generates macros / drives the UI and API of the closed host software | Native typed operations, MCP tools, headless mode, preview-and-commit with revision control |

| \*\*Primary strength\*\*| Deep integration with industry-standard tools engineers already use; production-ready feature trees inside SolidWorks | Open, inspectable, agent-native stack that anyone can run, modify, or self-host; no vendor lock-in |

| \*\*Primary limitation\*\* | Tied to proprietary CAD licences and the constraints of the host software’s automation surface | Still maturing (feature coverage, large-assembly performance, drawing standards, etc. are progressing rapidly but not yet at SolidWorks breadth) |



In short:  

\*\*MecAgent\*\* makes existing enterprise CAD tools agentic.  

\*\*PartMode\*\* tries to make a fully open, agent-native CAD system that does not require those tools.



They are complementary more than direct substitutes. An engineer might use MecAgent inside SolidWorks for production work and PartMode for open collaboration, education, rapid agent experimentation, or environments where SolidWorks licences are unavailable.



\### Benchmark metrics (VSR and IoU) and how to use them appropriately



These metrics come from the \*\*generative CAD / code-generation\*\* evaluation tradition (image-to-CadQuery, drawing-to-code, text-to-macro, etc.).



\- \*\*VSR – Valid Syntax Rate\*\*  

&#x20; Percentage of generated programs (CadQuery scripts, macros, etc.) that execute without crashing.  

&#x20; - Strength: Cheap, objective filter for basic usability.  

&#x20; - Limitation: A program can be syntactically valid and still produce the wrong shape, an uneditable “dumb” solid, or a feature tree that breaks on parameter change.



\- \*\*IoU / Best IoU\*\*  

&#x20; Intersection-over-Union of the volumes (or voxelised/meshed representations) of the predicted solid versus the ground-truth solid.  

&#x20; - Strength: Direct geometric fidelity measure.  

&#x20; - Limitations:  

&#x20;   - Sensitive to alignment, meshing density, and numerical tolerance.  

&#x20;   - Ignores parametric intent, feature history quality, constraints, tolerances, and manufacturing semantics.  

&#x20;   - Two models can have high IoU yet be completely different from an engineering perspective (e.g., one is a clean extrude + fillet feature tree; the other is a Boolean mess).



\#### Appropriate utilisation



1\. \*\*Use them as necessary but not sufficient filters\*\*  

&#x20;  High VSR + high IoU is a useful minimum bar for any generative system. Low scores reliably indicate failure. High scores do \*not\* guarantee a useful engineering model.



2\. \*\*Layer additional, more CAD-specific metrics\*\*  

&#x20;  Modern benchmarks (CADGenBench and others) already move in this direction:

&#x20;  - Geometric validity / watertightness / manifold checks (hard gate).  

&#x20;  - Shape similarity (surface F1, volume IoU, Chamfer, etc.).  

&#x20;  - Downstream usability: does the model rebuild correctly under parameter variation?  

&#x20;  - Feature-tree or construction-history similarity (harder, but more meaningful for parametric systems).  

&#x20;  - Assembly-level checks (mates, interference, kinematic degrees of freedom).  

&#x20;  - Export fidelity (STEP round-trip).



3\. \*\*Match the metric to the system type\*\*  

&#x20;  - Pure code-generation models (image → CadQuery) → VSR + IoU remain highly relevant.  

&#x20;  - Agent-driven systems that operate inside a real CAD kernel (MecAgent inside SolidWorks, or agents calling PartMode’s typed API) → prioritise rebuild success, topological stability, editability, and constraint satisfaction. Geometry IoU is still useful as a secondary check.  

&#x20;  - Full CAD applications like PartMode → regression suites on document model, topology naming, B-rep validity, and agent command reliability matter more than one-shot generative IoU.



4\. \*\*Practical recommendation\*\*  

&#x20;  Report \*\*VSR + IoU\*\* for comparability with the existing literature, but always accompany them with:

&#x20;  - A validity gate (does it produce a usable solid at all?).  

&#x20;  - At least one editability / parametric test (change a key dimension and re-evaluate).  

&#x20;  - Qualitative inspection of the feature tree when the system claims to be parametric.



In the current landscape, MecAgent is optimising for high real-world usefulness inside the tools engineers already pay for, while PartMode is optimising for openness and native agent control. The right benchmark suite for each therefore overlaps on geometry and validity but diverges on what “success” means beyond the solid.



\*\*CADGenBench Evaluation Details\*\*



CADGenBench (Hugging Face / collaboration involving Thomas Wolf, Michael Rabinovich, and others; related to Mecado) is a tool-agnostic benchmark for AI systems that generate or edit functional 3D mechanical parts. Submissions are simply STEP/BREP files (`output.step` or `.stp`). The system does not care whether you used build123d, CadQuery, SolidWorks, Fusion, Onshape, or a custom kernel.



\### Tasks

\- \*\*Generation\*\*: Given an engineering drawing (and description), produce a valid, geometrically correct 3D model.

\- \*\*Editing\*\*: Given an existing STEP file + a natural-language change request, apply that change correctly.



Ground truth is held private. Public inputs are available on the Hub; scoring happens server-side on the leaderboard Space.



\### Scoring Pipeline (per sample)



1\. \*\*Validity Gate (hard)\*\*  

&#x20;  The candidate must be a well-formed, watertight B-rep that meshes into a closed, orientable manifold.  

&#x20;  Failure → \*\*CAD Score = 0\*\* (no partial credit). This prevents invalid or non-manifold solids from ranking highly.



2\. \*\*Rigid Alignment\*\*  

&#x20;  Candidate is rigidly aligned to the ground truth (rotation + translation only; no scaling) using a robust multi-scale ICP procedure over a pool of candidate poses (including PCA orientations). This makes scoring pose-invariant within reasonable limits.



3\. \*\*Three Orthogonal Metrics\*\* (each in \[0, 1])  

&#x20;  These are deliberately designed to catch different classes of error:



&#x20;  | Metric              | What it measures                                      | Catches                                      | Blind to                          |

&#x20;  |---------------------|-------------------------------------------------------|----------------------------------------------|-----------------------------------|

&#x20;  | \*\*Shape Similarity\*\* | Mean of Surface Distance F1 + Volume IoU             | Wrong overall geometry / occupied volume    | Topology counts, exact feature placement |

&#x20;  | \*\*Topology Match\*\*   | Agreement on Betti numbers (b₀, b₁, b₂) of the boundary | Wrong number of pieces, through-holes, or voids | Feature positions, bulk shape    |

&#x20;  | \*\*Interface Match\*\*  | Mating-feature correctness via authored keep-in / keep-out sub-volumes | Misplaced, wrong-sized, or missing bolt patterns / mating faces | Overall shape, pure topology     |



&#x20;  - \*\*b₀\*\* = number of connected components  

&#x20;  - \*\*b₁\*\* = number of through-holes / handles  

&#x20;  - \*\*b₂\*\* = number of enclosed cavities / voids  



4\. \*\*Final CAD Score\*\*

&#x20;  - \*\*Generation tasks\*\*:  

&#x20;    `0.4 × Shape Similarity + 0.4 × Interface Match + 0.2 × Topology Match`  

&#x20;    (if valid; else 0)



&#x20;  - \*\*Editing tasks\*\* (special handling):  

&#x20;    Shape is \*renormalized\* against the no-op baseline (score of the unchanged input STEP vs GT). This prevents “do nothing” from scoring highly.  

&#x20;    Weights become shape-dominant:  

&#x20;    `0.6 × renormalized\_shape + 0.3 × Interface + 0.1 × Topology`



\### Design Philosophy

The three main metrics are \*\*orthogonal by construction\*\*. Pure volume IoU or Chamfer distance alone would miss many engineering failures (extra holes, missing voids, bolt patterns shifted by a few millimetres, non-manifold geometry, etc.). Validity is a hard gate rather than a soft score so that invalid solids never outrank valid but imperfect ones.



\### Practical Notes

\- Leaderboard entries start as unvalidated; promotion to the validated tier requires methodology review.

\- A reference baseline (iterative agent using build123d or CadQuery) is provided so participants can see an end-to-end example.

\- Current top validated scores (as of recent public discussion) remain well below 1.0 (examples around 0.53–0.64), indicating the benchmark is still challenging, especially on interface and precise topology.



This is a clear step beyond simple VSR + IoU used in earlier code-generation benchmarks: it adds engineering-relevant checks (mating interfaces, topological genus-like properties, and a strict validity gate) while remaining tool-agnostic via STEP.



Official sources:  

\- GitHub: \[huggingface/cadgenbench](https://github.com/huggingface/cadgenbench)  

\- Metrics overview: `docs/metrics.md` (and the per-axis deep-dives in `docs/metrics/`)  

\- Leaderboard Space: HuggingAI4Engineering/CADGenBench

