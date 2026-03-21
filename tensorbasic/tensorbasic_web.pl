/*  tensorBASIC Web Server
 *
 *  A self-contained SWI-Prolog HTTP server that provides:
 *    1. Syntax-highlighted tensorBASIC source viewing
 *    2. Mermaid sequence diagrams (gosub swim lanes, loop detection)
 *    3. UMAP scatter plots of program structure (via umap-js + Plotly)
 *
 *  All HTML/CSS/JS is generated server-side — no build step required.
 *  External dependencies loaded from CDN: Mermaid.js, Plotly, UMAP-JS.
 *
 *  Start
 *  ─────
 *    $ cd tensorbasic
 *    $ swipl -g "start_server(8080)" tensorbasic_web.pl
 *
 *    Then open  http://localhost:8080/
 *
 *  Routes
 *  ──────
 *    GET  /                         File picker (list .bas files)
 *    GET  /view?file=<name>         Three-panel view for a file
 *    GET  /api/source?file=<name>   Raw source text
 *    GET  /api/highlight?file=<name> Syntax-highlighted HTML fragment
 *    GET  /api/mermaid?file=<name>  Mermaid diagram source text
 *    GET  /api/umap?file=<name>     JSON feature vectors for UMAP
 *    GET  /api/ast?file=<name>      Raw AST (Prolog term)
 *    GET  /api/files                JSON list of available .bas files
 */

:- module(tensorbasic_web, [
       start_server/1,
       stop_server/0
   ]).

:- use_module(library(http/thread_httpd)).
:- use_module(library(http/http_dispatch)).
:- use_module(library(http/http_parameters)).
:- use_module(library(http/http_header)).
:- use_module(library(lists)).
:- use_module(library(filesex)).

:- use_module(tensorbasic_dcg, [parse_tensorbasic/2]).
:- use_module(tensorbasic_mermaid, [ast_to_mermaid/2]).
:- use_module(tensorbasic_highlight, [highlight_source/2]).
:- use_module(tensorbasic_umap, [ast_to_umap_json/2]).

% ---------------------------------------------------------------------------
%  Server lifecycle
% ---------------------------------------------------------------------------

:- dynamic server_port/1.

%% start_server(+Port)
start_server(Port) :-
    (   server_port(_)
    ->  stop_server
    ;   true
    ),
    http_server(http_dispatch, [port(Port)]),
    assert(server_port(Port)),
    format("~n╔══════════════════════════════════════════╗~n", []),
    format("║  tensorBASIC web server on port ~w      ║~n", [Port]),
    format("║  http://localhost:~w/                   ║~n", [Port]),
    format("╚══════════════════════════════════════════╝~n~n", []).

stop_server :-
    (   server_port(Port)
    ->  http_stop_server(Port, []),
        retract(server_port(Port)),
        format("Server stopped.~n", [])
    ;   format("No server running.~n", [])
    ).

% ---------------------------------------------------------------------------
%  Route declarations
% ---------------------------------------------------------------------------

:- http_handler(root(.),           handle_index,     []).
:- http_handler(root(view),        handle_view,      []).
:- http_handler(root(api/files),   handle_api_files, []).
:- http_handler(root(api/source),  handle_api_source, []).
:- http_handler(root(api/highlight), handle_api_highlight, []).
:- http_handler(root(api/mermaid), handle_api_mermaid, []).
:- http_handler(root(api/umap),    handle_api_umap,   []).
:- http_handler(root(api/ast),     handle_api_ast,    []).

% ---------------------------------------------------------------------------
%  Handlers
% ---------------------------------------------------------------------------

%% GET / — file picker
handle_index(Request) :-
    list_bas_files(Files),
    maplist(file_link_html, Files, Links),
    atomic_list_concat(Links, '\n', LinkBlock),
    format(atom(Body),
        '<div class="file-picker">
           <h2>tensorBASIC Programs</h2>
           <p>Select a program to explore:</p>
           <div class="file-grid">~w</div>
         </div>', [LinkBlock]),
    page_html("tensorBASIC Explorer", "", Body, HTML),
    reply_html(HTML).

file_link_html(File, HTML) :-
    file_name_extension(Base, _, File),
    format(atom(HTML),
        '<a class="file-card" href="/view?file=~w">
           <div class="file-icon">&#9000;</div>
           <div class="file-name">~w</div>
         </a>', [File, Base]).

%% GET /view?file=<name> — three-panel view
handle_view(Request) :-
    http_parameters(Request, [file(File, [])]),
    validate_filename(File),
    bas_file_path(File, Path),
    read_file_to_string(Path, Source, []),
    %  Parse once, reuse AST for all three views
    parse_tensorbasic(Source, AST),
    %  1. Syntax-highlighted source
    highlight_source(Source, HighlightedHTML),
    %  2. Mermaid diagram
    ast_to_mermaid(AST, MermaidSrc),
    %  3. UMAP JSON
    ast_to_umap_json(AST, UmapJSON),
    %  Build page
    format(atom(ExtraHead),
        '<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
         <script src="https://cdn.jsdelivr.net/npm/umap-js@1.4.0/lib/umap-js.min.js"></script>
         <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>', []),
    view_body_html(File, HighlightedHTML, MermaidSrc, UmapJSON, Body),
    page_html(File, ExtraHead, Body, HTML),
    reply_html(HTML).

%% API: GET /api/files
handle_api_files(_Request) :-
    list_bas_files(Files),
    maplist(quote_json_string, Files, QFiles),
    atomic_list_concat(QFiles, ',', Arr),
    format(atom(JSON), '[~w]', [Arr]),
    reply_json_atom(JSON).

%% API: GET /api/source?file=<name>
handle_api_source(Request) :-
    http_parameters(Request, [file(File, [])]),
    validate_filename(File),
    bas_file_path(File, Path),
    read_file_to_string(Path, Source, []),
    format('Content-Type: text/plain; charset=utf-8\r\n\r\n'),
    write(Source).

%% API: GET /api/highlight?file=<name>
handle_api_highlight(Request) :-
    http_parameters(Request, [file(File, [])]),
    validate_filename(File),
    bas_file_path(File, Path),
    read_file_to_string(Path, Source, []),
    highlight_source(Source, HTML),
    format('Content-Type: text/html; charset=utf-8\r\n\r\n'),
    write(HTML).

%% API: GET /api/mermaid?file=<name>
handle_api_mermaid(Request) :-
    http_parameters(Request, [file(File, [])]),
    validate_filename(File),
    bas_file_path(File, Path),
    read_file_to_string(Path, Source, []),
    parse_tensorbasic(Source, AST),
    ast_to_mermaid(AST, Mermaid),
    format('Content-Type: text/plain; charset=utf-8\r\n\r\n'),
    write(Mermaid).

%% API: GET /api/umap?file=<name>
handle_api_umap(Request) :-
    http_parameters(Request, [file(File, [])]),
    validate_filename(File),
    bas_file_path(File, Path),
    read_file_to_string(Path, Source, []),
    parse_tensorbasic(Source, AST),
    ast_to_umap_json(AST, JSON),
    reply_json_atom(JSON).

%% API: GET /api/ast?file=<name>
handle_api_ast(Request) :-
    http_parameters(Request, [file(File, [])]),
    validate_filename(File),
    bas_file_path(File, Path),
    read_file_to_string(Path, Source, []),
    parse_tensorbasic(Source, AST),
    format('Content-Type: text/plain; charset=utf-8\r\n\r\n'),
    print_term(AST, [output(current_output)]).

% ---------------------------------------------------------------------------
%  File management
% ---------------------------------------------------------------------------

%% list_bas_files(-Files)
%  List all .bas files in the examples directory.
list_bas_files(Files) :-
    examples_dir(Dir),
    directory_files(Dir, All),
    include(is_bas_file, All, Files0),
    sort(Files0, Files).

is_bas_file(F) :-
    file_name_extension(_, bas, F).

examples_dir(Dir) :-
    source_file(tensorbasic_web:start_server(_), ThisFile),
    file_directory_name(ThisFile, ModDir),
    directory_file_path(ModDir, examples, Dir).

bas_file_path(File, Path) :-
    examples_dir(Dir),
    directory_file_path(Dir, File, Path).

%% validate_filename(+File)
%  Security: reject path traversal attempts.
validate_filename(File) :-
    \+ sub_atom(File, _, _, _, '..'),
    \+ sub_atom(File, _, _, _, '/'),
    \+ sub_atom(File, _, _, _, '\\'),
    file_name_extension(_, bas, File).

% ---------------------------------------------------------------------------
%  HTML templates
% ---------------------------------------------------------------------------

%% page_html(+Title, +ExtraHead, +Body, -HTML)
page_html(Title, ExtraHead, Body, HTML) :-
    css_styles(CSS),
    format(atom(HTML),
'<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>~w — tensorBASIC</title>
  <style>~w</style>
  ~w
</head>
<body>
  <header>
    <a href="/" class="logo">tensor<span class="logo-basic">BASIC</span></a>
    <span class="subtitle">Explorer</span>
  </header>
  <main>~w</main>
</body>
</html>', [Title, CSS, ExtraHead, Body]).

%% view_body_html(+File, +HL_HTML, +MermaidSrc, +UmapJSON, -Body)
view_body_html(File, HL_HTML, MermaidSrc, UmapJSON, Body) :-
    mermaid_html_escape(MermaidSrc, MermaidEsc),
    format(atom(Body),
'<div class="toolbar">
  <h2>~w</h2>
  <div class="tab-bar">
    <button class="tab active" onclick="showTab(\'source\')">Source</button>
    <button class="tab" onclick="showTab(\'diagram\')">Sequence Diagram</button>
    <button class="tab" onclick="showTab(\'umap\')">UMAP Plot</button>
  </div>
</div>

<div id="source" class="panel active">
  <div class="source-container">
    <pre class="source-code"><code>~w</code></pre>
  </div>
</div>

<div id="diagram" class="panel">
  <div class="diagram-container">
    <pre class="mermaid">~w</pre>
  </div>
</div>

<div id="umap" class="panel">
  <div class="umap-controls">
    <label>n_neighbors: <input type="range" id="n-neighbors" min="2" max="50" value="15">
      <span id="nn-val">15</span></label>
    <label>min_dist: <input type="range" id="min-dist" min="0" max="100" value="10" step="1">
      <span id="md-val">0.1</span></label>
    <button onclick="runUMAP()">Re-run UMAP</button>
    <label><input type="checkbox" id="color-region" checked> Color by region</label>
  </div>
  <div id="umap-plot" style="width:100%;height:calc(100vh - 220px);"></div>
</div>

<script>
// --- Tab switching ---
function showTab(id) {
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.getElementById(id).classList.add("active");
  event.target.classList.add("active");
  if (id === "diagram") mermaid.run();
  if (id === "umap" && !window._umapDone) { runUMAP(); window._umapDone = true; }
}

// --- Mermaid init ---
mermaid.initialize({ startOnLoad: false, theme: "dark", securityLevel: "loose" });

// --- UMAP ---
const umapData = ~w;

document.getElementById("n-neighbors").oninput = function() {
  document.getElementById("nn-val").textContent = this.value;
};
document.getElementById("min-dist").oninput = function() {
  document.getElementById("md-val").textContent = (this.value / 100).toFixed(2);
};

async function runUMAP() {
  if (!umapData || umapData.length < 3) {
    document.getElementById("umap-plot").innerHTML = "<p>Not enough data points for UMAP.</p>";
    return;
  }

  const nNeighbors = parseInt(document.getElementById("n-neighbors").value);
  const minDist = parseInt(document.getElementById("min-dist").value) / 100;

  const features = umapData.map(d => d.features);

  // Use UMAP-JS
  const umap = new UMAP.UMAP({
    nNeighbors: Math.min(nNeighbors, features.length - 1),
    minDist: minDist,
    nComponents: 2,
    spread: 1.0
  });

  const embedding = await umap.fitAsync(features);

  const colorByRegion = document.getElementById("color-region").checked;

  // Group by category or region
  const groupKey = colorByRegion ? "region" : "category";
  const groups = {};
  umapData.forEach((d, i) => {
    const key = d[groupKey] || "other";
    if (!groups[key]) groups[key] = { x: [], y: [], text: [], name: key };
    groups[key].x.push(embedding[i][0]);
    groups[key].y.push(embedding[i][1]);
    groups[key].text.push("L" + d.line + ": " + d.label);
  });

  const traces = Object.values(groups).map(g => ({
    x: g.x, y: g.y, text: g.text, name: g.name,
    mode: "markers",
    type: "scatter",
    marker: { size: 8, opacity: 0.8 },
    hoverinfo: "text+name"
  }));

  const layout = {
    title: "UMAP — Program Structure",
    paper_bgcolor: "#1a1a2e",
    plot_bgcolor: "#16213e",
    font: { color: "#e0e0e0" },
    xaxis: { showgrid: false, zeroline: false, title: "" },
    yaxis: { showgrid: false, zeroline: false, title: "" },
    legend: { orientation: "h", y: -0.15 },
    margin: { l: 40, r: 20, t: 50, b: 60 }
  };

  Plotly.newPlot("umap-plot", traces, layout, { responsive: true });
}
</script>', [File, HL_HTML, MermaidEsc, UmapJSON]).

%% Escape Mermaid source for embedding in HTML <pre> tags.
mermaid_html_escape(In, Out) :-
    atom_string(In, S),
    string_codes(S, Codes),
    maplist(mermaid_esc_code, Codes, EscLists),
    append(EscLists, AllCodes),
    atom_codes(Out, AllCodes).

mermaid_esc_code(0'<, "&lt;") :- !.
mermaid_esc_code(0'>, "&gt;") :- !.
mermaid_esc_code(0'&, "&amp;") :- !.
mermaid_esc_code(C, [C]).

% ---------------------------------------------------------------------------
%  CSS
% ---------------------------------------------------------------------------

css_styles(CSS) :-
    CSS = '
:root {
  --bg: #0f0f23;
  --bg2: #1a1a2e;
  --bg3: #16213e;
  --fg: #e0e0e0;
  --accent: #00d4ff;
  --accent2: #7b68ee;
  --accent3: #50fa7b;
  --border: #2a2a4a;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", monospace;
  background: var(--bg);
  color: var(--fg);
  min-height: 100vh;
}

header {
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 12px 24px;
  display: flex;
  align-items: baseline;
  gap: 16px;
}

.logo {
  font-size: 1.4em;
  font-weight: bold;
  color: var(--accent);
  text-decoration: none;
}
.logo-basic { color: var(--accent3); }
.subtitle { color: #888; font-size: 0.9em; }

main { padding: 16px 24px; }

/* File picker */
.file-picker h2 { margin-bottom: 8px; color: var(--accent); }
.file-picker p { color: #888; margin-bottom: 16px; }
.file-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px;
}
.file-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px 16px;
  text-align: center;
  text-decoration: none;
  color: var(--fg);
  transition: border-color 0.2s, transform 0.15s;
}
.file-card:hover {
  border-color: var(--accent);
  transform: translateY(-2px);
}
.file-icon { font-size: 2em; margin-bottom: 8px; }
.file-name { font-weight: 600; }

/* Toolbar + tabs */
.toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}
.toolbar h2 { color: var(--accent); font-size: 1.1em; }
.tab-bar { display: flex; gap: 4px; }
.tab {
  background: var(--bg2);
  color: #888;
  border: 1px solid var(--border);
  border-radius: 6px 6px 0 0;
  padding: 8px 20px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.9em;
  transition: all 0.15s;
}
.tab:hover { color: var(--fg); }
.tab.active {
  background: var(--bg3);
  color: var(--accent);
  border-bottom-color: var(--bg3);
}

/* Panels */
.panel { display: none; }
.panel.active { display: block; }

/* Source code */
.source-container {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  overflow-x: auto;
  max-height: calc(100vh - 160px);
  overflow-y: auto;
}
.source-code {
  font-size: 0.85em;
  line-height: 1.7;
  tab-size: 4;
}

/* Syntax highlighting */
.hl-lineno  { color: #555; margin-right: 1em; user-select: none; }
.hl-keyword { color: #ff79c6; font-weight: bold; }
.hl-comment { color: #6272a4; font-style: italic; }
.hl-string  { color: #f1fa8c; }
.hl-number  { color: #bd93f9; }
.hl-builtin { color: #50fa7b; }
.hl-ident   { color: #8be9fd; }
.hl-op      { color: #ffb86c; }

/* Diagram */
.diagram-container {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 24px;
  overflow: auto;
  max-height: calc(100vh - 160px);
  text-align: center;
}
.diagram-container .mermaid { text-align: left; display: inline-block; }

/* UMAP controls */
.umap-controls {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 20px;
  flex-wrap: wrap;
}
.umap-controls label { color: #888; font-size: 0.85em; display: flex; align-items: center; gap: 6px; }
.umap-controls input[type=range] { width: 100px; }
.umap-controls button {
  background: var(--accent);
  color: var(--bg);
  border: none;
  border-radius: 4px;
  padding: 6px 16px;
  cursor: pointer;
  font-family: inherit;
  font-weight: bold;
}
.umap-controls button:hover { opacity: 0.85; }
'.

% ---------------------------------------------------------------------------
%  Response helpers
% ---------------------------------------------------------------------------

reply_html(HTML) :-
    format('Content-Type: text/html; charset=utf-8\r\n\r\n'),
    write(HTML).

reply_json_atom(JSON) :-
    format('Content-Type: application/json; charset=utf-8\r\n\r\n'),
    write(JSON).

quote_json_string(S, Q) :-
    format(atom(Q), '"~w"', [S]).
