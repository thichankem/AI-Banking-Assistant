import mermaid from 'mermaid';

mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',
    htmlLabels: false,
    suppressErrorRendering: true,
    theme: 'dark',
    maxTextSize: 3050,
    maxEdges: 64,
});

let nextDiagramId = 0;
const themes = new Set(['default', 'neutral', 'forest']);

export async function renderDiagram(element, visual, signal) {
    const source = themes.has(visual.theme) ? `---\nconfig:\n  theme: ${visual.theme}\n---\n${visual.source}` : visual.source;
    const { svg } = await mermaid.render(`abank-diagram-${++nextDiagramId}`, source);
    if (signal.aborted) return;

    // SVG in an image has no active scripts, links, or external resource loads.
    const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
    const image = new Image();
    image.alt = visual.title;
    image.src = url;
    try {
        await image.decode();
        if (!signal.aborted) element.replaceChildren(image);
    } finally {
        URL.revokeObjectURL(url);
    }
}
