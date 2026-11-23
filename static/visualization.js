import { composeLayout } from './visual_layout.js';

const observers = new WeakMap();
const CHART_KINDS = new Set(['bar', 'stacked_bar', 'line', 'area', 'pie', 'donut', 'scatter']);
const DIAGRAM_KINDS = new Set(['flowchart', 'sequence', 'class', 'er', 'mindmap', 'architecture']);

function stop(entry) {
    entry.run?.abort();
    entry.run = null;
    entry.dispose?.();
    entry.dispose = null;
}

function buildDataTable(visual) {
    const details = document.createElement('details');
    details.className = 'message-visualization-data';
    const summary = document.createElement('summary');
    summary.textContent = 'Dữ liệu biểu đồ';
    details.append(summary);
    details.addEventListener('toggle', () => {
        if (!details.open || details.children.length > 1) return;
        const scroll = document.createElement('div');
        const table = document.createElement('table');
        table.setAttribute('aria-label', visual.title);
        const headings = table.createTHead().insertRow();
        for (const heading of [visual.x_name, ...visual.series.map(series => series.name)]) {
            const cell = document.createElement('th');
            cell.scope = 'col';
            cell.textContent = heading;
            headings.append(cell);
        }
        const body = table.createTBody();
        visual.x.forEach((value, index) => {
            const row = body.insertRow();
            for (const item of [value, ...visual.series.map(series => series.values[index])]) {
                row.insertCell().textContent = item;
            }
        });
        scroll.append(table);
        details.append(scroll);
    });
    return details;
}

function visualSurface(entry) {
    const { visual, surface } = entry;
    if (!visual.layout) return surface;
    if (!entry.plot) {
        const plot = document.createElement('div');
        plot.className = `message-visualization-surface ${visual.engine}`;
        const layout = composeLayout(visual.layout, block => {
            if (block.type !== 'table') return plot;
            const table = buildDataTable(visual);
            if (block.text) table.querySelector('summary').textContent = block.text;
            return table;
        });
        surface.replaceChildren(layout);
        entry.plot = plot;
    }
    return entry.plot;
}

async function show(entry, onSettled) {
    if (entry.run || entry.dispose || !entry.figure.isConnected) return;
    const run = new AbortController();
    entry.run = run;
    const { visual, surface } = entry;
    try {
        if (visual.engine === 'echarts') {
            const { renderChart } = await import('./visualizations/chart.js?v=2');
            if (!run.signal.aborted) {
                const plot = visualSurface(entry);
                entry.dispose = renderChart(plot, visual);
                plot.setAttribute('role', 'img');
                plot.setAttribute('aria-label', visual.title);
            }
        } else if (visual.engine === 'mermaid') {
            const { renderDiagram } = await import('./visualizations/diagram.js?v=2');
            if (!run.signal.aborted) await renderDiagram(visualSurface(entry), visual, run.signal);
            if (!run.signal.aborted) onSettled();
        } else {
            const { renderSimulator } = await import('./simulator.js?v=6');
            if (!run.signal.aborted) {
                renderSimulator(surface, visual);
                onSettled();
            }
        }
    } catch {
        if (!run.signal.aborted) {
            stop(entry);
            surface.removeAttribute('role');
            surface.removeAttribute('aria-label');
            surface.classList.add('message-visualization-error');
            surface.textContent = 'Không thể hiển thị hình minh họa.';
            onSettled();
        }
    } finally {
        if (entry.run === run) entry.run = null;
    }
}

export function observeVisualization(figure, surface, visual, root) {
    const chart = visual.engine === 'echarts' && CHART_KINDS.has(visual.kind);
    const diagram = visual.engine === 'mermaid' && DIAGRAM_KINDS.has(visual.kind);
    const simulator = visual.engine === 'simulator' && visual.kind === 'calculator';
    if (!chart && !diagram && !simulator) return;
    if (chart && !visual.layout) figure.append(buildDataTable(visual));

    let tracker = observers.get(root);
    if (!tracker) {
        const entries = new Map();
        const observer = new IntersectionObserver(changes => {
            for (const change of changes) {
                const entry = entries.get(change.target);
                if (!entry) continue;
                if (change.isIntersecting) {
                    void show(entry, () => {
                        observer.unobserve(change.target);
                        entries.delete(change.target);
                    });
                } else {
                    stop(entry);
                }
            }
        }, { root, rootMargin: '160px 0px' });
        tracker = { observer, entries };
        observers.set(root, tracker);
    }
    tracker.entries.set(surface, { figure, surface, visual });
    tracker.observer.observe(surface);
}

export function releaseVisualizations(root) {
    const tracker = observers.get(root);
    if (!tracker) return;
    tracker.observer.disconnect();
    tracker.entries.forEach(stop);
    observers.delete(root);
}

