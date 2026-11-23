import { init, use } from 'echarts/core';
import { BarChart, LineChart, PieChart, ScatterChart } from 'echarts/charts';
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

use([BarChart, LineChart, PieChart, ScatterChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer]);

export function renderChart(element, visual) {
    const config = visual.config || {};
    const css = getComputedStyle(element);
    const foreground = css.getPropertyValue('--text-primary').trim();
    const muted = css.getPropertyValue('--text-muted').trim();
    const splitLine = { lineStyle: { color: css.getPropertyValue('--border-default').trim() } };
    const number = new Intl.NumberFormat('vi-VN', { maximumSignificantDigits: 17 });
    const pie = visual.kind === 'pie' || visual.kind === 'donut';
    const scatter = visual.kind === 'scatter';
    const stacked = visual.kind === 'stacked_bar';
    const line = visual.kind === 'line' || visual.kind === 'area';
    const multiple = visual.series.length > 1;
    const legend = config.legend ?? multiple;
    const options = {
        animation: false,
        color: config.colors?.length ? config.colors : ['#6698f5', '#42beab', '#f7ad59', '#cd8ef1', '#ed7d91', '#a5bc69'],
        textStyle: { color: foreground },
        tooltip: {
            trigger: pie || scatter ? 'item' : 'axis',
            renderMode: 'richText', confine: true, transitionDuration: 0,
            valueFormatter: value => Array.isArray(value) ? value.map(item => number.format(item)).join(' · ') : number.format(value),
        },
    };
    if (legend) {
        options.legend = { type: 'scroll', top: 0, textStyle: { color: foreground, width: 130, overflow: 'truncate' } };
    }
    if (pie) {
        options.series = [{
            type: 'pie', name: visual.series[0].name,
            radius: visual.kind === 'donut' ? [`${config.donut_hole ?? 38}%`, '68%'] : '68%',
            stillShowZeroSum: false,
            label: { color: foreground, width: 95, overflow: 'truncate' },
            data: visual.x.map((name, index) => ({ name, value: visual.series[0].values[index] })),
        }];
    } else {
        const xLabels = { color: muted, hideOverlap: true };
        options.grid = { left: 14, right: 18, top: legend ? 48 : 34, bottom: 44, outerBoundsMode: 'same' };
        options.xAxis = {
            type: scatter ? 'value' : 'category',
            data: scatter ? undefined : visual.x,
            name: visual.x_name, nameLocation: 'middle', nameGap: 29,
            nameTextStyle: { color: foreground, width: 200, overflow: 'truncate' },
            axisLabel: { ...xLabels, width: 86, overflow: 'truncate' },
            splitLine: scatter ? splitLine : undefined,
        };
        options.yAxis = {
            type: 'value', name: config.y_name ?? (multiple ? '' : visual.series[0].name),
            nameTextStyle: { color: foreground, align: 'left', width: 205, overflow: 'truncate' },
            axisLabel: { color: muted }, splitLine,
        };
        options.series = visual.series.map((series, index) => ({
            name: series.name,
            type: scatter ? 'scatter' : line ? 'line' : 'bar',
            data: scatter ? visual.x.map((value, i) => [value, series.values[i]]) : series.values,
            stack: stacked ? 'total' : undefined,
            symbol: ['circle', 'diamond', 'triangle', 'rect'][index],
            showSymbol: config.show_points ?? visual.x.length <= 25,
            smooth: line && config.smooth,
            lineStyle: index % 2 ? { width: 2, type: 'dashed' } : { width: 2 },
            areaStyle: visual.kind === 'area' ? { opacity: config.area_opacity ?? 0.16 } : undefined,
            barMaxWidth: 48,
        }));
    }

    const chart = init(element, undefined, { renderer: 'canvas', devicePixelRatio: Math.min(window.devicePixelRatio || 1, 2) });
    const resize = new ResizeObserver(() => {
        if (element.clientWidth && element.clientHeight) chart.resize();
    });
    try {
        chart.setOption(options);
        resize.observe(element);
    } catch (error) {
        chart.dispose();
        throw error;
    }
    return () => {
        resize.disconnect();
        chart.dispose();
    };
}
