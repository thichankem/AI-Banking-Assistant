import { composeLayout } from './visual_layout.js';

const MAX_MAGNITUDE = 100_000_000_000_000;
const numberFormats = new Map();
let nextControlId = 0;

function loanPeriods(principal, rate, periods) {
    const count = Math.round(periods);
    if (principal < 0 || rate < 0 || rate > 1 || count < 1 || count > 1200 || Math.abs(periods - count) > 1e-9) {
        return NaN;
    }
    return count;
}

function payment(principal, rate, periods) {
    const count = loanPeriods(principal, rate, periods);
    if (rate === 0) return principal / count;
    return principal * (rate / -Math.expm1(-count * Math.log1p(rate)));
}

const operations = new Map([
    ['add', { calculate: (...values) => values.reduce((a, b) => a + b), symbol: '+' }],
    ['subtract', { calculate: (a, b) => a - b, symbol: '−' }],
    ['multiply', { calculate: (...values) => values.reduce((a, b) => a * b), symbol: '×' }],
    ['divide', { calculate: (a, b) => a / b, symbol: '÷' }],
    ['power', { calculate: Math.pow, symbol: '^' }],
    ['min', { calculate: Math.min, name: 'MIN' }],
    ['max', { calculate: Math.max, name: 'MAX' }],
    ['pmt', { calculate: payment, name: 'PMT' }],
    ['declining_interest', {
        calculate: (principal, rate, periods) => principal * rate * (loanPeriods(principal, rate, periods) + 1) / 2,
        name: 'LÃI_GỐC_ĐỀU',
    }],
]);

export function calculateResults(calculations, inputs) {
    const values = new Map(inputs);
    for (const item of calculations) {
        const args = item.args.map(arg => typeof arg === 'string' ? values.get(arg) : arg);
        const operation = operations.get(item.op);
        if (!operation || !args.every(Number.isFinite)) continue;
        const result = operation.calculate(...args);
        if (Number.isFinite(result) && Math.abs(result) <= MAX_MAGNITUDE) values.set(item.id, result);
    }
    return values;
}

function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
}

function formula(item, names) {
    const operation = operations.get(item.op);
    const args = item.args.map(arg => {
        if (typeof arg === 'string') return names.get(arg);
        return String(arg);
    });
    if (operation.name) return `${operation.name}(${args.join(', ')})`;
    return args.join(` ${operation.symbol} `);
}

function formulaDetails(visual, names) {
    const details = node('details', 'simulator-formulas');
    const table = node('table');
    const heading = table.createTHead().insertRow();
    for (const label of ['Biến kết quả', 'Công thức']) {
        const cell = node('th', null, label);
        cell.scope = 'col';
        heading.append(cell);
    }
    const body = table.createTBody();
    for (const item of visual.calculations) {
        const row = body.insertRow();
        const name = node('th', null, item.label);
        name.scope = 'row';
        const expression = node('td', null, formula(item, names));
        row.append(name, expression);
    }
    const scroll = node('div', 'simulator-formula-table');
    scroll.append(table);
    details.append(node('summary', null, 'Công thức tính'), scroll);
    if (visual.calculations.some(item => item.op === 'pmt')) {
        details.append(node('p', 'simulator-note',
            'PMT = P × r ÷ (1 − (1 + r)⁻ⁿ). P là vốn vay, r là lãi suất mỗi kỳ dưới dạng thập phân và n là số kỳ. Khi r = 0, PMT = P ÷ n.'));
    }
    if (visual.calculations.some(item => item.op === 'declining_interest')) {
        details.append(node('p', 'simulator-note',
            'LÃI_GỐC_ĐỀU = P × r × (n + 1) ÷ 2. P là vốn vay, r là lãi suất mỗi kỳ dưới dạng thập phân và n là số kỳ.'));
    }
    return details;
}

function display(value, unit, decimals) {
    const fractionDigits = decimals ?? (/^(?:vnd|vnđ|đ|₫)$/iu.test(unit) ? 0 : 6);
    if (!numberFormats.has(fractionDigits)) {
        numberFormats.set(fractionDigits, new Intl.NumberFormat('vi-VN', { maximumFractionDigits: fractionDigits }));
    }
    // A floating-point residual that rounds to zero should not display as −0.
    const displayValue = Math.abs(value) < 0.5 * 10 ** -fractionDigits ? 0 : value;
    return numberFormats.get(fractionDigits).format(displayValue) + (unit ? ' ' + unit : '');
}

function readNumber(text, whole) {
    const value = text.trim();
    if (whole) {
        if (/^[+-]?[0-9]+$/.test(value)) return Number(value);
        if (/^[+-]?[1-9][0-9]{0,2}([., \u00a0\u202f])[0-9]{3}(?:\1[0-9]{3})*$/u.test(value)) {
            return Number(value.replace(/[., \u00a0\u202f]/gu, ''));
        }
        return NaN;
    }
    if (/^[+-]?(?:[0-9]+(?:[.,][0-9]*)?|[.,][0-9]+)(?:e[+-]?[0-9]{1,3})?$/i.test(value)) {
        return Number(value.replace(',', '.'));
    }
    return NaN;
}

function inputControl(item, refresh) {
    const id = `simulator-input-${++nextControlId}`;
    const row = node('div', 'simulator-input');
    const header = node('div', 'simulator-input-header');
    const showNumber = item.control !== 'slider';
    const showSlider = item.control !== 'number';
    const label = node('label', null, item.label);
    label.htmlFor = showNumber ? id : `${id}-slider`;
    const number = node('input', 'simulator-number');
    number.id = id;
    number.type = 'text';
    number.inputMode = item.whole ? 'numeric' : 'decimal';
    number.maxLength = 32;
    number.value = String(item.value).replace('.', ',');
    const valueField = node('div', 'simulator-value-field');
    const sliderValue = node('span');
    valueField.append(showNumber ? number : sliderValue);
    const descriptions = [];
    if (item.unit) {
        const unit = node('span', null, item.unit);
        unit.id = `${id}-unit`;
        descriptions.push(unit.id);
        valueField.append(unit);
    }
    header.append(label, valueField);

    const slider = node('input');
    slider.id = `${id}-slider`;
    slider.type = 'range';
    slider.min = item.min;
    slider.max = item.max;
    slider.step = item.step;
    slider.value = item.value;
    slider.setAttribute('aria-label', `${item.label}, thanh kéo`);
    const limits = node('div', 'simulator-limits');
    limits.setAttribute('aria-hidden', 'true');
    limits.append(node('span', null, display(item.min, item.unit)), node('span', null, display(item.max, item.unit)));
    const error = node('p', 'simulator-input-error');
    error.id = `${id}-error`;
    error.hidden = true;
    descriptions.push(error.id);
    number.setAttribute('aria-describedby', descriptions.join(' '));

    let current;
    function update() {
        const value = readNumber(number.value, item.whole);
        const valid = Number.isFinite(value) && value >= item.min && value <= item.max && (!item.whole || Number.isInteger(value));
        current = valid ? value : undefined;
        number.setAttribute('aria-invalid', String(!valid));
        error.hidden = valid;
        if (valid) {
            slider.value = value;
            slider.setAttribute('aria-valuetext', display(slider.valueAsNumber, item.unit));
            sliderValue.textContent = display(value, '');
        } else {
            const rule = item.whole ? 'số nguyên' : 'số';
            error.textContent = `Nhập ${rule} từ ${display(item.min, item.unit)} đến ${display(item.max, item.unit)}.`;
        }
    }
    update();
    number.addEventListener('input', () => { update(); refresh(); });
    slider.addEventListener('input', () => { number.value = slider.value.replace('.', ','); update(); refresh(); });
    row.append(header);
    if (showSlider) row.append(slider);
    row.append(limits, error);
    return {
        row,
        value: () => [item.id, current],
        reset() { number.value = String(item.value).replace('.', ','); update(); },
    };
}

export function renderSimulator(surface, visual) {
    const form = node('div', visual.layout ? 'simulator-composed' : 'simulator-panel');
    const summary = node('div', 'simulator-assumptions');
    const status = node('p', 'simulator-note');
    status.setAttribute('role', 'status');
    status.hidden = true;

    const names = new Map([...visual.inputs, ...visual.calculations].map(item => [item.id, item.label]));
    const controls = visual.inputs.map(item => inputControl(item, refresh));
    const output = visual.calculations.filter(item => item.show).map((item, index) => {
        const primary = !visual.layout && index === 0;
        const card = node('div', primary ? 'simulator-result simulator-result-primary' : 'simulator-result');
        const amount = node('strong');
        card.append(node('div', 'simulator-result-label', item.label), amount);
        if (primary) card.append(node('div', 'simulator-note', formula(item, names)));
        return { item, amount, card };
    });
    function refresh() {
        const values = calculateResults(visual.calculations, controls.map(control => control.value()));
        let unavailable = false;
        for (const { item, amount } of output) {
            const value = values.get(item.id);
            amount.textContent = Number.isFinite(value) ? display(value, item.unit, item.decimals) : '—';
            unavailable ||= !Number.isFinite(value);
        }
        status.hidden = !unavailable;
        status.textContent = unavailable ? 'Một số kết quả chưa tính được với các giá trị hiện tại.' : '';
    }

    summary.append(node('p', null, 'Giả định mô phỏng'));
    const assumptions = node('ul');
    assumptions.append(...visual.assumptions.map(text => node('li', null, text)));
    summary.append(assumptions);
    const reset = node('button', 'simulator-reset', 'Khôi phục giá trị ban đầu');
    reset.type = 'button';
    reset.addEventListener('click', () => { controls.forEach(control => control.reset()); refresh(); });
    if (visual.layout) {
        const inputs = new Map(visual.inputs.map((item, index) => [item.id, controls[index].row]));
        const outputs = new Map(output.map(({item, card}) => [item.id, card]));
        const components = {
            input: block => inputs.get(block.ref),
            result: block => outputs.get(block.ref),
            formulas: block => {
                const details = formulaDetails(visual, names);
                if (block.text) details.querySelector('summary').textContent = block.text;
                return details;
            },
            assumptions: block => {
                if (block.text) summary.firstElementChild.textContent = block.text;
                return summary;
            },
            reset: block => {
                if (block.text) reset.textContent = block.text;
                return reset;
            },
        };
        form.append(composeLayout(visual.layout, block => components[block.type](block)), status);
    } else {
        const fields = node('div', 'simulator-fields');
        const results = node('div', 'simulator-results');
        fields.append(...controls.map(control => control.row));
        results.append(...output.map(item => item.card));
        form.append(fields, results, status, formulaDetails(visual, names), summary, reset);
    }
    surface.replaceChildren(form);
    refresh();
}

