const containers = new Set(['column', 'row', 'grid']);

export function composeLayout(block, renderComponent) {
    const wrapper = document.createElement('div');
    wrapper.classList.add('visual-layout-block');
    if (block.tone === 'card' || block.tone === 'accent') {
        wrapper.classList.add(`visual-layout-${block.tone}`);
    }
    if (containers.has(block.type)) {
        wrapper.classList.add(`visual-layout-${block.type}`);
        if (block.type === 'grid') wrapper.style.setProperty('--layout-columns', block.columns ?? 2);
        wrapper.append(...block.children.map(child => composeLayout(child, renderComponent)));
    } else if (block.type === 'heading' || block.type === 'text') {
        const text = document.createElement(block.type === 'heading' ? 'h3' : 'p');
        text.textContent = block.text;
        wrapper.append(text);
    } else {
        wrapper.append(renderComponent(block));
    }
    return wrapper;
}
