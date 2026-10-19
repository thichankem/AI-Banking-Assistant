/**
 * Vietcombank AI Banking Assistant - Chunking & DLA Bounding Box Inspector
 * 
 * ĐIỀU PHỐI GIAO DIỆN CHIA 2 NỬA (DUAL-PANE SPLIT VIEW) - 108 TỆP
 * Nửa trái: Xem tệp gốc (PDF Canvas / DOCX Paper) & Bounding Box DLA (0% Regex)
 * Nửa phải: Danh sách Chunks & Tương tác 2 chiều mượt mà
 * Hỗ trợ Auto Fit-to-Width & Fit-to-Page: Luôn thấy trọn vẹn 100% trang không bị cắt mép
 */

// State quản lý ứng dụng
const state = {
  manifest: [],
  currentDocIndex: 0,
  currentDoc: null,
  currentLayoutData: null,
  activeBoxId: null,
  activeChunkId: null,
  zoomMode: 'fit-width', // 'fit-width', 'fit-page', 'custom'
  zoomLevel: 1.0,
  showBoxes: true,
  showText: true,
  showBadges: false,
  showRawBg: true,
  showChunkBoundaries: true,
  boxOpacity: 0.22,
  filterTypes: {
    title: true,
    heading_1: true,
    heading_2: true,
    heading_3: true,
    body: true,
    table: true,
    header_footer: true
  },
  searchTerm: '',
  selectedChunkType: 'ALL',
  docFilterFormat: 'ALL', // 'ALL', 'pdf', 'docx'
  theme: 'dark',
  rawPdfDoc: null,
  rawPdfBuffer: null,
  rawDocxBuffer: null,
  rawDocxHtml: null,
  isRendering: false
};

// Cấu hình bảng màu chuẩn cho các cấp phân bậc DLA (0% Regex)
const ELEMENT_COLORS = {
  title: { border: '#a855f7', bg: 'rgba(168, 85, 247, 0.25)', label: 'Title' },
  heading_1: { border: '#3b82f6', bg: 'rgba(59, 130, 246, 0.25)', label: 'H1 / Chương' },
  heading_2: { border: '#06b6d4', bg: 'rgba(6, 182, 212, 0.25)', label: 'H2 / Mục' },
  heading_3: { border: '#14b8a6', bg: 'rgba(20, 184, 166, 0.25)', label: 'H3 / Điểm' },
  body: { border: '#10b981', bg: 'rgba(16, 185, 129, 0.18)', label: 'Body' },
  table: { border: '#f59e0b', bg: 'rgba(245, 158, 11, 0.25)', label: 'Bảng biểu' },
  header_footer: { border: '#ef4444', bg: 'rgba(239, 68, 68, 0.20)', label: 'Header/Footer' }
};

// Khởi chạy khi DOM sẵn sàng
document.addEventListener('DOMContentLoaded', async () => {
  initUI();
  setupEventListeners();
  setupSplitResizer();
  setupDragAndDrop();
  await load108Manifest();
});

/**
 * Nạp danh mục 108 tệp từ window.DOCUMENTS_108_MANIFEST hoặc fetch API
 */
async function load108Manifest() {
  if (window.DOCUMENTS_108_MANIFEST && window.DOCUMENTS_108_MANIFEST.length > 0) {
    state.manifest = window.DOCUMENTS_108_MANIFEST;
  } else {
    try {
      const res = await fetch('manifest_108.json');
      if (res.ok) {
        state.manifest = await res.json();
      }
    } catch (err) {
      console.warn('Could not fetch manifest_108.json:', err);
    }
  }

  if (!state.manifest || state.manifest.length === 0) {
    console.error('Không tìm thấy danh mục 108 tệp!');
    return;
  }

  populateMasterDocSelect();

  // Chọn mặc định tệp PDF chuyển tiền ngoại tệ (vcb_pdf_24) hoặc tệp đầu tiên
  const defaultIdx = state.manifest.findIndex(m => m.name.includes('vcb_pdf_24'));
  const startIdx = defaultIdx !== -1 ? defaultIdx : 0;
  await selectDocumentByIndex(startIdx);
}

/**
 * Đổ danh sách 108 tệp vào Master Select Dropdown
 */
function populateMasterDocSelect() {
  const masterSelect = document.getElementById('doc-master-select');
  if (!masterSelect) return;

  masterSelect.innerHTML = '';

  const filter = state.docFilterFormat;
  const filteredDocs = state.manifest.filter(doc => {
    if (filter === 'pdf') return doc.format === 'pdf';
    if (filter === 'docx') return doc.format === 'docx';
    return true;
  });

  // Phân nhóm PDF và DOCX
  const pdfDocs = filteredDocs.filter(d => d.format === 'pdf');
  const docxDocs = filteredDocs.filter(d => d.format === 'docx');

  if (pdfDocs.length > 0) {
    const groupPdf = document.createElement('optgroup');
    groupPdf.label = `TỆP PDF NGÂN HÀNG & SLIDE (${pdfDocs.length} tệp)`;
    pdfDocs.forEach(doc => {
      const opt = document.createElement('option');
      opt.value = doc.name;
      opt.textContent = `${doc.name} (${doc.total_pages} trang • ${doc.elements_count} boxes • ${doc.chunks_count} chunks)`;
      groupPdf.appendChild(opt);
    });
    masterSelect.appendChild(groupPdf);
  }

  if (docxDocs.length > 0) {
    const groupDocx = document.createElement('optgroup');
    groupDocx.label = `TỆP DOCX QUY CHẾ & HƯỚNG DẪN (${docxDocs.length} tệp)`;
    docxDocs.forEach(doc => {
      const opt = document.createElement('option');
      opt.value = doc.name;
      opt.textContent = `${doc.name} (${doc.elements_count} boxes • ${doc.chunks_count} chunks)`;
      groupDocx.appendChild(opt);
    });
    masterSelect.appendChild(groupDocx);
  }
}

/**
 * Chọn và hiển thị tài liệu theo chỉ số
 */
async function selectDocumentByIndex(index) {
  if (index < 0 || index >= state.manifest.length) return;
  clearActiveSelection();
  state.currentDocIndex = index;
  state.currentDoc = state.manifest[index];

  const masterSelect = document.getElementById('doc-master-select');
  if (masterSelect) {
    masterSelect.value = state.currentDoc.name;
  }

  // Cập nhật số thứ tự tệp trên toolbar
  const statDocIndex = document.getElementById('stat-doc-index');
  if (statDocIndex) {
    statDocIndex.textContent = `${index + 1}/${state.manifest.length}`;
  }

  await loadAndRenderDocument(state.currentDoc);
}

/**
 * Chuẩn hóa và tự động ánh xạ 2 chiều giữa Elements và Chunks (đảm bảo 100% Khung Chunk luôn hiển thị)
 */
function normalizeChunkElementLinks(layoutData) {
  if (!layoutData || !layoutData.chunks || !layoutData.pages) return;
  const chunks = layoutData.chunks;
  const allElements = [];
  layoutData.pages.forEach(p => {
    (p.elements || []).forEach(el => allElements.push(el));
  });

  // 1. Ánh xạ 2 chiều c.element_ids <-> el.chunk_id
  chunks.forEach(c => {
    const cid = c.chunk_id;
    if (!c.element_ids) c.element_ids = [];
    const eids = new Set(c.element_ids);
    allElements.forEach(el => {
      if (eids.has(el.element_id)) {
        el.chunk_id = cid;
      } else if (el.chunk_id === cid) {
        c.element_ids.push(el.element_id);
      }
    });
  });

  // 2. Với các elements vẫn chưa có chunk_id: Tự động gom theo loại và nội dung
  chunks.forEach(c => {
    const cid = c.chunk_id;
    const isTable = (c.chunk_type || '').includes('table') || (c.content || '').includes('Cột:');
    const cContent = (c.content || c.raw_content || '').toLowerCase();

    allElements.forEach(el => {
      if (el.type === 'header_footer') return;
      if (el.chunk_id) return;

      const elText = (el.text || '').trim().toLowerCase();
      if (!elText) return;

      if (isTable && el.type === 'table') {
        el.chunk_id = cid;
        if (!c.element_ids.includes(el.element_id)) c.element_ids.push(el.element_id);
      } else if (!isTable && el.type !== 'table') {
        if (cContent.includes(elText) || (elText.length >= 10 && cContent.includes(elText.slice(0, 20)))) {
          el.chunk_id = cid;
          if (!c.element_ids.includes(el.element_id)) c.element_ids.push(el.element_id);
        }
      }
    });
  });

  // 3. Đảm bảo tuyệt đối KHÔNG có chunk nào bị 0 boxes
  chunks.forEach(c => {
    if (!c.element_ids || c.element_ids.length === 0) {
      const pNums = c.page_numbers || (c.location ? c.location.map(l => l.page) : [c.page_start || 1]);
      const candEls = allElements.filter(el => pNums.includes(el.page_num) && el.type !== 'header_footer');
      if (candEls.length > 0) {
        c.element_ids = candEls.map(e => e.element_id);
        candEls.forEach(e => { if (!e.chunk_id) e.chunk_id = c.chunk_id; });
      } else if (allElements.length > 0) {
        c.element_ids = [allElements[0].element_id];
      }
    }
  });

  // 4. Quét lần cuối theo từng trang để không bỏ sót bất kỳ box nào
  layoutData.pages.forEach(p => {
    const pageEls = (p.elements || []).filter(el => el.type !== 'header_footer');
    const pageChunks = chunks.filter(c => {
      const pNums = c.page_numbers || (c.location ? c.location.map(l => l.page) : [c.page_start || 1]);
      return pNums.includes(p.page_num);
    });
    const fallbackChunk = pageChunks.find(c => !(c.chunk_type || '').includes('table')) || pageChunks[0] || chunks[0];
    if (fallbackChunk) {
      const fcId = fallbackChunk.chunk_id;
      if (!fallbackChunk.element_ids) fallbackChunk.element_ids = [];
      pageEls.forEach(el => {
        if (!el.chunk_id) {
          el.chunk_id = fcId;
          if (!fallbackChunk.element_ids.includes(el.element_id)) {
            fallbackChunk.element_ids.push(el.element_id);
          }
        }
      });
    }
  });
}

/**
 * Tải Layout DLA chuẩn và Tệp gốc (PDF / DOCX)
 */
async function loadAndRenderDocument(docInfo) {
  try {
    // 1. Tải Layout DLA JSON
    let layoutData = null;
    try {
      const res = await fetch(`layouts/${docInfo.name}.json`);
      if (res.ok) {
        layoutData = await res.json();
      }
    } catch (e) {
      console.warn(`Lỗi fetch layouts/${docInfo.name}.json:`, e);
    }

    if (!layoutData) {
      console.error(`Không tìm thấy dữ liệu layout cho ${docInfo.name}`);
      return;
    }

    // Luôn đảm bảo 100% ánh xạ giữa Elements và Khung Chunk
    normalizeChunkElementLinks(layoutData);

    state.currentLayoutData = layoutData;
    state.activeBoxId = null;
    state.activeChunkId = null;

    updateHeaderStats();

    // 2. Tải và render tệp gốc
    const rawUrl = docInfo.raw_url;
    let arrayBuffer = null;
    try {
      const res = await fetch(rawUrl);
      if (res.ok) {
        arrayBuffer = await res.arrayBuffer();
      }
    } catch (e) {
      console.warn(`Lỗi tải raw file từ ${rawUrl}:`, e);
    }

    const badge = document.getElementById('file-origin-badge');
    if (docInfo.format === 'pdf') {
      if (badge) {
        badge.textContent = `PDF Gốc (${layoutData.total_pages} trang)`;
        badge.style.display = 'inline-block';
        badge.style.background = 'rgba(16, 185, 129, 0.2)';
        badge.style.color = '#10b981';
      }
      if (arrayBuffer && window.pdfjsLib) {
        try {
          pdfjsLib.GlobalWorkerOptions.workerSrc = 'libs/pdf.worker.min.js';
        } catch (e) {}
        const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
        const pdfDoc = await loadingTask.promise;
        state.rawPdfDoc = pdfDoc;
        state.rawPdfBuffer = arrayBuffer;
        await renderPdfCanvasView(pdfDoc, layoutData);
      } else {
        renderSimulatedLayoutView(layoutData);
      }
    } else {
      // DOCX
      if (badge) {
        badge.textContent = `DOCX Gốc (${layoutData.elements_count} elements)`;
        badge.style.display = 'inline-block';
        badge.style.background = 'rgba(59, 130, 246, 0.2)';
        badge.style.color = '#38bdf8';
      }
      if (arrayBuffer && window.mammoth) {
        const result = await mammoth.convertToHtml({ arrayBuffer });
        state.rawDocxHtml = result.value;
        renderDocxPageView(result.value, layoutData);
      } else {
        renderSimulatedLayoutView(layoutData);
      }
    }

    // 3. Render Chunks bên phải
    renderChunksList();
    clearActiveSelection();

  } catch (err) {
    console.error('Lỗi khi nạp tài liệu:', err);
  }
}

/**
 * Tính toán tỷ lệ Scale hiển thị sao cho VỪA KHÍT 100% Khung nhìn
 */
function calculateOptimalScale(pageData) {
  const container = document.getElementById('page-viewer-scroll');
  if (!container) return 1.0;

  // Chiều rộng và chiều cao thực tế khả dụng của vùng cuộn
  const padH = 48;
  const availableWidth = Math.max(container.clientWidth - padH, 320);
  const availableHeight = Math.max(container.clientHeight - 80, 400);

  const baseW = pageData.width || 595.0;
  const baseH = pageData.height || 842.0;

  if (state.zoomMode === 'fit-width') {
    // Vừa trọn vẹn 100% bề ngang khung nhìn
    const scale = availableWidth / baseW;
    return Math.max(0.35, Math.min(scale, 2.4));
  } else if (state.zoomMode === 'fit-page') {
    // Thấy trọn vẹn cả chiều cao và chiều ngang của trang
    const scaleW = availableWidth / baseW;
    const scaleH = availableHeight / baseH;
    const scale = Math.min(scaleW, scaleH);
    return Math.max(0.35, Math.min(scale, 2.0));
  } else {
    // Custom Zoom
    return state.zoomLevel;
  }
}

/**
 * Kết xuất PDF Canvas thực tế với Bounding Box phủ trên từng trang
 * Tự động căn chỉnh vừa vặn khung nhìn, không bao giờ bị cắt mép!
 */
async function renderPdfCanvasView(pdfDoc, layoutData) {
  if (state.isRendering) return;
  state.isRendering = true;

  const container = document.getElementById('page-viewer-scroll');
  container.innerHTML = '';

  const pageJumpSelect = document.getElementById('page-jump-select');
  if (pageJumpSelect) {
    pageJumpSelect.innerHTML = '';
  }

  const pages = layoutData.pages || [];
  const maxRenderPages = Math.min(pdfDoc.numPages, pages.length || pdfDoc.numPages);

  // Tính scale chuẩn dựa trên trang đầu tiên
  const firstPageData = pages[0] || { width: 595.0, height: 842.0 };
  const targetScale = calculateOptimalScale(firstPageData);
  state.zoomLevel = targetScale;
  updateZoomLabel();

  for (let p = 1; p <= maxRenderPages; p++) {
    const pageData = pages[p - 1] || { elements: [], width: 595, height: 842 };
    const elements = pageData.elements || [];

    // Cập nhật jump select
    if (pageJumpSelect) {
      const opt = document.createElement('option');
      opt.value = `page-container-${p}`;
      opt.textContent = `Trang ${p} / ${pdfDoc.numPages} (${elements.length} boxes)`;
      pageJumpSelect.appendChild(opt);
    }

    // Lấy kích thước trang từ PDF
    const pdfPage = await pdfDoc.getPage(p);
    const viewport = pdfPage.getViewport({ scale: targetScale });

    // Khởi tạo container trang
    const pageWrapper = document.createElement('div');
    pageWrapper.className = 'page-canvas-container';
    pageWrapper.id = `page-container-${p}`;
    pageWrapper.style.position = 'relative';
    pageWrapper.style.width = `${Math.round(viewport.width)}px`;

    const pageHeader = document.createElement('div');
    pageHeader.className = 'page-divider-header';
    pageHeader.innerHTML = `<span>Trang ${p} / ${pdfDoc.numPages}</span>`;
    pageWrapper.appendChild(pageHeader);

    // Canvas vẽ PDF gốc
    const canvas = document.createElement('canvas');
    canvas.className = 'pdf-raw-canvas';
    canvas.width = Math.round(viewport.width);
    canvas.height = Math.round(viewport.height);
    canvas.style.width = `${Math.round(viewport.width)}px`;
    canvas.style.height = `${Math.round(viewport.height)}px`;
    pageWrapper.appendChild(canvas);

    // Layer phủ Bounding Box
    const boxLayer = document.createElement('div');
    boxLayer.className = 'canvas-bbox-layer';
    boxLayer.style.width = `${Math.round(viewport.width)}px`;
    boxLayer.style.height = `${Math.round(viewport.height)}px`;
    pageWrapper.appendChild(boxLayer);

    container.appendChild(pageWrapper);

    // Căn chuẩn vị trí đỉnh của boxLayer khớp với mặt trên của canvas
    const headerHeight = pageHeader.offsetHeight || 36;
    boxLayer.style.top = `${headerHeight}px`;

    const ctx = canvas.getContext('2d');
    await pdfPage.render({ canvasContext: ctx, viewport }).promise;

    // Tỷ lệ scale giữa viewport PDF.js và tọa độ gốc của trang
    const scaleX = viewport.width / (pageData.width || viewport.width);
    const scaleY = viewport.height / (pageData.height || viewport.height);

    // 1. Dựng các Ô BAO QUANH CHUNK (Chunk Boundary Envelope Boxes)
    const chunkMap = new Map();
    elements.forEach(el => {
      let cId = el.chunk_id;
      if (!cId && layoutData.chunks) {
        const found = layoutData.chunks.find(c => c.element_ids && c.element_ids.includes(el.element_id));
        if (found) cId = found.chunk_id;
      }
      if (cId && el.box && el.box.length === 4) {
        if (!chunkMap.has(cId)) {
          chunkMap.set(cId, []);
        }
        chunkMap.get(cId).push(el);
      }
    });

    // Bắt sự kiện click trên Canvas để dù ấn ở bất kỳ đâu trên vùng trang, bên phải cũng móc ngay đến cùng!
    canvas.style.cursor = 'pointer';
    canvas.addEventListener('click', (e) => {
      const rect = canvas.getBoundingClientRect();
      const clickX = (e.clientX - rect.left) / scaleX;
      const clickY = (e.clientY - rect.top) / scaleY;

      // Tìm element chứa tọa độ click
      const clickedEl = elements.find(el => {
        if (!el.box || el.box.length < 4) return false;
        return clickX >= el.box[0] && clickX <= el.box[2] && clickY >= el.box[1] && clickY <= el.box[3];
      });

      if (clickedEl) {
        handleElementClick(clickedEl.element_id, clickedEl.chunk_id, clickedEl.text);
        return;
      }

      // Tìm chunk boundary chứa tọa độ click
      let foundChunkId = null;
      chunkMap.forEach((chunkElements, cId) => {
        if (foundChunkId) return;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        chunkElements.forEach(el => {
          const [x0, y0, x1, y1] = el.box;
          if (x0 < minX) minX = x0;
          if (y0 < minY) minY = y0;
          if (x1 > maxX) maxX = x1;
          if (y1 > maxY) maxY = y1;
        });
        const pad = 8;
        if (clickX >= minX - pad && clickX <= maxX + pad && clickY >= minY - pad && clickY <= maxY + pad) {
          foundChunkId = cId;
        }
      });

      if (foundChunkId) {
        const targetChunk = (layoutData.chunks || []).find(c => c.chunk_id === foundChunkId);
        handleChunkClick(foundChunkId, targetChunk?.element_ids);
      }
    });

    // Gom nhóm các box theo cụm liên tục (Clusters) để không tạo ô bao trùm lên bảng biểu/đoạn khác
    chunkMap.forEach((chunkElements, chunkId) => {
      const sortedEls = [...chunkElements].sort((a, b) => a.box[1] - b.box[1]);
      const clusters = [];
      let currentCluster = [];

      sortedEls.forEach(el => {
        if (currentCluster.length === 0) {
          currentCluster.push(el);
        } else {
          const prevEl = currentCluster[currentCluster.length - 1];
          const gap = el.box[1] - prevEl.box[3];
          if (gap > 65) {
            clusters.push(currentCluster);
            currentCluster = [el];
          } else {
            currentCluster.push(el);
          }
        }
      });
      if (currentCluster.length > 0) {
        clusters.push(currentCluster);
      }

      clusters.forEach((clusterEls, clusterIdx) => {
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        clusterEls.forEach(el => {
          const [x0, y0, x1, y1] = el.box;
          if (x0 < minX) minX = x0;
          if (y0 < minY) minY = y0;
          if (x1 > maxX) maxX = x1;
          if (y1 > maxY) maxY = y1;
        });

        if (isFinite(minX) && isFinite(minY) && isFinite(maxX) && isFinite(maxY)) {
          const pad = 6;
          const left = Math.max(0, (minX * scaleX) - pad);
          const top = Math.max(0, (minY * scaleY) - pad);
          const width = Math.min(viewport.width - left, ((maxX - minX) * scaleX) + (pad * 2));
          const height = Math.min(viewport.height - top, ((maxY - minY) * scaleY) + (pad * 2));

          const chunkBox = document.createElement('div');
          chunkBox.className = 'chunk-boundary-box';
          chunkBox.id = `chunk-boundary-${chunkId}-p${p}-c${clusterIdx}`;
          chunkBox.setAttribute('data-chunk-id', chunkId);
          chunkBox.setAttribute('data-page-num', p);
          chunkBox.style.left = `${left}px`;
          chunkBox.style.top = `${top}px`;
          chunkBox.style.width = `${width}px`;
          chunkBox.style.height = `${height}px`;

          // Nhãn tag của Chunk
          const tag = document.createElement('div');
          tag.className = 'chunk-boundary-tag';
          const chunkIdx = (layoutData.chunks || []).findIndex(c => c.chunk_id === chunkId);
          const displayId = getCleanChunkLabel(chunkId, chunkIdx >= 0 ? chunkIdx : null);
          const partSuffix = clusters.length > 1 ? ` (Phần ${clusterIdx + 1})` : '';
          tag.innerHTML = `<span>${escapeHtml(displayId + partSuffix)}</span>`;
          chunkBox.appendChild(tag);

          // Sự kiện Click vào Chunk Boundary: kích hoạt soi toàn bộ chunk
          chunkBox.addEventListener('click', (e) => {
            e.stopPropagation();
            const targetChunk = (layoutData.chunks || []).find(c => c.chunk_id === chunkId);
            handleChunkClick(chunkId, targetChunk?.element_ids);
          });

          boxLayer.appendChild(chunkBox);
        }
      });
    });

    // 2. Vẽ từng Bounding Box Element con
    elements.forEach(el => {
      const boxDiv = createBoundingBoxElement(el, scaleX, scaleY);
      boxLayer.appendChild(boxDiv);
    });
  }

  applyViewLayers();
  state.isRendering = false;
}

/**
 * Cập nhật nhãn hiển thị mức Zoom trên Toolbar
 */
function updateZoomLabel() {
  const zoomText = document.getElementById('zoom-level-text');
  if (!zoomText) return;

  if (state.zoomMode === 'fit-width') {
    zoomText.textContent = `Vừa ngang (${Math.round(state.zoomLevel * 100)}%)`;
  } else if (state.zoomMode === 'fit-page') {
    zoomText.textContent = `Toàn trang (${Math.round(state.zoomLevel * 100)}%)`;
  } else {
    zoomText.textContent = `${Math.round(state.zoomLevel * 100)}%`;
  }
}

/**
 * Re-render lại view hiện tại (dùng khi zoom hoặc resize)
 */
async function reRenderCurrentDocView() {
  if (!state.currentLayoutData) return;

  if (state.currentDoc?.format === 'pdf' && state.rawPdfDoc) {
    await renderPdfCanvasView(state.rawPdfDoc, state.currentLayoutData);
  } else if (state.rawDocxHtml) {
    renderDocxPageView(state.rawDocxHtml, state.currentLayoutData);
  }
}

/**
 * Kết xuất tệp DOCX dạng Paper View với Bounding Box song song
 */
function renderDocxPageView(htmlContent, layoutData) {
  const container = document.getElementById('page-viewer-scroll');
  container.innerHTML = '';

  const pageJumpSelect = document.getElementById('page-jump-select');
  if (pageJumpSelect) {
    pageJumpSelect.innerHTML = '<option value="page-container-1">Trang DOCX hoàn chỉnh</option>';
  }

  const availableWidth = container ? Math.max(container.clientWidth - 48, 320) : 750;
  const maxWidth = state.zoomMode === 'fit-width' ? `${availableWidth}px` : (state.zoomMode === 'fit-page' ? '680px' : `${Math.round(780 * state.zoomLevel)}px`);

  const pageWrapper = document.createElement('div');
  pageWrapper.className = 'docx-page-container';
  pageWrapper.id = 'page-container-1';
  pageWrapper.style.maxWidth = maxWidth;
  pageWrapper.style.width = '100%';
  pageWrapper.style.margin = '20px auto';
  pageWrapper.style.padding = '36px 44px';
  pageWrapper.style.background = '#ffffff';
  pageWrapper.style.color = '#1e293b';
  pageWrapper.style.boxShadow = '0 10px 30px rgba(0,0,0,0.4)';
  pageWrapper.style.borderRadius = '8px';
  pageWrapper.style.lineHeight = '1.7';
  pageWrapper.style.fontFamily = "'Inter', serif";

  // Thêm header
  const headerDiv = document.createElement('div');
  headerDiv.style.borderBottom = '2px solid #e2e8f0';
  headerDiv.style.paddingBottom = '12px';
  headerDiv.style.marginBottom = '24px';
  headerDiv.innerHTML = `
    <div style="font-size:11px; text-transform:uppercase; color:#64748b; font-weight:700;">Vietcombank Document Viewer</div>
    <div style="font-size:18px; font-weight:700; color:#0f172a;">${layoutData.title}</div>
  `;
  pageWrapper.appendChild(headerDiv);

  // Vùng nội dung HTML
  const contentDiv = document.createElement('div');
  contentDiv.className = 'docx-html-body';
  contentDiv.innerHTML = htmlContent;
  pageWrapper.appendChild(contentDiv);

  container.appendChild(pageWrapper);

  // Gắn các thẻ tương tác cho các đoạn văn và bảng trong HTML, gom nhóm theo Khung Chunk
  const paragraphs = contentDiv.querySelectorAll('p, h1, h2, h3, h4, table');
  const elements = [];
  (layoutData.pages || []).forEach(p => elements.push(...(p.elements || [])));

  let currentChunkId = null;
  let currentChunkWrapper = null;

  paragraphs.forEach((domEl, idx) => {
    const elData = elements[idx] || {
      element_id: `${layoutData.filename}_dom_${idx}`,
      type: domEl.tagName.toLowerCase().startsWith('h') ? 'heading_1' : (domEl.tagName.toLowerCase() === 'table' ? 'table' : 'body'),
      chunk_id: null
    };

    domEl.classList.add('docx-interactive-block');
    domEl.setAttribute('data-element-id', elData.element_id);
    if (elData.chunk_id) {
      domEl.setAttribute('data-chunk-id', elData.chunk_id);
    }

    const conf = ELEMENT_COLORS[elData.type] || ELEMENT_COLORS.body;
    domEl.style.borderLeft = `3.5px solid ${conf.border}`;
    domEl.style.paddingLeft = '10px';
    domEl.style.marginBottom = '12px';
    domEl.style.transition = 'all 0.2s';
    domEl.style.cursor = 'pointer';

    domEl.addEventListener('mouseenter', () => {
      domEl.style.background = conf.bg;
    });
    domEl.addEventListener('mouseleave', () => {
      if (state.activeBoxId !== elData.element_id) {
        domEl.style.background = 'transparent';
      }
    });

    domEl.addEventListener('click', (e) => {
      e.stopPropagation();
      handleElementClick(elData.element_id, elData.chunk_id, domEl.innerText);
    });

    // Dựng khung bao Chunk Boundary cho DOCX
    if (elData.chunk_id && domEl.parentElement === contentDiv) {
      if (elData.chunk_id !== currentChunkId || !currentChunkWrapper) {
        currentChunkId = elData.chunk_id;
        currentChunkWrapper = document.createElement('div');
        currentChunkWrapper.className = 'docx-chunk-boundary';
        currentChunkWrapper.setAttribute('data-chunk-id', currentChunkId);

        const tag = document.createElement('div');
        tag.className = 'chunk-boundary-tag';
        const chunkIdx = (layoutData.chunks || []).findIndex(c => c.chunk_id === currentChunkId);
        const displayId = getCleanChunkLabel(currentChunkId, chunkIdx >= 0 ? chunkIdx : null);
        tag.innerHTML = `<span>${escapeHtml(displayId)}</span>`;
        currentChunkWrapper.appendChild(tag);

        currentChunkWrapper.addEventListener('click', (e) => {
          if (e.target === currentChunkWrapper) {
            handleChunkClick(currentChunkId, layoutData.chunks?.find(c => c.chunk_id === currentChunkId)?.element_ids);
          }
        });

        contentDiv.insertBefore(currentChunkWrapper, domEl);
      }
      currentChunkWrapper.appendChild(domEl);
    } else if (domEl.parentElement === contentDiv) {
      currentChunkId = null;
      currentChunkWrapper = null;
    }
  });

  applyViewLayers();
  updateZoomLabel();
}

/**
 * Kết xuất mô phỏng nếu không có Canvas gốc
 */
function renderSimulatedLayoutView(layoutData) {
  const container = document.getElementById('page-viewer-scroll');
  container.innerHTML = '';

  const pages = layoutData.pages || [];
  pages.forEach(p => {
    const pageWrapper = document.createElement('div');
    pageWrapper.className = 'page-canvas-container';
    pageWrapper.id = `page-container-${p.page_num}`;
    pageWrapper.style.width = '680px';
    pageWrapper.style.minHeight = '900px';
    pageWrapper.style.margin = '20px auto';
    pageWrapper.style.background = '#ffffff';
    pageWrapper.style.position = 'relative';

    const boxLayer = document.createElement('div');
    boxLayer.className = 'canvas-bbox-layer';
    boxLayer.style.width = '100%';
    boxLayer.style.height = '100%';
    pageWrapper.appendChild(boxLayer);

    // Gom nhóm elements theo chunk_id để vẽ Khung Chunk
    const chunkMap = new Map();
    (p.elements || []).forEach(el => {
      if (el.chunk_id && el.box && el.box.length === 4) {
        if (!chunkMap.has(el.chunk_id)) {
          chunkMap.set(el.chunk_id, []);
        }
        chunkMap.get(el.chunk_id).push(el);
      }
    });

    chunkMap.forEach((chunkElements, chunkId) => {
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      chunkElements.forEach(el => {
        const [x0, y0, x1, y1] = el.box;
        if (x0 < minX) minX = x0;
        if (y0 < minY) minY = y0;
        if (x1 > maxX) maxX = x1;
        if (y1 > maxY) maxY = y1;
      });

      if (isFinite(minX) && isFinite(minY) && isFinite(maxX) && isFinite(maxY)) {
        const pad = 5;
        const chunkBox = document.createElement('div');
        chunkBox.className = 'chunk-boundary-box';
        chunkBox.id = `chunk-boundary-${chunkId}-p${p.page_num}`;
        chunkBox.setAttribute('data-chunk-id', chunkId);
        chunkBox.setAttribute('data-page-num', p.page_num);
        chunkBox.style.left = `${Math.max(0, minX - pad)}px`;
        chunkBox.style.top = `${Math.max(0, minY - pad)}px`;
        chunkBox.style.width = `${(maxX - minX) + (pad * 2)}px`;
        chunkBox.style.height = `${(maxY - minY) + (pad * 2)}px`;

        const tag = document.createElement('div');
        tag.className = 'chunk-boundary-tag';
        const chunkIdx = (layoutData.chunks || []).findIndex(c => c.chunk_id === chunkId);
        const displayId = getCleanChunkLabel(chunkId, chunkIdx >= 0 ? chunkIdx : null);
        tag.innerHTML = `<span>${escapeHtml(displayId)}</span>`;
        chunkBox.appendChild(tag);

        chunkBox.addEventListener('click', (e) => {
          e.stopPropagation();
          const targetChunk = (layoutData.chunks || []).find(c => c.chunk_id === chunkId);
          handleChunkClick(chunkId, targetChunk?.element_ids);
        });

        boxLayer.appendChild(chunkBox);
      }
    });

    (p.elements || []).forEach(el => {
      const boxDiv = createBoundingBoxElement(el, 1.0, 1.0);
      boxLayer.appendChild(boxDiv);
    });

    container.appendChild(pageWrapper);
  });

  applyViewLayers();
}

/**
 * Tạo một thẻ Bounding Box DOM trực quan
 */
function createBoundingBoxElement(el, scaleX, scaleY) {
  const box = el.box || [0, 0, 100, 30];
  const x = box[0] * scaleX;
  const y = box[1] * scaleY;
  const w = (box[2] - box[0]) * scaleX;
  const h = (box[3] - box[1]) * scaleY;

  const conf = ELEMENT_COLORS[el.type] || ELEMENT_COLORS.body;

  const div = document.createElement('div');
  div.className = `element-bbox bbox-${el.type}`;
  div.id = `bbox-${el.element_id}`;
  div.setAttribute('data-element-id', el.element_id);
  if (el.chunk_id) {
    div.setAttribute('data-chunk-id', el.chunk_id);
  }

  div.style.left = `${x}px`;
  div.style.top = `${y}px`;
  div.style.width = `${w}px`;
  div.style.height = `${h}px`;
  div.style.borderColor = conf.border;
  div.style.backgroundColor = conf.bg;
  div.style.opacity = state.showBoxes ? '1' : '0';

  if (!state.filterTypes[el.type]) {
    div.classList.add('filtered-out');
  }

  // Label badge nếu được bật
  if (state.showBadges) {
    const badge = document.createElement('span');
    badge.className = 'bbox-type-badge';
    badge.style.background = conf.border;
    badge.textContent = conf.label;
    div.appendChild(badge);
  }

  // Sự kiện Click vào Bounding Box
  div.addEventListener('click', (e) => {
    e.stopPropagation();
    handleElementClick(el.element_id, el.chunk_id, el.text);
  });

  // Tooltip khi hover
  div.addEventListener('mouseenter', (e) => {
    showTooltip(e, el, conf);
  });
  div.addEventListener('mouseleave', () => {
    hideTooltip();
  });

  return div;
}

/**
 * Xử lý khi người dùng Click vào một Element Box ở nửa trái
 */
function handleElementClick(elementId, chunkId, text) {
  // 1. Tự động tìm chunkId từ layoutData nếu element chưa được gán trực tiếp
  if (!chunkId && state.currentLayoutData && state.currentLayoutData.chunks) {
    const foundChunk = state.currentLayoutData.chunks.find(c => 
      (c.element_ids && c.element_ids.includes(elementId)) ||
      (c.content && text && c.content.includes(text.trim()))
    );
    if (foundChunk) {
      chunkId = foundChunk.chunk_id;
    }
  }

  state.activeBoxId = elementId;
  state.activeChunkId = chunkId;

  // 2. Highlight Box bên trái được click
  document.querySelectorAll('.element-bbox, .docx-interactive-block').forEach(b => {
    b.classList.remove('active-box');
  });
  const targetBox = document.getElementById(`bbox-${elementId}`) || document.querySelector(`[data-element-id="${elementId}"]`);
  if (targetBox) {
    targetBox.classList.add('active-box');
  }

  // 3. Highlight Khung Chunk cha và tất cả sibling boxes cùng chunk
  document.querySelectorAll('.chunk-boundary-box, .docx-chunk-boundary').forEach(cb => {
    cb.classList.remove('active-chunk-boundary');
  });
  document.querySelectorAll('.element-bbox, .docx-interactive-block').forEach(b => {
    b.classList.remove('chunk-linked-box');
  });

  if (chunkId) {
    const parentBoundaries = document.querySelectorAll(`.chunk-boundary-box[data-chunk-id="${chunkId}"], .docx-chunk-boundary[data-chunk-id="${chunkId}"]`);
    parentBoundaries.forEach(cb => cb.classList.add('active-chunk-boundary'));

    const siblingBoxes = document.querySelectorAll(`.element-bbox[data-chunk-id="${chunkId}"], .docx-interactive-block[data-chunk-id="${chunkId}"]`);
    siblingBoxes.forEach(b => {
      if (b !== targetBox) {
        b.classList.add('chunk-linked-box');
      }
    });
  }

  // 4. Banner thông báo
  const banner = document.getElementById('active-inspect-banner');
  const info = document.getElementById('active-inspect-info');
  if (banner && info) {
    banner.classList.remove('hidden');
    const cleanElId = getCleanElementId(elementId);
    let chunkLabel = '';
    if (chunkId) {
      const cIdx = state.currentLayoutData?.chunks?.findIndex(c => c.chunk_id === chunkId);
      chunkLabel = getCleanChunkLabel(chunkId, cIdx >= 0 ? cIdx : null);
    }
    info.innerHTML = `
      <span>Element: <strong>${escapeHtml(cleanElId)}</strong></span>
      ${chunkId ? ` • <span>${escapeHtml(chunkLabel)}</span>` : '<span style="color:#f59e0b;"> (Chưa gán chunk)</span>'}
    `;
  }

  // 5. MÓC ĐẾN CÙNG: Cuộn danh sách Chunks bên phải đến tận thẻ Chunk tương ứng
  if (chunkId) {
    hookAndScrollRightPaneToChunk(chunkId, text);
  }
}

/**
 * MÓC CHUẨN XÁC VÀ CUỘN DANH SÁCH CHUNKS BÊN PHẢI ĐẾN TẬN CÙNG
 * Tự động giải phóng mọi bộ lọc (Search, Chip type), cuộn container mượt mà,
 * và tạo hiệu ứng nhấp nháy làm nổi bật.
 */
function hookAndScrollRightPaneToChunk(chunkId, targetElementText = null) {
  if (!chunkId) return;

  const container = document.getElementById('chunks-list-container');
  if (!container) return;

  // 1. Kiểm tra xem thẻ Chunk có đang tồn tại trong DOM không
  let targetCard = document.getElementById(`chunk-card-${chunkId}`);

  // Nếu thẻ chưa có trong DOM (do đang bị lọc bởi Tìm kiếm hoặc Filter Chip), tự động reset bộ lọc để nạp lại đầy đủ 100% Chunks!
  if (!targetCard) {
    let needRerender = false;
    const searchInput = document.getElementById('chunk-search-input');
    if (searchInput && searchInput.value) {
      searchInput.value = '';
      state.searchTerm = '';
      needRerender = true;
    }
    if (state.selectedChunkType !== 'ALL') {
      state.selectedChunkType = 'ALL';
      document.querySelectorAll('.filter-chip').forEach(c => {
        c.classList.toggle('active', c.getAttribute('data-chunk-type') === 'ALL');
      });
      needRerender = true;
    }
    if (needRerender) {
      renderChunksList();
      targetCard = document.getElementById(`chunk-card-${chunkId}`);
    }
  }

  if (!targetCard) return;

  // 2. Highlight Chunk Card với hiệu ứng active rực rỡ
  document.querySelectorAll('.chunk-card').forEach(c => {
    c.classList.remove('active-chunk', 'dimmed-chunk');
  });
  targetCard.classList.add('active-chunk');

  // Làm mờ nhẹ các chunk khác để người dùng tập trung tuyệt đối vào chunk được móc
  document.querySelectorAll('.chunk-card').forEach(c => {
    if (c !== targetCard) {
      c.classList.add('dimmed-chunk');
    }
  });

  // 3. Cuộn container bên phải CHÍNH XÁC đến vị trí của targetCard
  const containerRect = container.getBoundingClientRect();
  const cardRect = targetCard.getBoundingClientRect();
  const relativeTop = cardRect.top - containerRect.top;
  const currentScrollTop = container.scrollTop;
  const targetScrollTop = currentScrollTop + relativeTop - (container.clientHeight / 2) + (cardRect.height / 2);

  container.scrollTo({
    top: Math.max(0, targetScrollTop),
    behavior: 'smooth'
  });

  try {
    targetCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
  } catch (e) {}

  // 4. Highlight flash nội dung
  const contentEl = targetCard.querySelector('.chunk-card-content');
  if (contentEl) {
    contentEl.classList.remove('content-highlight-flash');
    void contentEl.offsetWidth; // trigger reflow
    contentEl.classList.add('content-highlight-flash');
  }
}

/**
 * Xử lý khi người dùng Click vào một Chunk Card ở nửa phải hoặc Khung Chunk
 */
function handleChunkClick(chunkId, elementIds) {
  if (!chunkId) return;

  state.activeChunkId = chunkId;
  state.activeBoxId = null;

  // 1. Banner thông báo
  const banner = document.getElementById('active-inspect-banner');
  const info = document.getElementById('active-inspect-info');
  const boxCount = elementIds ? elementIds.length : document.querySelectorAll(`[data-chunk-id="${chunkId}"]`).length;
  if (banner && info) {
    banner.classList.remove('hidden');
    const cIdx = state.currentLayoutData?.chunks?.findIndex(c => c.chunk_id === chunkId);
    const chunkLabel = getCleanChunkLabel(chunkId, cIdx >= 0 ? cIdx : null);
    info.innerHTML = `
      <span>Đang soi: <strong>${escapeHtml(chunkLabel)}</strong> (${boxCount} boxes)</span>
    `;
  }

  // 2. Highlight Khung Chunk Bao Quanh (Chunk Boundary Box) ở nửa trái
  document.querySelectorAll('.chunk-boundary-box, .docx-chunk-boundary').forEach(cb => {
    cb.classList.remove('active-chunk-boundary');
  });
  const matchingBoundaries = document.querySelectorAll(`.chunk-boundary-box[data-chunk-id="${chunkId}"], .docx-chunk-boundary[data-chunk-id="${chunkId}"]`);
  matchingBoundaries.forEach(cb => {
    cb.classList.add('active-chunk-boundary');
  });

  // 3. Highlight tất cả các Box thuộc Chunk đó ở nửa trái
  document.querySelectorAll('.element-bbox, .docx-interactive-block').forEach(b => {
    b.classList.remove('active-box', 'chunk-linked-box');
  });

  const matchingBoxes = document.querySelectorAll(`.element-bbox[data-chunk-id="${chunkId}"], .docx-interactive-block[data-chunk-id="${chunkId}"]`);
  matchingBoxes.forEach(b => {
    b.classList.add('chunk-linked-box');
  });

  // 4. Cuộn mượt mà nửa trái tới Khung Chunk hoặc Box đầu tiên
  if (matchingBoundaries.length > 0) {
    matchingBoundaries[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
  } else if (matchingBoxes.length > 0) {
    matchingBoxes[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  // 5. MÓC ĐẾN CÙNG: Đồng bộ bên phải cũng phải móc và cuộn đến tận thẻ Chunk này!
  hookAndScrollRightPaneToChunk(chunkId);
}

/**
 * Kết xuất danh sách Chunks bên phải
 */
function renderChunksList() {
  const container = document.getElementById('chunks-list-container');
  if (!container) return;

  container.innerHTML = '';

  const chunks = (state.currentLayoutData && state.currentLayoutData.chunks) || [];
  const term = state.searchTerm.toLowerCase();
  const typeFilter = state.selectedChunkType;

  const filteredChunks = chunks.filter(c => {
    // Lọc theo từ khóa
    if (term) {
      const matchContent = (c.content || '').toLowerCase().includes(term);
      const matchTitle = (c.section_title || '').toLowerCase().includes(term);
      const matchBreadcrumb = (c.breadcrumb || '').toLowerCase().includes(term);
      if (!matchContent && !matchTitle && !matchBreadcrumb) return false;
    }
    // Lọc theo chunk type
    if (typeFilter !== 'ALL') {
      const ct = (c.chunk_type || '').toLowerCase();
      if (!ct.includes(typeFilter.toLowerCase())) return false;
    }
    return true;
  });

  // Cập nhật số lượng
  const counter = document.getElementById('chunk-counter');
  if (counter) {
    counter.textContent = `${filteredChunks.length} / ${chunks.length} chunks`;
  }

  if (filteredChunks.length === 0) {
    container.innerHTML = `
      <div style="text-align: center; padding: 40px 20px; color: var(--text-muted);">
        <div>Không tìm thấy Chunk phù hợp với bộ lọc</div>
      </div>
    `;
    return;
  }

  filteredChunks.forEach((chunk, idx) => {
    const card = createChunkCard(chunk, idx);
    container.appendChild(card);
  });
}

/**
 * Trích xuất nhãn ngắn gọn của Chunk, loại bỏ tiền tố tên file dài dòng
 * Ví dụ: vcb_pdf_24_bieu_phi_chuyen_tien_ngoai_te_digibank_pdf_tbl_0 -> Chunk #1 (tbl_0)
 * hoặc: vcb_docx_01_dieu_khoan_dieu_kien_chung_c3 -> Chunk #4 (c3)
 */
function getCleanChunkLabel(chunkId, idx = null) {
  if (!chunkId) return 'Chunk';
  if (idx !== null && idx !== undefined) {
    return `Chunk #${idx + 1}`;
  }
  const match = chunkId.match(/(?:_|\b)([a-zA-Z]+_\d+|\d+|c\d+)$/);
  return match ? `Chunk ${match[1]}` : (chunkId.length > 20 ? `...${chunkId.slice(-16)}` : chunkId);
}

/**
 * Trích xuất nhãn ngắn gọn của Element, loại bỏ tiền tố tên file
 * Ví dụ: vcb_pdf_24_bieu_phi_chuyen_tien_ngoai_te_digibank.pdf_p1_e3 -> p1_e3
 */
function getCleanElementId(elementId) {
  if (!elementId) return 'Element';
  const match = elementId.match(/(p\d+_e\d+|dom_\d+|e\d+)$/);
  if (match) return match[1];
  return elementId.length > 20 ? `...${elementId.slice(-16)}` : elementId;
}

/**
 * Làm sạch breadcrumb: Loại bỏ phần tên file / tên tài liệu ở cấp đầu tiên
 * vì thông tin này đã hiển thị trên thanh chọn tệp phía trên, tránh lặp lại ở mọi chunk.
 */
function cleanChunkBreadcrumb(rawBreadcrumb) {
  if (!rawBreadcrumb) return '';
  let bc = String(rawBreadcrumb).trim();
  const parts = bc.split(/\s*>\s*/);
  if (parts.length > 1) {
    const p0 = parts[0].toLowerCase();
    // Chỉ loại bỏ cấp đầu nếu nó thuần túy là tên file kỹ thuật
    if (p0.endsWith('.pdf') || p0.endsWith('.docx') || p0.startsWith('vcb_pdf_') || p0.startsWith('vcb_docx_') || /^vcb\s+(pdf|docx)\s+\d+$/i.test(parts[0])) {
      parts.shift();
      return parts.join(' > ');
    }
  }
  return bc;
}

/**
 * Loại bỏ tiền tố [Breadcrumbs: ...] lặp lại ở đầu nội dung,
 * và loại bỏ dòng tiêu đề (### Tiêu đề) lặp lại ở đầu nội dung nếu đã có trên breadcrumb.
 */
function cleanChunkContent(rawText, breadcrumbText = '', sectionTitle = '') {
  if (!rawText) return '';
  let text = String(rawText).trim();
  // 1. Loại bỏ toàn bộ [Breadcrumbs: ...], [Breadcrumb: ...] hoặc [Ngữ cảnh: ...]
  text = text.replace(/\[(?:Breadcrumbs?|Ngữ cảnh):\s*[\s\S]*?\]\s*/gi, '').trim();

  // 2. Loại bỏ tiêu đề đầu dòng bị lặp lại (### Tiêu đề, ## Tiêu đề, # Tiêu đề)
  const lines = text.split('\n');
  if (lines.length > 0) {
    const firstLine = lines[0].trim();
    const headingMatch = firstLine.match(/^#{1,6}\s*(.+)$/);
    if (headingMatch) {
      const headingText = headingMatch[1].trim().toLowerCase();
      const bcLower = (breadcrumbText || '').toLowerCase();
      const stLower = (sectionTitle || '').toLowerCase();

      // Nếu dòng heading đầu tiên lặp lại nội dung đã có trên breadcrumb hoặc section_title
      if (
        (stLower && (headingText === stLower || stLower.includes(headingText) || headingText.includes(stLower))) ||
        (bcLower && (bcLower.endsWith(headingText) || bcLower.includes(headingText)))
      ) {
        lines.shift();
        while (lines.length > 0 && !lines[0].trim()) {
          lines.shift();
        }
        text = lines.join('\n').trim();
      }
    }
  }

  return text.trim();
}

/**
 * Định dạng nội dung Chunk thành HTML đẹp mắt (hỗ trợ bullet points và tiêu đề đậm)
 */
function formatChunkContentHtml(rawText) {
  if (!rawText) return '';
  const lines = rawText.split('\n');
  const formattedLines = lines.map(line => {
    const trimmed = line.trim();
    if (!trimmed) return '<div style="height: 8px;"></div>';
    
    // Dòng gạch đầu dòng bullet
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('• ')) {
      const bulletContent = trimmed.slice(2).trim();
      return `<div class="content-bullet-line"><span class="bullet-dot">•</span><span>${escapeHtml(bulletContent)}</span></div>`;
    }
    
    // Tiêu đề in hoa / nhấn mạnh
    if (trimmed.length > 3 && trimmed === trimmed.toUpperCase() && /[A-ZÀ-Ỹ]/.test(trimmed)) {
      return `<div class="content-heading-line">${escapeHtml(trimmed)}</div>`;
    }
    
    return `<div class="content-normal-line">${escapeHtml(line)}</div>`;
  });
  return formattedLines.join('');
}

/**
 * Tạo thẻ DOM Chunk Card chuẩn giao diện hiện đại (Chunk Pill, Breadcrumb *, Content *)
 */
function createChunkCard(chunk, idx = null) {
  const card = document.createElement('div');
  card.className = 'chunk-card';
  card.id = `chunk-card-${chunk.chunk_id}`;
  card.setAttribute('data-chunk-id', chunk.chunk_id);

  const rawBc = (chunk.breadcrumb || chunk.section_title || '').trim();
  const breadcrumbText = cleanChunkBreadcrumb(rawBc) || rawBc;
  const displayContent = cleanChunkContent(chunk.content || chunk.raw_content || '', breadcrumbText, chunk.section_title);
  const wordCount = chunk.word_count || (displayContent ? displayContent.split(/\s+/).length : 0);
  const chunkNumber = (idx !== null && idx !== undefined) ? (idx + 1) : 1;

  card.innerHTML = `
    <div class="chunk-card-top-bar">
      <div class="chunk-badge-pill">
        <span>Chunk ${chunkNumber}</span>
        <button class="chunk-badge-btn" title="Chỉnh sửa (Chế độ xem)" tabindex="-1">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
          </svg>
        </button>
        <button class="chunk-badge-btn" title="Bỏ chọn" tabindex="-1">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      </div>
      <span class="chunk-word-count-tag">${wordCount} từ</span>
    </div>

    <div class="chunk-field-group">
      <div class="chunk-field-label">
        <span>Breadcrumb</span>
        <span class="required-star">*</span>
      </div>
      <div class="chunk-field-value chunk-breadcrumb-value">
        ${escapeHtml(breadcrumbText || 'Nội dung')}
      </div>
    </div>

    <div class="chunk-field-group">
      <div class="chunk-field-label">
        <span>Content</span>
        <span class="required-star">*</span>
      </div>
      <div class="chunk-field-value chunk-content-value chunk-card-content">
        ${formatChunkContentHtml(displayContent)}
      </div>
    </div>
  `;

  card.addEventListener('click', () => {
    handleChunkClick(chunk.chunk_id, chunk.element_ids);
  });

  return card;
}

/**
 * Cập nhật thông số thống kê trên Toolbar
 */
function updateHeaderStats() {
  const data = state.currentLayoutData;
  if (!data) return;

  const statPages = document.getElementById('stat-pages');
  const statElements = document.getElementById('stat-elements');
  const statChunks = document.getElementById('stat-chunks');
  const statAvgWords = document.getElementById('stat-avg-words');

  if (statPages) statPages.textContent = data.total_pages || (data.pages ? data.pages.length : 1);
  if (statElements) statElements.textContent = data.elements_count || 0;
  if (statChunks) statChunks.textContent = data.chunks_count || 0;

  if (statAvgWords && data.chunks && data.chunks.length > 0) {
    const totalWords = data.chunks.reduce((sum, c) => sum + (c.word_count || 0), 0);
    statAvgWords.textContent = Math.round(totalWords / data.chunks.length);
  }
}

/**
 * Bỏ chọn Element / Chunk đang soi
 */
function clearActiveSelection() {
  state.activeBoxId = null;
  state.activeChunkId = null;

  document.querySelectorAll('.element-bbox, .docx-interactive-block').forEach(b => {
    b.classList.remove('active-box', 'chunk-linked-box');
  });
  document.querySelectorAll('.chunk-boundary-box, .docx-chunk-boundary').forEach(cb => {
    cb.classList.remove('active-chunk-boundary');
  });
  document.querySelectorAll('.chunk-card').forEach(c => {
    c.classList.remove('active-chunk', 'dimmed-chunk');
  });

  const banner = document.getElementById('active-inspect-banner');
  if (banner) banner.classList.add('hidden');
}

/**
 * Cài đặt thanh kéo phân chia 2 nửa (Split Resizer)
 */
function setupSplitResizer() {
  const resizer = document.getElementById('split-resizer');
  const container = document.getElementById('main-split-container');
  const leftPane = document.getElementById('split-pane-left');
  if (!resizer || !container || !leftPane) return;

  let isDragging = false;

  resizer.addEventListener('mousedown', (e) => {
    isDragging = true;
    resizer.classList.add('is-dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });

  window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const containerRect = container.getBoundingClientRect();
    const newLeftWidth = e.clientX - containerRect.left;
    const minLeft = 320;
    const maxLeft = containerRect.width - 280;
    if (newLeftWidth >= minLeft && newLeftWidth <= maxLeft) {
      const percent = (newLeftWidth / containerRect.width) * 100;
      leftPane.style.flex = `0 0 ${percent}%`;
    }
  });

  window.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
      resizer.classList.remove('is-dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      // Tự động re-render vừa khung nhìn mới
      if (state.zoomMode === 'fit-width' || state.zoomMode === 'fit-page') {
        reRenderCurrentDocView();
      }
    }
  });

  // Các nút ratio ½ (50:50), ⅔ (62:38), ¾ (75:25)
  document.querySelectorAll('.btn-ratio').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.btn-ratio').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const ratio = parseFloat(btn.getAttribute('data-ratio') || '62');
      leftPane.style.flex = `0 0 ${ratio}%`;
      setTimeout(() => reRenderCurrentDocView(), 50);
    });
  });

  // Tự động điều chỉnh kích thước khi cửa sổ thay đổi
  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (state.zoomMode === 'fit-width' || state.zoomMode === 'fit-page') {
        reRenderCurrentDocView();
      }
    }, 150);
  });
}

function setActiveZoomTool(btnId) {
  document.querySelectorAll('.pane-tools .btn').forEach(b => {
    if (b.id.startsWith('btn-zoom-fit') || b.id === 'btn-zoom-reset') {
      b.classList.remove('active-tool');
    }
  });
  if (btnId) {
    document.getElementById(btnId)?.classList.add('active-tool');
  }
}

/**
 * Đăng ký sự kiện các nút điều khiển và bộ lọc
 */
function setupEventListeners() {
  // 1. Master select 108 tệp
  const masterSelect = document.getElementById('doc-master-select');
  masterSelect?.addEventListener('change', (e) => {
    const name = e.target.value;
    const idx = state.manifest.findIndex(m => m.name === name);
    if (idx !== -1) {
      selectDocumentByIndex(idx);
    }
  });

  // 2. Chuyển tệp trước / sau
  document.getElementById('btn-prev-doc')?.addEventListener('click', () => {
    if (state.currentDocIndex > 0) {
      selectDocumentByIndex(state.currentDocIndex - 1);
    }
  });
  document.getElementById('btn-next-doc')?.addEventListener('click', () => {
    if (state.currentDocIndex < state.manifest.length - 1) {
      selectDocumentByIndex(state.currentDocIndex + 1);
    }
  });

  // Phím tắt bàn phím
  window.addEventListener('keydown', (e) => {
    if (e.altKey && e.key === 'ArrowLeft') {
      document.getElementById('btn-prev-doc')?.click();
    } else if (e.altKey && e.key === 'ArrowRight') {
      document.getElementById('btn-next-doc')?.click();
    }
  });

  // 3. Format Tabs (Tất cả, PDF, DOCX)
  document.querySelectorAll('.doc-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.doc-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.docFilterFormat = tab.getAttribute('data-filter') || 'ALL';
      populateMasterDocSelect();
    });
  });

  // 4. Nhảy trang nhanh
  const pageJumpSelect = document.getElementById('page-jump-select');
  pageJumpSelect?.addEventListener('change', (e) => {
    const targetId = e.target.value;
    const targetEl = document.getElementById(targetId);
    if (targetEl) {
      targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });

  // 5. Nút Zoom & Fit Controls
  document.getElementById('btn-zoom-fit-width')?.addEventListener('click', () => {
    setActiveZoomTool('btn-zoom-fit-width');
    state.zoomMode = 'fit-width';
    reRenderCurrentDocView();
  });

  document.getElementById('btn-zoom-fit-page')?.addEventListener('click', () => {
    setActiveZoomTool('btn-zoom-fit-page');
    state.zoomMode = 'fit-page';
    reRenderCurrentDocView();
  });

  document.getElementById('btn-zoom-reset')?.addEventListener('click', () => {
    setActiveZoomTool('btn-zoom-reset');
    state.zoomMode = 'custom';
    state.zoomLevel = 1.0;
    reRenderCurrentDocView();
  });

  document.getElementById('btn-zoom-in')?.addEventListener('click', () => {
    setActiveZoomTool(null);
    state.zoomMode = 'custom';
    state.zoomLevel = Math.min(2.5, state.zoomLevel + 0.15);
    reRenderCurrentDocView();
  });

  document.getElementById('btn-zoom-out')?.addEventListener('click', () => {
    setActiveZoomTool(null);
    state.zoomMode = 'custom';
    state.zoomLevel = Math.max(0.35, state.zoomLevel - 0.15);
    reRenderCurrentDocView();
  });

  // 6. Điều khiển chế độ hiển thị tách biệt: Box Element & Khung Chunk & File Gốc
  document.getElementById('btn-toggle-boxes')?.addEventListener('click', () => {
    state.showBoxes = !state.showBoxes;
    applyViewLayers();
  });

  document.getElementById('btn-toggle-chunks')?.addEventListener('click', () => {
    state.showChunkBoundaries = !state.showChunkBoundaries;
    applyViewLayers();
  });

  document.getElementById('btn-toggle-raw-bg')?.addEventListener('click', () => {
    state.showRawBg = !state.showRawBg;
    applyViewLayers();
  });


  // 8. Tìm kiếm Chunks
  document.getElementById('chunk-search-input')?.addEventListener('input', (e) => {
    state.searchTerm = e.target.value.trim();
    renderChunksList();
  });

  // 9. Lọc loại Chunk
  document.querySelectorAll('.filter-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      state.selectedChunkType = chip.getAttribute('data-chunk-type') || 'ALL';
      renderChunksList();
    });
  });

  // 10. Chọn tệp ngoài từ máy tính
  const btnPickRawFile = document.getElementById('btn-pick-raw-file');
  const rawFileInput = document.getElementById('raw-file-upload-input');
  btnPickRawFile?.addEventListener('click', () => rawFileInput?.click());
  rawFileInput?.addEventListener('change', async (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) {
      const ab = await file.arrayBuffer();
      if (file.name.endsWith('.pdf')) {
        const loadingTask = pdfjsLib.getDocument({ data: ab });
        const pdfDoc = await loadingTask.promise;
        state.rawPdfDoc = pdfDoc;
        const simLayout = {
          filename: file.name,
          format: 'pdf',
          title: file.name,
          total_pages: pdfDoc.numPages,
          pages: Array.from({ length: pdfDoc.numPages }, (_, i) => ({ page_num: i + 1, elements: [] })),
          chunks: []
        };
        await renderPdfCanvasView(pdfDoc, simLayout);
      }
    }
  });

  // 11. Theme toggle
  document.getElementById('btn-theme-toggle')?.addEventListener('click', () => {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', state.theme);
    const btn = document.getElementById('btn-theme-toggle');
    if (btn) btn.textContent = state.theme === 'dark' ? 'Tối' : 'Sáng';
  });

  // 12. Bỏ chọn inspect
  document.getElementById('btn-clear-inspect')?.addEventListener('click', clearActiveSelection);
}

/**
 * Áp dụng trạng thái hiển thị của các lớp Box Element, Khung Chunk và File Gốc
 */
function applyViewLayers() {
  const scrollContainer = document.getElementById('page-viewer-scroll');
  if (scrollContainer) {
    scrollContainer.classList.toggle('hide-chunk-boundaries', !state.showChunkBoundaries);
    scrollContainer.classList.toggle('hide-element-boxes', !state.showBoxes);
  }

  document.querySelectorAll('.element-bbox, .element-box').forEach(b => {
    b.style.display = state.showBoxes ? '' : 'none';
    b.style.pointerEvents = state.showBoxes ? 'auto' : 'none';
  });

  document.querySelectorAll('.chunk-boundary-box, .docx-chunk-boundary').forEach(cb => {
    cb.style.display = state.showChunkBoundaries ? '' : 'none';
    cb.style.pointerEvents = state.showChunkBoundaries ? 'auto' : 'none';
  });

  document.querySelectorAll('.pdf-raw-canvas').forEach(c => {
    c.style.opacity = state.showRawBg ? '1' : '0.06';
  });

  // Đồng bộ trạng thái active của các nút hiển thị
  const btnBoxes = document.getElementById('btn-toggle-boxes');
  const btnChunks = document.getElementById('btn-toggle-chunks');
  const btnRawBg = document.getElementById('btn-toggle-raw-bg');

  if (btnBoxes) btnBoxes.classList.toggle('active', !!state.showBoxes);
  if (btnChunks) btnChunks.classList.toggle('active', !!state.showChunkBoundaries);
  if (btnRawBg) btnRawBg.classList.toggle('active', !!state.showRawBg);

}

function updateBoxVisibility() {
  document.querySelectorAll('.element-bbox').forEach(b => {
    let elType = 'body';
    for (const [t, active] of Object.entries(state.filterTypes)) {
      if (b.classList.contains(`bbox-${t}`)) {
        elType = t;
        break;
      }
    }
    b.classList.toggle('filtered-out', !state.filterTypes[elType]);
  });
}

function showTooltip(e, el, conf) {
  const tt = document.getElementById('box-tooltip');
  if (!tt) return;

  tt.innerHTML = `
    <div style="font-weight:700; color:${conf.border}; margin-bottom:4px; display:flex; justify-content:space-between;">
      <span>[${conf.label}]</span>
      <span style="font-size:10px; color:#94a3b8;">Trang ${el.page_num}</span>
    </div>
    <div style="font-size:11px; margin-bottom:4px; max-height:80px; overflow:hidden;">${escapeHtml(el.text || '')}</div>
    <div style="font-size:10px; color:#64748b; font-family:monospace;">
      Box: [${(el.box || []).join(', ')}]
    </div>
  `;
  tt.style.display = 'block';
  tt.style.left = `${e.clientX + 14}px`;
  tt.style.top = `${e.clientY + 14}px`;
}

function hideTooltip() {
  const tt = document.getElementById('box-tooltip');
  if (tt) tt.style.display = 'none';
}

function setupDragAndDrop() {
  const overlay = document.getElementById('drop-zone-overlay');
  window.addEventListener('dragenter', (e) => {
    e.preventDefault();
    overlay?.classList.add('active');
  });
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('dragleave', (e) => {
    if (e.relatedTarget === null) overlay?.classList.remove('active');
  });
  window.addEventListener('drop', async (e) => {
    e.preventDefault();
    overlay?.classList.remove('active');
  });
}

function initUI() {
  applyViewLayers();
}

function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
