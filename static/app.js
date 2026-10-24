/**
 * AI Banking Assistant - Frontend Client
 * Kết nối liên hoàn 7 Nodes RAG và quản lý Folder-to-Folder Data Flow
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const chatMessages = document.getElementById("chatMessages");
    const modelBadge = document.getElementById("modelBadge");
    const chunksCountBadge = document.getElementById("chunksCountBadge");
    const sampleQueries = document.getElementById("sampleQueries");

    // Modal elements
    const nodeDetailModal = document.getElementById("nodeDetailModal");
    const closeNodeDetailBtn = document.getElementById("closeNodeDetailBtn");
    const nodeDetailTitle = document.getElementById("nodeDetailTitle");
    const nodeDetailSub = document.getElementById("nodeDetailSub");
    const nodeInputPath = document.getElementById("nodeInputPath");
    const nodeOutputPath = document.getElementById("nodeOutputPath");
    const nodeInputContent = document.getElementById("nodeInputContent");
    const nodeOutputContent = document.getElementById("nodeOutputContent");
    const ioTabs = document.querySelectorAll(".io-tab");
    const ioPanels = document.querySelectorAll(".io-panel");

    // Folder Explorer elements
    const folderExplorerModal = document.getElementById("folderExplorerModal");
    const openFolderModalBtn = document.getElementById("openFolderModalBtn");
    const closeFolderExplorerBtn = document.getElementById("closeFolderExplorerBtn");
    const nodesSidebar = document.getElementById("nodesSidebar");
    const nodeFolderFiles = document.getElementById("nodeFolderFiles");
    const triggerPipelineBtn = document.getElementById("triggerPipelineBtn");

    // File Viewer elements
    const fileViewerModal = document.getElementById("fileViewerModal");
    const closeFileViewerBtn = document.getElementById("closeFileViewerBtn");
    const viewerFileName = document.getElementById("viewerFileName");
    const viewerFilePath = document.getElementById("viewerFilePath");
    const viewerFileContent = document.getElementById("viewerFileContent");

    // Toast
    const toast = document.getElementById("toastNotification");

    let currentNodesData = null;
    let selectedNodeName = null;

    // 1. Tải cấu hình ban đầu
    async function fetchConfig() {
        try {
            const res = await fetch("/api/config");
            const data = await res.json();
            if (modelBadge && data.default_model) {
                modelBadge.textContent = data.default_model;
            }
            if (chunksCountBadge) {
                chunksCountBadge.textContent = data.points_count || 0;
            }
        } catch (err) {
            console.error("Không thể tải cấu hình:", err);
        }
    }
    fetchConfig();

    // 2. Hiển thị thông báo Toast
    function showToast(message, duration = 3000) {
        if (!toast) return;
        toast.textContent = message;
        toast.style.display = "block";
        setTimeout(() => {
            toast.style.display = "none";
        }, duration);
    }

    // 3. Render tin nhắn chat
    function appendMessage(sender, text, nodesTrace = null) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender}-message`;

        const bubble = document.createElement("div");
        bubble.className = "bubble";

        // Định dạng Markdown cơ bản (không dùng regex phức tạp)
        let formatted = text
            .split("\n\n").join("<br><br>")
            .split("\n").join("<br>");

        bubble.innerHTML = formatted;

        // Nếu là bot và có dữ liệu trace 7 nodes, render thanh điều khiển tracker
        if (sender === "bot" && nodesTrace) {
            const trackerContainer = document.createElement("div");
            trackerContainer.className = "node-tracker-container";
            trackerContainer.innerHTML = `
                <div class="node-tracker-title">
                    <span>🔍 Luồng dữ liệu qua 7 Nodes (Bấm từng node để xem Input & Output):</span>
                </div>
                <div class="node-tracker-pills">
                    <div class="node-pill" data-node="01_chunking">
                        <span class="node-num">N1</span>
                        <span class="node-name">Chunking</span>
                        <span class="node-status">✓ DLA</span>
                    </div>
                    <div class="node-pill" data-node="02_embedding">
                        <span class="node-num">N2</span>
                        <span class="node-name">Embed</span>
                        <span class="node-status">1024-d</span>
                    </div>
                    <div class="node-pill" data-node="03_vectordb">
                        <span class="node-num">N3</span>
                        <span class="node-name">Index</span>
                        <span class="node-status">HNSW/BM25</span>
                    </div>
                    <div class="node-pill" data-node="04_query_parser">
                        <span class="node-num">N4</span>
                        <span class="node-name">Parser</span>
                        <span class="node-status">${nodesTrace["04_query_parser"]?.latency_ms || 0}ms</span>
                    </div>
                    <div class="node-pill" data-node="05_retrieval">
                        <span class="node-num">N5</span>
                        <span class="node-name">Retrieval</span>
                        <span class="node-status">Top-${nodesTrace["05_retrieval"]?.candidates_count || 5}</span>
                    </div>
                    <div class="node-pill" data-node="06_reranking">
                        <span class="node-num">N6</span>
                        <span class="node-name">Rerank</span>
                        <span class="node-status">Top-3</span>
                    </div>
                    <div class="node-pill" data-node="07_generator">
                        <span class="node-num">N7</span>
                        <span class="node-name">Generate</span>
                        <span class="node-status">${nodesTrace["07_generator"]?.latency_ms || 0}ms</span>
                    </div>
                </div>
            `;

            // Gắn sự kiện click cho từng node pill
            trackerContainer.querySelectorAll(".node-pill").forEach(pill => {
                pill.addEventListener("click", () => {
                    const nodeKey = pill.getAttribute("data-node");
                    openNodeDetail(nodeKey, nodesTrace[nodeKey]);
                });
            });

            bubble.appendChild(trackerContainer);
        }

        msgDiv.appendChild(bubble);
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return msgDiv;
    }

    // 4. Mở modal chi tiết Input & Output của một Node
    function openNodeDetail(nodeKey, nodeData) {
        if (!nodeData) return;
        nodeDetailTitle.textContent = nodeData.name || `Chi Tiết ${nodeKey}`;
        nodeDetailSub.textContent = `Thư mục lưu vết: ${nodeData.input_folder} và ${nodeData.output_folder}`;

        nodeInputPath.textContent = `📁 Input Folder: ${nodeData.input_folder}`;
        nodeOutputPath.textContent = `📁 Output Folder: ${nodeData.output_folder}`;

        // Định dạng nội dung hiển thị
        let inContent = "";
        let outContent = "";

        if (nodeKey === "01_chunking") {
            inContent = JSON.stringify({ "description": "Tài liệu thô từ " + nodeData.input_folder, "types": [".docx", ".pdf"] }, null, 2);
            outContent = JSON.stringify({ "status": nodeData.status, "total_chunks_in_store": nodeData.total_chunks_in_store, "breadcrumbs_enabled": true }, null, 2);
        } else if (nodeKey === "02_embedding") {
            inContent = JSON.stringify({ "input_query_or_chunks": "Text strings to embed", "model": nodeData.model }, null, 2);
            outContent = JSON.stringify({ "dimension": nodeData.dimension, "sample_values": nodeData.vector_sample, "latency_ms": nodeData.latency_ms }, null, 2);
        } else if (nodeKey === "03_vectordb") {
            inContent = JSON.stringify({ "embedded_chunks": "Dense vectors 1024-dim + full text" }, null, 2);
            outContent = JSON.stringify({ "qdrant_collection": nodeData.qdrant_collection, "qdrant_points_count": nodeData.qdrant_points_count, "bm25_indexed_chunks": nodeData.bm25_indexed_chunks }, null, 2);
        } else if (nodeKey === "04_query_parser") {
            inContent = JSON.stringify({ "raw_query": nodeData.input_query }, null, 2);
            outContent = JSON.stringify({ "normalized_query": nodeData.normalized_query, "keywords": nodeData.keywords, "entities": nodeData.entities, "latency_ms": nodeData.latency_ms }, null, 2);
        } else if (nodeKey === "05_retrieval") {
            inContent = JSON.stringify({ "query_vector": "1024-dim vector", "bm25_query": "Keyword tokens", "rrf_k": 60 }, null, 2);
            outContent = JSON.stringify(nodeData.candidates, null, 2);
        } else if (nodeKey === "06_reranking") {
            inContent = JSON.stringify({ "model": nodeData.model, "candidates_count": 5 }, null, 2);
            outContent = JSON.stringify(nodeData.top_chunks, null, 2);
        } else if (nodeKey === "07_generator") {
            inContent = JSON.stringify({ "top_context_chunks": "Top 3 Cross-Encoder verified chunks", "model": nodeData.model }, null, 2);
            outContent = JSON.stringify({ "final_answer": nodeData.answer, "citations": nodeData.citations, "latency_ms": nodeData.latency_ms }, null, 2);
        }

        nodeInputContent.textContent = inContent;
        nodeOutputContent.textContent = outContent;

        // Reset về tab Input
        ioTabs.forEach(t => t.classList.remove("active"));
        ioPanels.forEach(p => { p.classList.remove("active"); p.style.display = "none"; });
        const inputTab = document.querySelector('.io-tab[data-tab="tab-input"]');
        const inputPanel = document.getElementById("tab-input");
        if (inputTab && inputPanel) {
            inputTab.classList.add("active");
            inputPanel.classList.add("active");
            inputPanel.style.display = "block";
        }

        nodeDetailModal.style.display = "flex";
    }

    // Đóng Modal Node Detail
    closeNodeDetailBtn.addEventListener("click", () => {
        nodeDetailModal.style.display = "none";
    });

    // Chuyển Tab Input / Output
    ioTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            ioTabs.forEach(t => t.classList.remove("active"));
            ioPanels.forEach(p => { p.classList.remove("active"); p.style.display = "none"; });

            tab.classList.add("active");
            const targetId = tab.getAttribute("data-tab");
            const targetPanel = document.getElementById(targetId);
            if (targetPanel) {
                targetPanel.classList.add("active");
                targetPanel.style.display = "block";
            }
        });
    });

    // 5. Gửi câu hỏi Chat
    async function submitQuery(queryText) {
        if (!queryText) return;
        appendMessage("user", queryText);

        userInput.value = "";
        userInput.disabled = true;
        sendBtn.disabled = true;

        const loadingDiv = document.createElement("div");
        loadingDiv.className = "message bot-message";
        loadingDiv.innerHTML = `<div class="bubble"><span style="color: #38bdf8;">⏳ Đang định tuyến qua 7 Nodes (Chunking ➔ Embedding ➔ Retrieval ➔ Rerank ➔ LLM)...</span></div>`;
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: queryText })
            });

            loadingDiv.remove();

            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: "Lỗi kết nối máy chủ" }));
                appendMessage("bot", `❌ Lỗi: ${err.detail}`);
            } else {
                const data = await res.json();
                appendMessage("bot", data.reply, data.nodes);
                fetchConfig();
            }
        } catch (e) {
            loadingDiv.remove();
            appendMessage("bot", `❌ Lỗi kết nối: ${e.message}`);
        } finally {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    }

    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        submitQuery(text);
    });

    // Click câu hỏi mẫu
    if (sampleQueries) {
        sampleQueries.addEventListener("click", (e) => {
            const chip = e.target.closest(".query-chip");
            if (chip) {
                const q = chip.getAttribute("data-q");
                submitQuery(q);
            }
        });
    }

    // 6. Quản Lý Folder Nodes Explorer
    async function loadNodesExplorer() {
        nodesSidebar.innerHTML = "<div class='empty-state'>Đang quét 7 thư mục node...</div>";
        try {
            const res = await fetch("/api/nodes");
            currentNodesData = await res.json();

            nodesSidebar.innerHTML = "";
            const nodeKeys = Object.keys(currentNodesData);

            nodeKeys.forEach((nodeKey, idx) => {
                const nodeInfo = currentNodesData[nodeKey];
                const totalFiles = (nodeInfo.input_files.length) + (nodeInfo.output_files.length);
                const btn = document.createElement("button");
                btn.className = `node-item-btn ${idx === 0 ? "active" : ""}`;
                btn.innerHTML = `
                    <span>Node ${idx + 1}: ${nodeKey}</span>
                    <span class="file-count-badge">${totalFiles} tệp</span>
                `;
                btn.addEventListener("click", () => {
                    document.querySelectorAll(".node-item-btn").forEach(b => b.classList.remove("active"));
                    btn.classList.add("active");
                    renderNodeFolderFiles(nodeKey);
                });
                nodesSidebar.appendChild(btn);
            });

            if (nodeKeys.length > 0) {
                renderNodeFolderFiles(nodeKeys[0]);
            }
        } catch (e) {
            nodesSidebar.innerHTML = `<div class='empty-state' style='color:#f87171;'>Lỗi tải thư mục: ${e.message}</div>`;
        }
    }

    function renderNodeFolderFiles(nodeKey) {
        selectedNodeName = nodeKey;
        const nodeInfo = currentNodesData[nodeKey];
        if (!nodeInfo) return;

        function buildFileList(files, ioType) {
            if (!files || files.length === 0) {
                return `<div class="empty-state">Thư mục trống (Chưa có tệp tin)</div>`;
            }
            return files.map(f => `
                <div class="file-row">
                    <div class="file-info-col">
                        <span class="file-name-text">${f.name}</span>
                        <span class="file-meta-text">${(f.size_bytes / 1024).toFixed(1)} KB • ${f.modified}</span>
                    </div>
                    <button class="btn-view-file" data-node="${nodeKey}" data-io="${ioType}" data-file="${f.name}">
                        👁️ Xem File
                    </button>
                </div>
            `).join("");
        }

        nodeFolderFiles.innerHTML = `
            <div class="io-folder-columns">
                <div class="io-folder-card">
                    <h4>📥 Thư Mục Đầu Vào (input/)</h4>
                    <div class="folder-path-sub">${nodeInfo.input_dir}</div>
                    <div class="file-list">
                        ${buildFileList(nodeInfo.input_files, "input")}
                    </div>
                </div>
                <div class="io-folder-card">
                    <h4>📤 Thư Mục Đầu Ra (output/)</h4>
                    <div class="folder-path-sub">${nodeInfo.output_dir}</div>
                    <div class="file-list">
                        ${buildFileList(nodeInfo.output_files, "output")}
                    </div>
                </div>
            </div>
        `;

        // Gắn sự kiện xem file
        nodeFolderFiles.querySelectorAll(".btn-view-file").forEach(btn => {
            btn.addEventListener("click", () => {
                const node = btn.getAttribute("data-node");
                const io = btn.getAttribute("data-io");
                const file = btn.getAttribute("data-file");
                viewFileContent(node, io, file);
            });
        });
    }

    // Xem nội dung file
    async function viewFileContent(nodeName, ioType, fileName) {
        viewerFileName.textContent = fileName;
        viewerFilePath.textContent = `rag_banking/${nodeName}/${ioType}/${fileName}`;
        viewerFileContent.textContent = "Đang tải nội dung tệp...";
        fileViewerModal.style.display = "flex";

        try {
            const res = await fetch(`/api/nodes/${nodeName}/${ioType}/${fileName}`);
            if (!res.ok) {
                viewerFileContent.textContent = "Lỗi khi đọc tệp từ máy chủ.";
                return;
            }
            const data = await res.json();
            if (data.is_json) {
                viewerFileContent.textContent = JSON.stringify(data.content, null, 2);
            } else {
                viewerFileContent.textContent = data.content;
            }
        } catch (e) {
            viewerFileContent.textContent = `Lỗi tải tệp: ${e.message}`;
        }
    }

    // Sự kiện mở/đóng Folder Explorer
    openFolderModalBtn.addEventListener("click", () => {
        folderExplorerModal.style.display = "flex";
        loadNodesExplorer();
    });

    closeFolderExplorerBtn.addEventListener("click", () => {
        folderExplorerModal.style.display = "none";
    });

    closeFileViewerBtn.addEventListener("click", () => {
        fileViewerModal.style.display = "none";
    });

    // 7. Kích hoạt chạy toàn bộ Pipeline từ Folders
    triggerPipelineBtn.addEventListener("click", async () => {
        triggerPipelineBtn.disabled = true;
        triggerPipelineBtn.textContent = "⏳ Đang chạy...";
        showToast("🚀 Đang khởi động đường ống liên hoàn 7 Nodes từ các thư mục input...", 4000);

        try {
            const res = await fetch("/api/pipeline/run-folders", { method: "POST" });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: "Lỗi chạy pipeline" }));
                alert(`Lỗi thực thi: ${err.detail}`);
            } else {
                const summary = await res.json();
                showToast(`🎉 Đã xử lý xong ${summary.total_chunks} chunks & ${summary.total_queries} queries qua 7 nodes!`, 4000);
                fetchConfig();
                if (folderExplorerModal.style.display === "flex") {
                    loadNodesExplorer();
                }
            }
        } catch (e) {
            alert(`Lỗi kết nối: ${e.message}`);
        } finally {
            triggerPipelineBtn.disabled = false;
            triggerPipelineBtn.textContent = "▶️ Chạy Pipeline Từ Folder";
        }
    });

    // Đóng modal khi click ra ngoài overlay
    window.addEventListener("click", (e) => {
        if (e.target === nodeDetailModal) nodeDetailModal.style.display = "none";
        if (e.target === folderExplorerModal) folderExplorerModal.style.display = "none";
        if (e.target === fileViewerModal) fileViewerModal.style.display = "none";
    });
});
