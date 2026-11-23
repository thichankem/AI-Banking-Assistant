/**
 * A-Bank web chat
 * Tuân thủ 100%: 0% Regex Law & Zero Ternary Law.
 */

document.addEventListener("DOMContentLoaded", () => {
    const chatContainer = document.getElementById("chatContainer");
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const chatMessages = document.getElementById("chatMessages");

    // Stepper & Live DAG DOM Elements
    const nodeStepper = document.getElementById("nodeStepper");
    const stepperDesc = document.getElementById("stepperDesc");
    const agentActions = document.getElementById("agentActions");
    const agentPlan = document.getElementById("agentPlan");
    const dagLatencyBadge = document.getElementById("dagLatencyBadge");
    const dagModelBadge = document.getElementById("dagModelBadge");
    const dagTokenBadge = document.getElementById("dagTokenBadge");
    const dagAuditBadge = document.getElementById("dagAuditBadge");
    const dagGraphTrack = document.getElementById("dagGraphTrack");
    const dagNodeInspector = document.getElementById("dagNodeInspector");

    // Multi-Session Sidebar DOM Elements
    const sessionSidebar = document.getElementById("sessionSidebar");
    const sidebarToggleBtn = document.getElementById("sidebarToggleBtn");
    const newSessionBtn = document.getElementById("newSessionBtn");
    const sessionSearchInput = document.getElementById("sessionSearchInput");
    const sessionCountBadge = document.getElementById("sessionCountBadge");
    const sessionList = document.getElementById("sessionList");

    // Attachment & Voice DOM Elements
    const chatComposer = document.getElementById("chatComposer");
    const composerStatusBanner = document.getElementById("composerStatusBanner");
    const attachmentPreviewBar = document.getElementById("attachmentPreviewBar");
    const imageUploadInput = document.getElementById("imageUploadInput");
    const attachImageBtn = document.getElementById("attachImageBtn");
    const voiceInputBtn = document.getElementById("voiceInputBtn");

    function newSessionId() {
        return Array.from(window.crypto.getRandomValues(new Uint32Array(4)), value => value.toString(16).padStart(8, "0")).join("");
    }

    const sessionKey = "banking-agent-session";
    const storedSessionBeforeLoad = sessionStorage.getItem(sessionKey);
    let currentSessionId = storedSessionBeforeLoad || newSessionId();
    sessionStorage.setItem(sessionKey, currentSessionId);
    let pendingCard = null;
    let busy = false;

    const MAX_ATTACHMENTS = 3;
    const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
    const ALLOWED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/pjpeg", "image/x-png", "image/webp", "image/gif"];
    const ALLOWED_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".gif"];
    let pendingAttachments = [];
    let activeImageReadPromise = Promise.resolve();
    let statusBannerTimer = null;

    let speechRecognition = null;
    let isRecording = false;
    let voiceBaseText = "";
    let voiceCapturedText = "";

    if (window.ABankDebugInspector) {
        window.ABankDebugInspector.init({ sessionId: currentSessionId });
    }

    function exitWelcomeState() {
        if (chatContainer) {
            chatContainer.classList.remove("is-empty");
        }
    }

    function enterWelcomeState() {
        if (chatContainer) {
            chatContainer.classList.add("is-empty");
        }
    }

    // Thoát ký tự HTML an toàn (0% Regex)
    function escapeHtml(str) {
        if (!str) {
            return "";
        }
        return str
            .split("&").join("&amp;")
            .split("<").join("&lt;")
            .split(">").join("&gt;")
            .split('"').join("&quot;")
            .split("'").join("&#039;");
    }

    function isDigitsOnly(str) {
        if (!str || str.length === 0) {
            return false;
        }
        for (let i = 0; i < str.length; i++) {
            const ch = str.charAt(i);
            if (ch < "0" || ch > "9") {
                return false;
            }
        }
        return true;
    }

    // Chuyển các nhãn [Nguồn X] hoặc [Nguồn 1, 2] thành nút bấm xem chi tiết (0% Regex, Zero Ternary)
    function linkifyCitationMarkers(safeHtml, citationByIndex) {
        if (!safeHtml || !citationByIndex || citationByIndex.size === 0) {
            return safeHtml;
        }
        let out = [];
        let cursor = 0;
        while (cursor < safeHtml.length) {
            const openIdx = safeHtml.indexOf("[Nguồn", cursor);
            if (openIdx < 0) {
                out.push(safeHtml.slice(cursor));
                break;
            }
            const lastTagOpen = safeHtml.lastIndexOf("<", openIdx);
            const lastTagClose = safeHtml.lastIndexOf(">", openIdx);
            if (lastTagOpen > lastTagClose) {
                out.push(safeHtml.slice(cursor, openIdx + 1));
                cursor = openIdx + 1;
                continue;
            }
            const closeIdx = safeHtml.indexOf("]", openIdx);
            if (closeIdx < 0 || closeIdx - openIdx > 80) {
                out.push(safeHtml.slice(cursor, openIdx + 1));
                cursor = openIdx + 1;
                continue;
            }
            const inner = safeHtml.slice(openIdx + 1, closeIdx);
            const normalizedParts = inner.split(";").join(",").split(",");
            const buttons = [];
            let allValid = true;
            for (let p = 0; p < normalizedParts.length; p++) {
                let token = normalizedParts[p].trim();
                if (!token) {
                    continue;
                }
                if (token.startsWith("Nguồn ")) {
                    token = token.slice(6).trim();
                } else if (token.startsWith("Nguồn")) {
                    token = token.slice(5).trim();
                } else if (p === 0) {
                    allValid = false;
                    break;
                }
                if (token.indexOf(" ") >= 0) {
                    token = token.split(" ")[0].trim();
                }
                if (!isDigitsOnly(token)) {
                    allValid = false;
                    break;
                }
                const num = Number(token);
                if (!citationByIndex.has(num)) {
                    allValid = false;
                    break;
                }
                buttons.push(
                    '<button type="button" class="inline-citation-btn" data-cite-idx="'
                    + num
                    + '" title="Xem đoạn gốc Nguồn '
                    + num
                    + '">[Nguồn '
                    + num
                    + ']</button>'
                );
            }
            if (allValid && buttons.length > 0) {
                out.push(safeHtml.slice(cursor, openIdx));
                out.push(buttons.join(" "));
                cursor = closeIdx + 1;
            } else {
                out.push(safeHtml.slice(cursor, openIdx + 1));
                cursor = openIdx + 1;
            }
        }
        return out.join("");
    }

    // Xử lý in đậm, code inline và nút trích dẫn (0% Regex, Zero Ternary)
    function renderInlineFormatting(str, citationByIndex = null) {
        let safe = escapeHtml(str);
        safe = safe
            .split("&lt;br&gt;").join("<br>")
            .split("&lt;br/&gt;").join("<br>")
            .split("&lt;br /&gt;").join("<br>")
            .split("&lt;BR&gt;").join("<br>")
            .split("&lt;BR/&gt;").join("<br>")
            .split("&lt;BR /&gt;").join("<br>");

        let boldParts = safe.split("**");
        if (boldParts.length > 1) {
            let boldOut = [];
            for (let i = 0; i < boldParts.length; i++) {
                if (i % 2 === 1) {
                    boldOut.push("<strong>" + boldParts[i] + "</strong>");
                } else {
                    boldOut.push(boldParts[i]);
                }
            }
            safe = boldOut.join("");
        }

        let codeParts = safe.split("`");
        if (codeParts.length > 1) {
            let codeOut = [];
            for (let i = 0; i < codeParts.length; i++) {
                if (i % 2 === 1) {
                    codeOut.push("<code>" + codeParts[i] + "</code>");
                } else {
                    codeOut.push(codeParts[i]);
                }
            }
            safe = codeOut.join("");
        }

        if (citationByIndex && citationByIndex.size > 0) {
            safe = linkifyCitationMarkers(safe, citationByIndex);
        }

        return safe;
    }

    // Parse Markdown (0% Regex, Zero Ternary)
    function parseMarkdown(text, citationByIndex = null) {
        if (!text) {
            return "";
        }
        const lines = text.split("\n");
        let htmlParts = [];
        let inList = false;
        let inTable = false;
        let tableRows = [];

        function flushTable() {
            if (!inTable) {
                return;
            }
            if (tableRows.length === 0) {
                inTable = false;
                tableRows = [];
                return;
            }
            let tblHtml = ['<div class="table-wrap"><table>'];
            let startIdx = 0;

            if (tableRows.length >= 2) {
                tblHtml.push('<thead><tr>');
                for (let c of tableRows[0]) {
                    tblHtml.push("<th>" + renderInlineFormatting(c, citationByIndex) + "</th>");
                }
                tblHtml.push('</tr></thead>');
                startIdx = 2;
            }

            tblHtml.push('<tbody>');
            for (let r = startIdx; r < tableRows.length; r++) {
                tblHtml.push('<tr>');
                for (let c of tableRows[r]) {
                    tblHtml.push("<td>" + renderInlineFormatting(c, citationByIndex) + "</td>");
                }
                tblHtml.push('</tr>');
            }
            tblHtml.push('</tbody></table></div>');
            htmlParts.push(tblHtml.join(""));
            inTable = false;
            tableRows = [];
        }

        for (let i = 0; i < lines.length; i++) {
            let line = lines[i];
            let trimmed = line.trim();

            if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
                if (inList) {
                    htmlParts.push("</ul>");
                    inList = false;
                }
                let rawCells = trimmed.substring(1, trimmed.length - 1).split("|");
                let cells = [];
                for (let cell of rawCells) {
                    cells.push(cell.trim());
                }
                inTable = true;
                tableRows.push(cells);
                continue;
            } else {
                if (inTable) {
                    flushTable();
                }
            }

            if (trimmed.startsWith("### ")) {
                if (inList) {
                    htmlParts.push("</ul>");
                    inList = false;
                }
                htmlParts.push("<h4>" + renderInlineFormatting(trimmed.substring(4), citationByIndex) + "</h4>");
                continue;
            }
            if (trimmed.startsWith("## ")) {
                if (inList) {
                    htmlParts.push("</ul>");
                    inList = false;
                }
                htmlParts.push("<h3>" + renderInlineFormatting(trimmed.substring(3), citationByIndex) + "</h3>");
                continue;
            }

            if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
                if (!inList) {
                    htmlParts.push("<ul>");
                    inList = true;
                }
                htmlParts.push("<li>" + renderInlineFormatting(trimmed.substring(2), citationByIndex) + "</li>");
                continue;
            } else {
                if (inList) {
                    htmlParts.push("</ul>");
                    inList = false;
                }
            }

            if (trimmed === "") {
                continue;
            }

            htmlParts.push("<p>" + renderInlineFormatting(trimmed, citationByIndex) + "</p>");
        }

        if (inList) {
            htmlParts.push("</ul>");
        }
        if (inTable) {
            flushTable();
        }

        return htmlParts.join("");
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function renderPlan(plan, markCompleted = false) {
        if (!Array.isArray(plan)) {
            return;
        }
        const labels = {
            pending: "Chờ xử lý", in_progress: "Đang xử lý",
            done: "Hoàn tất", blocked: "Cần bổ sung", canceled: "Đã hủy"
        };
        const items = plan.map(step => {
            const item = document.createElement("li");
            let statusKey = "pending";
            if (step && step.status) {
                statusKey = String(step.status);
            }
            if (markCompleted && (statusKey === "pending" || statusKey === "in_progress")) {
                statusKey = "done";
            }
            let statusLabel = statusKey;
            if (Object.prototype.hasOwnProperty.call(labels, statusKey)) {
                statusLabel = labels[statusKey];
            }
            let taskText = "";
            if (step && step.task) {
                taskText = String(step.task);
            }
            item.textContent = taskText + " · " + statusLabel;
            return item;
        });
        agentPlan.replaceChildren(...items);
    }

    function resetStepper() {
        renderPlan([], false);
        agentActions.replaceChildren();
        stepperDesc.textContent = "Đang nhận yêu cầu...";
    }

    function updateStepperProgress(description, action) {
        stepperDesc.textContent = description;
        if (action) {
            const lastItem = agentActions.lastElementChild;
            if (lastItem && lastItem.textContent === description) {
                return;
            }
            const item = document.createElement("li");
            item.textContent = description;
            agentActions.appendChild(item);
        }
    }

    function finishProgress(message = "Đã hoàn tất phản hồi") {
        stepperDesc.textContent = message;
    }

    function finalizeDagForTurn(auditStatus, turnStatus, fallbackTextHint = "") {
        const isAwaiting = turnStatus === "awaiting_input" || auditStatus === "AWAITING_INPUT" || auditStatus === "NEEDS_INPUT";
        const isCanceled = auditStatus === "CANCELED" || turnStatus === "canceled";
        const isBlocked = auditStatus === "BLOCKED";
        const isError = auditStatus === "ERROR" || auditStatus === "FAIL";

        let resolvedAudit = "PASS";
        if (isAwaiting) {
            resolvedAudit = "AWAITING_INPUT";
        } else if (auditStatus) {
            resolvedAudit = String(auditStatus).toUpperCase();
        }
        dagUiState.auditStatus = resolvedAudit;

        if (dagUiState.totalLatencyMs <= 0 && dagUiState.streamStartPerf > 0) {
            dagUiState.totalLatencyMs = Math.max(1, Math.round(performance.now() - dagUiState.streamStartPerf));
        }

        const stageOrder = ["input_guardrail", "coordinator", "workers_tools", "answer_review", "output_guardrail"];
        for (const sid of stageOrder) {
            const st = dagUiState.stages[sid];
            if (!st) {
                continue;
            }
            const upperStatus = String(st.status || "IDLE").toUpperCase();
            if (upperStatus === "SUCCESS" || upperStatus === "COMPLETED") {
                st.status = "PASS";
            } else if (upperStatus === "RUNNING") {
                if (isAwaiting && sid === "workers_tools") {
                    st.status = "AWAITING_INPUT";
                } else if (isCanceled && (sid === "workers_tools" || sid === "output_guardrail")) {
                    st.status = "CANCELED";
                } else if (isError) {
                    st.status = "ERROR";
                } else {
                    st.status = "PASS";
                }
                st.executed = true;
            }
        }

        if (!isAwaiting && !isBlocked && !isError && !isCanceled) {
            const requiredIds = ["input_guardrail", "coordinator", "answer_review", "output_guardrail"];
            for (const reqId of requiredIds) {
                const st = dagUiState.stages[reqId];
                if (!st) {
                    continue;
                }
                if (String(st.status || "IDLE").toUpperCase() === "IDLE") {
                    st.status = "PASS";
                    st.executed = true;
                    if (reqId === "coordinator" || reqId === "answer_review") {
                        if (st.profile === "DETERMINISTIC") {
                            st.profile = "STANDARD";
                        }
                    }
                }
            }
        }

        const tokObj = dagUiState.tokenUsage || { input_tokens: 0, output_tokens: 0, total_tokens: 0, estimated: false };
        if (!isAwaiting && !isError && Number(tokObj.total_tokens || 0) === 0) {
            const textLen = String(fallbackTextHint || "").length;
            const estOut = Math.max(24, Math.round(textLen / 4));
            const estIn = Math.max(48, estOut);
            dagUiState.tokenUsage = {
                input_tokens: estIn,
                output_tokens: estOut,
                total_tokens: estIn + estOut,
                estimated: true
            };
        }

        renderDagTelemetryBar();
        renderDagGraph();
        if (dagUiState.selectedStageId) {
            renderDagInspector(dagUiState.selectedStageId);
        }
    }

    function finishProgressForStatus(auditStatus, turnStatus, fallbackTextHint = "") {
        stopDagLiveTimer();
        finalizeDagForTurn(auditStatus, turnStatus, fallbackTextHint);
        if (turnStatus === "awaiting_input" || auditStatus === "AWAITING_INPUT" || auditStatus === "NEEDS_INPUT") {
            finishProgress("Đang chờ lựa chọn của bạn để tiếp tục");
            return;
        }
        if (auditStatus === "CANCELED" || turnStatus === "canceled") {
            finishProgress("Đã hủy yêu cầu theo lựa chọn của bạn");
            return;
        }
        if (auditStatus === "BLOCKED") {
            finishProgress("Yêu cầu đã bị chặn bởi bộ lọc an toàn");
            return;
        }
        if (auditStatus === "PASS") {
            finishProgress("Đã hoàn tất và kiểm duyệt phản hồi");
            return;
        }
        if (auditStatus === "ERROR" || auditStatus === "FAIL") {
            finishProgress("Đã kết thúc xử lý (dữ liệu hiện có bị giới hạn)");
            return;
        }
        finishProgress("Đã hoàn tất phản hồi");
    }

    // ------------------------------------------------------------------
    // Live LangGraph StateGraph DAG Visualizer & Telemetry Controller
    // ------------------------------------------------------------------
    const DAG_STAGE_DEFS = [
        { id: "input_guardrail", label: "Input Guardrail" },
        { id: "coordinator", label: "Coordinator" },
        { id: "workers_tools", label: "Parallel Workers / Tools" },
        { id: "answer_review", label: "Answer Review" },
        { id: "output_guardrail", label: "Output Guardrail" }
    ];

    function createEmptyDagStage(stageDef) {
        return {
            stage_id: stageDef.id,
            label: stageDef.label,
            status: "IDLE",
            executed: false,
            call_count: 0,
            latency_ms: 0,
            model: "",
            profile: "DETERMINISTIC",
            tokens: { input_tokens: 0, output_tokens: 0, total_tokens: 0, estimated: false },
            nodes: [],
            input_summary: {},
            output_summary: {},
            executions: []
        };
    }

    let dagUiState = {
        stages: {},
        selectedStageId: null,
        totalLatencyMs: 0,
        activeModel: "",
        activeProfile: "STANDARD",
        tokenUsage: { input_tokens: 0, output_tokens: 0, total_tokens: 0, estimated: false },
        auditStatus: "READY",
        retryLoops: 0,
        liveTimer: null,
        streamStartPerf: 0,
        visionExecutions: [],
        latestTraceId: null
    };

    function formatDagDuration(msVal) {
        if (typeof msVal !== "number" || Number.isNaN(msVal) || msVal <= 0) {
            return "0 ms";
        }
        if (msVal >= 1000) {
            return (msVal / 1000).toFixed(2) + " s";
        }
        return Math.round(msVal) + " ms";
    }

    function stopDagLiveTimer() {
        if (dagUiState.liveTimer !== null) {
            window.clearInterval(dagUiState.liveTimer);
            dagUiState.liveTimer = null;
        }
    }

    function dagCardStatusClass(statusVal) {
        const lower = String(statusVal || "IDLE").toLowerCase();
        if (lower === "running") {
            return "is-running";
        }
        if (lower === "pass" || lower === "success" || lower === "completed" || lower === "passed") {
            return "is-pass";
        }
        if (lower === "retry") {
            return "is-retry";
        }
        if (lower === "awaiting_input" || lower === "needs_input" || lower === "suspended") {
            return "is-awaiting";
        }
        if (lower === "error" || lower === "fail" || lower === "failed") {
            return "is-error";
        }
        if (lower === "blocked" || lower === "canceled") {
            return "is-blocked";
        }
        return "is-idle";
    }

    function dagProfileShortLabel(profileVal) {
        const upper = String(profileVal || "DETERMINISTIC").toUpperCase();
        if (upper === "DEEP") {
            return "DEEP";
        }
        if (upper === "STANDARD") {
            return "STD";
        }
        return "RULE";
    }

    function renderDagTelemetryBar() {
        const dlg = document.getElementById("debugTraceDialog");
        if (dlg && dlg.open) {
            return;
        }
        if (dagLatencyBadge) {
            let latencyMs = dagUiState.totalLatencyMs;
            if (dagUiState.liveTimer !== null && dagUiState.streamStartPerf > 0) {
                latencyMs = Math.max(latencyMs, performance.now() - dagUiState.streamStartPerf);
            }
            dagLatencyBadge.textContent = formatDagDuration(latencyMs);
        }
        if (dagModelBadge) {
            const prof = String(dagUiState.activeProfile || "STANDARD").toUpperCase();
            let modelText = prof;
            if (dagUiState.activeModel && dagUiState.activeModel !== "deterministic-engine") {
                let shortModel = dagUiState.activeModel;
                if (shortModel.indexOf("/") >= 0) {
                    shortModel = shortModel.split("/").pop();
                }
                if (shortModel.length > 22) {
                    shortModel = shortModel.slice(0, 19) + "...";
                }
                modelText = prof + " · " + shortModel;
            }
            dagModelBadge.textContent = modelText;
            dagModelBadge.className = "dag-telemetry-pill dag-pill-model";
            if (prof === "DEEP") {
                dagModelBadge.classList.add("is-deep");
            } else if (prof === "STANDARD") {
                dagModelBadge.classList.add("is-standard");
            }
        }
        if (dagTokenBadge) {
            const tok = dagUiState.tokenUsage || {};
            const totalTok = Number(tok.total_tokens || 0);
            let prefix = "";
            if (tok.estimated && totalTok > 0) {
                prefix = "~";
            }
            dagTokenBadge.textContent = prefix + totalTok + " tok";
        }
        if (dagAuditBadge) {
            const auditText = String(dagUiState.auditStatus || "READY").toUpperCase();
            dagAuditBadge.textContent = auditText;
            dagAuditBadge.className = "dag-telemetry-pill dag-pill-audit";
            if (auditText === "PASS" || auditText === "COMPLETED") {
                dagAuditBadge.classList.add("is-pass");
            } else if (auditText === "RETRY" || auditText === "AWAITING_INPUT" || auditText === "NEEDS_INPUT") {
                dagAuditBadge.classList.add("is-retry");
            } else if (auditText === "RUNNING") {
                dagAuditBadge.classList.add("is-running");
            } else if (auditText === "ERROR" || auditText === "FAIL" || auditText === "BLOCKED") {
                dagAuditBadge.classList.add("is-error");
            }
        }
    }

    function renderDagInspector(stageId) {
        const dlg = document.getElementById("debugTraceDialog");
        if (dlg && dlg.open) {
            return;
        }
        if (!dagNodeInspector) {
            return;
        }
        if (!stageId || !dagUiState.stages[stageId]) {
            dagNodeInspector.hidden = true;
            dagNodeInspector.replaceChildren();
            return;
        }
        const st = dagUiState.stages[stageId];
        dagNodeInspector.hidden = false;
        dagNodeInspector.replaceChildren();

        const header = document.createElement("div");
        header.className = "dag-inspector-header";

        const titleWrap = document.createElement("div");
        titleWrap.className = "dag-inspector-title";
        const nameSpan = document.createElement("span");
        nameSpan.textContent = st.label;
        const statusChip = document.createElement("span");
        statusChip.className = "dag-node-status";
        statusChip.textContent = String(st.status || "IDLE");
        const profChip = document.createElement("span");
        profChip.className = "dag-node-profile";
        const profUpper = String(st.profile || "DETERMINISTIC").toUpperCase();
        if (profUpper === "DEEP") {
            profChip.classList.add("is-deep");
        } else if (profUpper === "DETERMINISTIC") {
            profChip.classList.add("is-rule");
        }
        profChip.textContent = profUpper;
        titleWrap.append(nameSpan, statusChip, profChip);

        const closeBtn = document.createElement("button");
        closeBtn.type = "button";
        closeBtn.className = "dag-inspector-close";
        closeBtn.textContent = "×";
        closeBtn.setAttribute("aria-label", "Đóng chi tiết nút đồ thị");
        closeBtn.addEventListener("click", () => {
            dagUiState.selectedStageId = null;
            renderDagGraph();
            renderDagInspector(null);
        });

        header.append(titleWrap, closeBtn);
        dagNodeInspector.appendChild(header);

        const statsRow = document.createElement("div");
        statsRow.className = "dag-inspector-stats";
        const tok = st.tokens || {};
        const totalTok = Number(tok.total_tokens || 0);
        const inTok = Number(tok.input_tokens || 0);
        const outTok = Number(tok.output_tokens || 0);
        let tokMode = "Thực tế";
        if (tok.estimated) {
            tokMode = "Ước tính";
        }
        let modelDisplay = "deterministic-engine";
        if (st.model) {
            modelDisplay = st.model;
        }
        let nodesListText = "—";
        if (Array.isArray(st.nodes) && st.nodes.length > 0) {
            nodesListText = st.nodes.join(", ");
        }
        const statTexts = [
            "Độ trễ: " + formatDagDuration(Number(st.latency_ms || 0)),
            "Model: " + modelDisplay,
            "Profile: " + profUpper,
            "Tokens (" + tokMode + "): " + totalTok + " (In " + inTok + " / Out " + outTok + ")",
            "Số lượt chạy: " + Number(st.call_count || 0),
            "Nodes: " + nodesListText
        ];
        for (const txt of statTexts) {
            const chip = document.createElement("span");
            chip.className = "dag-inspector-stat";
            chip.textContent = txt;
            statsRow.appendChild(chip);
        }
        dagNodeInspector.appendChild(statsRow);

        const ioGrid = document.createElement("div");
        ioGrid.className = "dag-io-grid";

        const inBox = document.createElement("div");
        inBox.className = "dag-io-box";
        const inLabel = document.createElement("div");
        inLabel.className = "dag-io-label";
        inLabel.textContent = "Input Summary";
        const inPre = document.createElement("pre");
        inPre.className = "dag-io-pre";
        inPre.textContent = JSON.stringify(st.input_summary || {}, null, 2);
        inBox.append(inLabel, inPre);

        const outBox = document.createElement("div");
        outBox.className = "dag-io-box";
        const outLabel = document.createElement("div");
        outLabel.className = "dag-io-label";
        outLabel.textContent = "Output Summary";
        const outPre = document.createElement("pre");
        outPre.className = "dag-io-pre";
        const outPayload = {
            ...(st.output_summary || {})
        };
        if (Array.isArray(st.executions) && st.executions.length > 1) {
            outPayload.executions = st.executions;
        }
        outPre.textContent = JSON.stringify(outPayload, null, 2);
        outBox.append(outLabel, outPre);

        ioGrid.append(inBox, outBox);
        dagNodeInspector.appendChild(ioGrid);
    }

    function renderDagGraph() {
        const dlg = document.getElementById("debugTraceDialog");
        if (dlg && dlg.open) {
            return;
        }
        if (!dagGraphTrack) {
            return;
        }
        dagGraphTrack.replaceChildren();
        for (let idx = 0; idx < DAG_STAGE_DEFS.length; idx++) {
            const def = DAG_STAGE_DEFS[idx];
            const st = dagUiState.stages[def.id] || createEmptyDagStage(def);

            const card = document.createElement("button");
            card.type = "button";
            card.className = "dag-node-card " + dagCardStatusClass(st.status);
            card.setAttribute("data-stage-id", def.id);
            if (dagUiState.selectedStageId === def.id) {
                card.classList.add("is-selected");
                card.setAttribute("aria-pressed", "true");
            } else {
                card.setAttribute("aria-pressed", "false");
            }

            const topRow = document.createElement("div");
            topRow.className = "dag-node-top";
            const dot = document.createElement("span");
            dot.className = "dag-node-dot";
            const titleEl = document.createElement("span");
            titleEl.className = "dag-node-title";
            titleEl.textContent = def.label;
            const statusEl = document.createElement("span");
            statusEl.className = "dag-node-status";
            statusEl.textContent = String(st.status || "IDLE");
            topRow.append(dot, titleEl, statusEl);

            const metaRow = document.createElement("div");
            metaRow.className = "dag-node-meta";

            const latEl = document.createElement("span");
            if (st.executed) {
                latEl.textContent = formatDagDuration(Number(st.latency_ms || 0));
            } else {
                latEl.textContent = "—";
            }

            const rightMeta = document.createElement("span");
            const tokCount = Number((st.tokens && st.tokens.total_tokens) || 0);
            if (tokCount > 0) {
                rightMeta.textContent = tokCount + "t";
            } else {
                rightMeta.textContent = "";
            }

            const profEl = document.createElement("span");
            const profUpper = String(st.profile || "DETERMINISTIC").toUpperCase();
            profEl.className = "dag-node-profile";
            if (profUpper === "DEEP") {
                profEl.classList.add("is-deep");
            } else if (profUpper === "DETERMINISTIC") {
                profEl.classList.add("is-rule");
            }
            profEl.textContent = dagProfileShortLabel(profUpper);

            metaRow.append(latEl, rightMeta, profEl);
            card.append(topRow, metaRow);

            card.addEventListener("click", () => {
                if (dagUiState.selectedStageId === def.id) {
                    dagUiState.selectedStageId = null;
                } else {
                    dagUiState.selectedStageId = def.id;
                }
                renderDagGraph();
                renderDagInspector(dagUiState.selectedStageId);
            });

            dagGraphTrack.appendChild(card);

            if (idx < DAG_STAGE_DEFS.length - 1) {
                const conn = document.createElement("div");
                conn.className = "dag-node-connector";
                const nextDef = DAG_STAGE_DEFS[idx + 1];
                const nextSt = dagUiState.stages[nextDef.id];
                if (def.id === "workers_tools" && (dagUiState.retryLoops > 0 || (nextSt && nextSt.status === "RETRY"))) {
                    conn.classList.add("is-retry");
                    conn.textContent = "⇄";
                    conn.title = "Audit RETRY Loop";
                } else if (st.executed || (nextSt && nextSt.executed)) {
                    conn.classList.add("is-active");
                    conn.textContent = "→";
                } else {
                    conn.textContent = "→";
                }
                dagGraphTrack.appendChild(conn);
            }
        }
    }

    function resetDagState(isRunning = false, initialRunningStage = "input_guardrail") {
        stopDagLiveTimer();
        const freshStages = {};
        for (const def of DAG_STAGE_DEFS) {
            freshStages[def.id] = createEmptyDagStage(def);
        }
        dagUiState.stages = freshStages;
        dagUiState.selectedStageId = null;
        dagUiState.totalLatencyMs = 0;
        dagUiState.activeModel = "";
        dagUiState.activeProfile = "STANDARD";
        dagUiState.tokenUsage = { input_tokens: 0, output_tokens: 0, total_tokens: 0, estimated: false };
        dagUiState.retryLoops = 0;
        dagUiState.visionExecutions = [];
        dagUiState.latestTraceId = null;
        if (isRunning) {
            dagUiState.auditStatus = "RUNNING";
            dagUiState.streamStartPerf = performance.now();
            if (dagUiState.stages[initialRunningStage]) {
                dagUiState.stages[initialRunningStage].status = "RUNNING";
            }
            dagUiState.liveTimer = window.setInterval(() => {
                renderDagTelemetryBar();
            }, 120);
        } else {
            dagUiState.auditStatus = "READY";
            dagUiState.streamStartPerf = 0;
        }
        renderDagTelemetryBar();
        renderDagGraph();
        renderDagInspector(null);
    }

    function mergeVisionIntoDagStages() {
        if (!Array.isArray(dagUiState.visionExecutions) || dagUiState.visionExecutions.length === 0) {
            return;
        }
        const st = dagUiState.stages.input_guardrail;
        if (!st) {
            return;
        }
        for (const vEvt of dagUiState.visionExecutions) {
            const alreadyHasVision = Array.isArray(st.nodes) && st.nodes.includes("vision_ocr");
            if (!alreadyHasVision) {
                st.executed = true;
                st.call_count = Number(st.call_count || 0) + 1;
                st.latency_ms = Number((Number(st.latency_ms || 0) + Number(vEvt.latency_ms || 0)).toFixed(2));
                if (!Array.isArray(st.nodes)) {
                    st.nodes = [];
                }
                st.nodes.unshift("vision_ocr");
                if (!Array.isArray(st.executions)) {
                    st.executions = [];
                }
                st.executions.unshift(vEvt);
                if (vEvt.status && st.status === "IDLE") {
                    st.status = vEvt.status;
                }
                if (vEvt.profile === "STANDARD" && st.profile === "DETERMINISTIC") {
                    st.profile = "STANDARD";
                    st.model = vEvt.model || "cloud-vision";
                }
                const vTok = vEvt.tokens || {};
                st.tokens.input_tokens = Number(st.tokens.input_tokens || 0) + Number(vTok.input_tokens || 0);
                st.tokens.output_tokens = Number(st.tokens.output_tokens || 0) + Number(vTok.output_tokens || 0);
                st.tokens.total_tokens = Number(st.tokens.total_tokens || 0) + Number(vTok.total_tokens || 0);
                if (vTok.estimated) {
                    st.tokens.estimated = true;
                }
                dagUiState.tokenUsage.input_tokens = Number(dagUiState.tokenUsage.input_tokens || 0) + Number(vTok.input_tokens || 0);
                dagUiState.tokenUsage.output_tokens = Number(dagUiState.tokenUsage.output_tokens || 0) + Number(vTok.output_tokens || 0);
                dagUiState.tokenUsage.total_tokens = Number(dagUiState.tokenUsage.total_tokens || 0) + Number(vTok.total_tokens || 0);
                if (vTok.estimated) {
                    dagUiState.tokenUsage.estimated = true;
                }
            }
        }
    }

    const NODE_NAME_TO_STAGE = {
        vision_ocr: "input_guardrail",
        input_guardrail: "input_guardrail",
        context_prepare: "input_guardrail",
        deep_decide: "coordinator",
        plan_update: "coordinator",
        memory_update: "coordinator",
        react_action: "workers_tools",
        subagent: "workers_tools",
        skill_inspect: "workers_tools",
        prepare_interaction: "workers_tools",
        await_interaction: "workers_tools",
        answer_review: "answer_review",
        output_guardrail: "output_guardrail"
    };

    function advanceDagFromNodeName(nodeName, description = "") {
        const rawNode = String(nodeName || "").trim();
        if (!rawNode) {
            return;
        }
        const stageId = NODE_NAME_TO_STAGE[rawNode] || "coordinator";
        const st = dagUiState.stages[stageId];
        if (!st) {
            return;
        }
        st.executed = true;
        st.call_count = Number(st.call_count || 0) + 1;
        if (rawNode === "prepare_interaction" || rawNode === "await_interaction") {
            st.status = "AWAITING_INPUT";
        } else {
            st.status = "PASS";
        }
        if (!Array.isArray(st.nodes)) {
            st.nodes = [];
        }
        if (!st.nodes.includes(rawNode)) {
            st.nodes.push(rawNode);
        }
        if (dagUiState.streamStartPerf > 0) {
            const elapsedNow = Math.max(1, Math.round(performance.now() - dagUiState.streamStartPerf));
            st.latency_ms = Math.max(Number(st.latency_ms || 0), elapsedNow);
            dagUiState.totalLatencyMs = elapsedNow;
        }
        if (stageId === "coordinator" || stageId === "answer_review") {
            if (st.profile === "DETERMINISTIC") {
                st.profile = "STANDARD";
            }
        }
        if (description) {
            st.output_summary = { ...(st.output_summary || {}), description: String(description) };
        }

        const nextStageMap = {
            input_guardrail: "coordinator",
            coordinator: "workers_tools",
            workers_tools: "answer_review",
            answer_review: "output_guardrail"
        };
        const nextSid = nextStageMap[stageId];
        if (nextSid && dagUiState.stages[nextSid] && dagUiState.stages[nextSid].status === "IDLE") {
            dagUiState.stages[nextSid].status = "RUNNING";
        }

        renderDagTelemetryBar();
        renderDagGraph();
        if (dagUiState.selectedStageId) {
            renderDagInspector(dagUiState.selectedStageId);
        }
    }

    function applyDagNodeEvent(dagNode) {
        if (!dagNode || typeof dagNode !== "object") {
            return;
        }
        const sid = dagNode.stage_id || "coordinator";
        if (!dagUiState.stages[sid]) {
            return;
        }
        if (dagNode.node === "vision_ocr") {
            if (dagNode.status === "RUNNING") {
                dagUiState.stages.input_guardrail.status = "RUNNING";
                dagUiState.stages.input_guardrail.input_summary = dagNode.input_summary || {};
            } else {
                dagUiState.visionExecutions.push(dagNode);
                mergeVisionIntoDagStages();
            }
            renderDagTelemetryBar();
            renderDagGraph();
            if (dagUiState.selectedStageId) {
                renderDagInspector(dagUiState.selectedStageId);
            }
            return;
        }
        advanceDagFromNodeName(dagNode.node, (dagNode.output_summary && dagNode.output_summary.description) || "");
    }

    function applyDagTelemetry(telemetry, isLiveStream = false) {
        if (!telemetry || typeof telemetry !== "object") {
            return;
        }
        if (telemetry.stages && typeof telemetry.stages === "object") {
            for (const def of DAG_STAGE_DEFS) {
                const incoming = telemetry.stages[def.id];
                if (incoming && typeof incoming === "object") {
                    let nodesCopy = [];
                    if (Array.isArray(incoming.nodes)) {
                        nodesCopy = incoming.nodes.slice();
                    }
                    let execsCopy = [];
                    if (Array.isArray(incoming.executions)) {
                        execsCopy = incoming.executions.slice();
                    }
                    dagUiState.stages[def.id] = {
                        ...createEmptyDagStage(def),
                        ...incoming,
                        tokens: {
                            input_tokens: 0,
                            output_tokens: 0,
                            total_tokens: 0,
                            estimated: false,
                            ...(incoming.tokens || {})
                        },
                        nodes: nodesCopy,
                        executions: execsCopy
                    };
                }
            }
        }
        dagUiState.totalLatencyMs = Number(telemetry.total_latency_ms || 0);
        dagUiState.activeModel = String(telemetry.active_model || "");
        dagUiState.activeProfile = String(telemetry.active_profile || "STANDARD");
        dagUiState.tokenUsage = {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            estimated: false,
            ...(telemetry.token_usage || {})
        };
        dagUiState.retryLoops = Number(telemetry.retry_loops || 0);
        if (telemetry.audit_status) {
            dagUiState.auditStatus = String(telemetry.audit_status);
        }
        mergeVisionIntoDagStages();

        if (isLiveStream) {
            if (dagUiState.stages.input_guardrail.executed && !dagUiState.stages.coordinator.executed) {
                dagUiState.stages.coordinator.status = "RUNNING";
            } else if (dagUiState.stages.coordinator.executed && !dagUiState.stages.output_guardrail.executed) {
                let lastEvents = [];
                if (Array.isArray(telemetry.node_events)) {
                    lastEvents = telemetry.node_events;
                }
                let latestEvt = null;
                if (lastEvents.length > 0) {
                    latestEvt = lastEvents[lastEvents.length - 1];
                }
                if (latestEvt && latestEvt.stage_id === "coordinator") {
                    const decKind = String((latestEvt.output_summary && latestEvt.output_summary.decision_kind) || "");
                    if (decKind === "final") {
                        dagUiState.stages.answer_review.status = "RUNNING";
                    } else if (decKind === "tool" || decKind === "skill" || decKind === "delegate" || decKind === "clarify" || decKind === "confirm") {
                        dagUiState.stages.workers_tools.status = "RUNNING";
                    }
                } else if (latestEvt && latestEvt.stage_id === "workers_tools" && latestEvt.status !== "AWAITING_INPUT") {
                    dagUiState.stages.coordinator.status = "RUNNING";
                } else if (latestEvt && latestEvt.stage_id === "answer_review" && latestEvt.status === "RETRY") {
                    dagUiState.stages.coordinator.status = "RUNNING";
                } else if (latestEvt && latestEvt.stage_id === "answer_review" && latestEvt.status === "PASS") {
                    dagUiState.stages.output_guardrail.status = "RUNNING";
                }
            }
        }

        renderDagTelemetryBar();
        renderDagGraph();
        if (dagUiState.selectedStageId) {
            renderDagInspector(dagUiState.selectedStageId);
        }
    }

    // ------------------------------------------------------------------
    // Multi-Session SQLite History Sidebar Controller
    // ------------------------------------------------------------------
    const SIDEBAR_COLLAPSED_KEY = "abank-sidebar-collapsed-v2";
    const sessionUiState = {
        sessions: [],
        total: 0,
        searchQuery: "",
        editingSessionId: null,
        confirmDeleteId: null,
        searchTimer: null
    };

    function formatSessionTimestamp(isoStr) {
        if (!isoStr) {
            return "";
        }
        const dt = new Date(isoStr);
        if (Number.isNaN(dt.getTime())) {
            return "";
        }
        const hh = String(dt.getHours()).padStart(2, "0");
        const mm = String(dt.getMinutes()).padStart(2, "0");
        const dd = String(dt.getDate()).padStart(2, "0");
        const mo = String(dt.getMonth() + 1).padStart(2, "0");
        return hh + ":" + mm + " · " + dd + "/" + mo;
    }

    function syncSidebarCollapsedState(collapsed) {
        if (!chatContainer) {
            return;
        }
        chatContainer.classList.toggle("is-sidebar-collapsed", collapsed);
        if (sidebarToggleBtn) {
            sidebarToggleBtn.setAttribute("aria-expanded", String(!collapsed));
        }
        try {
            let storedVal = "0";
            if (collapsed) {
                storedVal = "1";
            }
            localStorage.setItem(SIDEBAR_COLLAPSED_KEY, storedVal);
        } catch {
            // ignore storage errors
        }
    }

    function initSidebarToggle() {
        let initialCollapsed = false;
        try {
            if (localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1") {
                initialCollapsed = true;
            }
        } catch {
            initialCollapsed = false;
        }
        if (window.innerWidth <= 960) {
            initialCollapsed = true;
        }
        syncSidebarCollapsedState(initialCollapsed);
        if (sidebarToggleBtn) {
            sidebarToggleBtn.addEventListener("click", () => {
                const currentlyCollapsed = chatContainer && chatContainer.classList.contains("is-sidebar-collapsed");
                syncSidebarCollapsedState(!currentlyCollapsed);
            });
        }
    }

    function deriveOptimisticSessionTitle(queryText) {
        let clean = String(queryText || "").split("\n")[0].trim();
        if (!clean) {
            return "Cuộc hội thoại mới";
        }
        if (clean.startsWith("Đã gửi ") && clean.includes("hình ảnh đính kèm")) {
            return "Phân tích hình ảnh đính kèm";
        }
        const fillerPrefixes = [
            "cho tôi hỏi về ",
            "cho tôi hỏi ",
            "cho mình hỏi về ",
            "cho mình hỏi ",
            "cho em hỏi về ",
            "cho em hỏi ",
            "cho hỏi về ",
            "cho hỏi ",
            "cho tôi biết ",
            "xin hỏi về ",
            "xin hỏi ",
            "tôi muốn tìm hiểu về ",
            "tôi muốn hỏi về ",
            "tôi muốn biết về ",
            "tôi muốn hỏi ",
            "tôi cần tìm hiểu về ",
            "tôi cần hỏi về ",
            "tôi cần hỏi ",
            "mình muốn tìm hiểu về ",
            "mình muốn hỏi về ",
            "mình muốn hỏi ",
            "bạn ơi cho hỏi ",
            "bạn ơi ",
            "trợ lý ơi ",
            "ad ơi ",
            "bạn hãy giúp tôi ",
            "bạn hãy ",
            "hãy giúp tôi ",
            "giúp tôi ",
            "làm ơn ",
            "vui lòng ",
            "hãy "
        ];
        let changed = true;
        while (changed) {
            changed = false;
            const lowered = clean.toLowerCase();
            for (const prefix of fillerPrefixes) {
                if (lowered.startsWith(prefix) && clean.length > prefix.length + 3) {
                    clean = clean.slice(prefix.length).trim();
                    changed = true;
                    break;
                }
            }
        }
        while (clean.length > 0 && ".?!,;:-".includes(clean[clean.length - 1])) {
            clean = clean.slice(0, -1).trim();
        }
        if (!clean) {
            return "Cuộc hội thoại mới";
        }
        clean = clean[0].toUpperCase() + clean.slice(1);
        if (clean.length <= 58) {
            return clean;
        }
        let cutoff = clean.slice(0, 55).trimEnd();
        const lastSpace = cutoff.lastIndexOf(" ");
        if (lastSpace >= 16) {
            cutoff = cutoff.slice(0, lastSpace).trimEnd();
        }
        return cutoff + "...";
    }

    function applyResolvedSessionTitle(sessionId, titleText) {
        if (!sessionId || !titleText) {
            return;
        }
        const idx = sessionUiState.sessions.findIndex(s => s && s.session_id === sessionId);
        if (idx >= 0) {
            sessionUiState.sessions[idx] = {
                ...sessionUiState.sessions[idx],
                title: titleText
            };
            renderSessionList();
        }
    }

    function buildSessionItemElement(item) {
        const card = document.createElement("div");
        card.className = "session-item";
        card.setAttribute("role", "listitem");
        if (item.session_id === currentSessionId) {
            card.classList.add("is-active");
        }

        if (sessionUiState.editingSessionId === item.session_id) {
            const form = document.createElement("form");
            form.className = "session-rename-form";
            const input = document.createElement("input");
            input.type = "text";
            input.className = "session-rename-input";
            input.maxLength = 80;
            input.value = item.title || "";
            input.setAttribute("aria-label", "Tên mới cho cuộc hội thoại");
            const saveBtn = document.createElement("button");
            saveBtn.type = "submit";
            saveBtn.className = "session-rename-save";
            saveBtn.textContent = "Lưu";
            const cancelBtn = document.createElement("button");
            cancelBtn.type = "button";
            cancelBtn.className = "session-rename-cancel";
            cancelBtn.textContent = "Hủy";
            cancelBtn.addEventListener("click", () => {
                sessionUiState.editingSessionId = null;
                renderSessionList();
            });
            form.addEventListener("submit", async (event) => {
                event.preventDefault();
                const nextTitle = input.value.trim();
                if (!nextTitle) {
                    sessionUiState.editingSessionId = null;
                    renderSessionList();
                    return;
                }
                await renameSession(item.session_id, nextTitle);
            });
            form.append(input, saveBtn, cancelBtn);
            card.appendChild(form);
            window.setTimeout(() => {
                input.focus();
                input.select();
            }, 0);
            return card;
        }

        const topRow = document.createElement("div");
        topRow.className = "session-item-top";

        const selectBtn = document.createElement("button");
        selectBtn.type = "button";
        selectBtn.className = "session-item-select-btn";
        selectBtn.addEventListener("click", () => {
            switchToSession(item.session_id);
        });

        const titleRow = document.createElement("div");
        titleRow.className = "session-item-title-row";
        const titleEl = document.createElement("span");
        titleEl.className = "session-item-title";
        titleEl.textContent = item.title || "Cuộc hội thoại mới";
        titleRow.appendChild(titleEl);

        const previewEl = document.createElement("div");
        previewEl.className = "session-item-preview";
        previewEl.textContent = item.preview || "Chưa có tin nhắn";

        selectBtn.append(titleRow, previewEl);

        const actions = document.createElement("div");
        actions.className = "session-item-actions";

        const renameBtn = document.createElement("button");
        renameBtn.type = "button";
        renameBtn.className = "session-action-btn";
        renameBtn.textContent = "✎";
        renameBtn.title = "Đổi tên phiên";
        renameBtn.setAttribute("aria-label", "Đổi tên cuộc hội thoại");
        renameBtn.addEventListener("click", (event) => {
            event.stopPropagation();
            sessionUiState.editingSessionId = item.session_id;
            sessionUiState.confirmDeleteId = null;
            renderSessionList();
        });

        const deleteBtn = document.createElement("button");
        deleteBtn.type = "button";
        deleteBtn.className = "session-action-btn is-danger";
        if (sessionUiState.confirmDeleteId === item.session_id) {
            deleteBtn.textContent = "✓";
            deleteBtn.title = "Nhấn lần nữa để xóa vĩnh viễn";
            deleteBtn.setAttribute("aria-label", "Xác nhận xóa cuộc hội thoại");
        } else {
            deleteBtn.textContent = "×";
            deleteBtn.title = "Xóa cuộc hội thoại";
            deleteBtn.setAttribute("aria-label", "Xóa cuộc hội thoại");
        }
        deleteBtn.addEventListener("click", (event) => {
            event.stopPropagation();
            if (sessionUiState.confirmDeleteId === item.session_id) {
                sessionUiState.confirmDeleteId = null;
                deleteSessionById(item.session_id);
            } else {
                sessionUiState.confirmDeleteId = item.session_id;
                renderSessionList();
                window.setTimeout(() => {
                    if (sessionUiState.confirmDeleteId === item.session_id) {
                        sessionUiState.confirmDeleteId = null;
                        renderSessionList();
                    }
                }, 3500);
            }
        });

        actions.append(renameBtn, deleteBtn);
        topRow.append(selectBtn, actions);

        const metaRow = document.createElement("div");
        metaRow.className = "session-item-meta";
        const countEl = document.createElement("span");
        countEl.textContent = Number(item.message_count || 0) + " tin nhắn · " + formatSessionTimestamp(item.updated_at);
        metaRow.appendChild(countEl);

        if (item.status === "awaiting_input") {
            const chip = document.createElement("span");
            chip.className = "session-status-chip is-awaiting";
            chip.textContent = "Chờ chọn";
            metaRow.appendChild(chip);
        } else if (item.audit_status === "PASS") {
            const chip = document.createElement("span");
            chip.className = "session-status-chip is-pass";
            chip.textContent = "PASS";
            metaRow.appendChild(chip);
        }

        card.append(topRow, metaRow);
        return card;
    }

    function renderSessionList() {
        if (!sessionList) {
            return;
        }
        if (sessionCountBadge) {
            sessionCountBadge.textContent = String(sessionUiState.total);
        }
        sessionList.replaceChildren();

        if (!Array.isArray(sessionUiState.sessions) || sessionUiState.sessions.length === 0) {
            const emptyEl = document.createElement("div");
            emptyEl.className = "session-empty-note";
            if (sessionUiState.searchQuery.trim()) {
                emptyEl.textContent = "Không tìm thấy cuộc hội thoại nào khớp với từ khóa.";
            } else {
                emptyEl.textContent = "Chưa có lịch sử hội thoại nào trong SQLite.";
            }
            sessionList.appendChild(emptyEl);
            return;
        }

        const lbl = document.createElement("div");
        lbl.className = "session-group-label";
        lbl.textContent = "Gần đây (" + sessionUiState.sessions.length + ")";
        sessionList.appendChild(lbl);
        for (const rItem of sessionUiState.sessions) {
            sessionList.appendChild(buildSessionItemElement(rItem));
        }
    }

    function upsertOptimisticSessionInSidebar(queryText, statusVal = "running") {
        if (!currentSessionId) {
            return;
        }
        const cleanTitle = deriveOptimisticSessionTitle(queryText);
        let cleanPreview = String(queryText || "").trim();
        if (cleanPreview.length > 110) {
            cleanPreview = cleanPreview.slice(0, 107) + "...";
        }
        if (!cleanPreview) {
            cleanPreview = "Đang xử lý yêu cầu...";
        }
        const nowIso = new Date().toISOString();
        const existingIdx = sessionUiState.sessions.findIndex(s => s && s.session_id === currentSessionId);
        if (existingIdx >= 0) {
            const existing = { ...sessionUiState.sessions[existingIdx] };
            if (!existing.title || existing.title === "Cuộc hội thoại mới") {
                existing.title = cleanTitle;
            }
            existing.preview = cleanPreview;
            existing.updated_at = nowIso;
            existing.status = statusVal;
            existing.message_count = Math.max(1, Number(existing.message_count || 0));
            sessionUiState.sessions.splice(existingIdx, 1);
            sessionUiState.sessions.unshift(existing);
        } else {
            sessionUiState.sessions.unshift({
                session_id: currentSessionId,
                title: cleanTitle,
                created_at: nowIso,
                updated_at: nowIso,
                message_count: 1,
                preview: cleanPreview,
                status: statusVal,
                audit_status: ""
            });
            sessionUiState.total = Number(sessionUiState.total || 0) + 1;
        }
        renderSessionList();
    }

    async function fetchSessions() {
        try {
            const params = new URLSearchParams();
            if (sessionUiState.searchQuery.trim()) {
                params.set("q", sessionUiState.searchQuery.trim());
            }
            params.set("limit", "100");
            const resp = await fetch("/api/sessions?" + params.toString());
            if (!resp.ok) {
                return;
            }
            const data = await resp.json();
            if (Array.isArray(data.sessions)) {
                sessionUiState.sessions = data.sessions;
            } else {
                sessionUiState.sessions = [];
            }
            sessionUiState.total = Number(data.total || sessionUiState.sessions.length);
            renderSessionList();
        } catch (err) {
            console.warn("Fetch sessions failed:", err);
        }
    }

    function resetChatWorkspaceToEmpty() {
        pendingCard = null;
        pendingAttachments = [];
        renderAttachmentPreviewBar();
        stopVoiceRecording();
        clearComposerStatus();
        setBusy(false);
        chatMessages.innerHTML = "";
        enterWelcomeState();
        resetDagState(false);
        if (nodeStepper) {
            nodeStepper.style.display = "none";
        }
    }

    async function createNewSession(focusInput = true) {
        if (busy) {
            return;
        }
        const nextId = newSessionId();
        currentSessionId = nextId;
        sessionStorage.setItem(sessionKey, currentSessionId);
        resetChatWorkspaceToEmpty();
        if (window.ABankDebugInspector) {
            window.ABankDebugInspector.setSessionId(currentSessionId);
        }
        try {
            await fetch("/api/sessions", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: currentSessionId, title: "Cuộc hội thoại mới" })
            });
        } catch {
            // ignore offline fallback
        }
        await fetchSessions();
        if (focusInput && userInput) {
            userInput.value = "";
            userInput.focus();
        }
    }

    async function switchToSession(targetSessionId) {
        if (!targetSessionId) {
            return;
        }
        if (busy) {
            showComposerStatus("Vui lòng đợi phản hồi hiện tại hoàn tất trước khi chuyển phiên.", "info", 2500);
            return;
        }
        try {
            const resp = await fetch("/api/sessions/" + encodeURIComponent(targetSessionId));
            if (!resp.ok) {
                return false;
            }
            const detail = await resp.json();
            currentSessionId = targetSessionId;
            sessionStorage.setItem(sessionKey, currentSessionId);
            if (window.ABankDebugInspector) {
                window.ABankDebugInspector.setSessionId(currentSessionId);
            }

            pendingCard = null;
            pendingAttachments = [];
            renderAttachmentPreviewBar();
            stopVoiceRecording();
            clearComposerStatus();
            setBusy(false);
            chatMessages.innerHTML = "";

            let msgs = [];
            if (Array.isArray(detail.messages)) {
                msgs = detail.messages;
            }
            if (msgs.length === 0 && !detail.interaction) {
                enterWelcomeState();
                resetDagState(false);
                if (nodeStepper) {
                    nodeStepper.style.display = "none";
                }
            } else {
                exitWelcomeState();
                for (const msg of msgs) {
                    if (!msg || !msg.content) {
                        continue;
                    }
                    if (msg.role === "user") {
                        appendMessage("user", msg.content);
                    } else {
                        appendMessage("bot", msg.content, msg);
                    }
                }
                if (detail.dag_telemetry || (Array.isArray(detail.plan) && detail.plan.length > 0)) {
                    resetDagState(false);
                    let restoredTraceId = detail.latest_trace_id || "";
                    if (!restoredTraceId) {
                        for (let i = msgs.length - 1; i >= 0; i--) {
                            if (msgs[i] && msgs[i].role === "assistant" && msgs[i].trace_id) {
                                restoredTraceId = String(msgs[i].trace_id);
                                break;
                            }
                        }
                    }
                    if (restoredTraceId) {
                        dagUiState.latestTraceId = restoredTraceId;
                    }
                    if (nodeStepper) {
                        nodeStepper.style.display = "block";
                    }
                    if (detail.dag_telemetry) {
                        applyDagTelemetry(detail.dag_telemetry, false);
                    }
                    renderPlan(detail.plan || [], detail.status === "completed");
                    finishProgressForStatus(detail.audit_status, detail.status);
                } else if (nodeStepper) {
                    nodeStepper.style.display = "none";
                }
                if (detail.interaction) {
                    renderInteraction(detail);
                }
            }
            renderSessionList();
            if (window.innerWidth <= 960) {
                syncSidebarCollapsedState(true);
            }
            return true;
        } catch (err) {
            console.warn("Switch session failed:", err);
            return false;
        }
    }

    async function renameSession(sessionId, newTitle) {
        try {
            const resp = await fetch("/api/sessions/" + encodeURIComponent(sessionId), {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ title: newTitle })
            });
            sessionUiState.editingSessionId = null;
            if (resp.ok) {
                await fetchSessions();
            } else {
                renderSessionList();
            }
        } catch (err) {
            console.warn("Rename session failed:", err);
            sessionUiState.editingSessionId = null;
            renderSessionList();
        }
    }

    async function deleteSessionById(sessionId) {
        try {
            const resp = await fetch("/api/sessions/" + encodeURIComponent(sessionId), {
                method: "DELETE"
            });
            if (!resp.ok) {
                return;
            }
            await fetchSessions();
            if (sessionId === currentSessionId) {
                if (sessionUiState.sessions.length > 0) {
                    await switchToSession(sessionUiState.sessions[0].session_id);
                } else {
                    await createNewSession(true);
                }
            }
        } catch (err) {
            console.warn("Delete session failed:", err);
        }
    }

    let lightboxElements = null;
    let lightboxState = { images: [], activeIndex: 0 };

    function formatImageMetaSummary(item) {
        if (!item) {
            return "";
        }
        const subParts = [];
        if (item.document_name) {
            subParts.push(item.document_name);
        }
        if (item.page_number) {
            subParts.push("Trang " + item.page_number);
        }
        if (typeof item.region_index === "number" && item.region_index >= 0 && typeof item.region_count === "number" && item.region_count > 1) {
            subParts.push("Khung " + (item.region_index + 1) + "/" + item.region_count);
        }
        return subParts.join(" · ");
    }

    function attachDialogLightDismissFallback(dialogEl, onClose) {
        if (!dialogEl || ("closedBy" in HTMLDialogElement.prototype)) {
            return;
        }
        let pressedOutside = false;
        function isOutside(event) {
            const rect = dialogEl.getBoundingClientRect();
            return (
                event.clientX < rect.left
                || event.clientX > rect.right
                || event.clientY < rect.top
                || event.clientY > rect.bottom
            );
        }
        dialogEl.addEventListener("pointerdown", (event) => {
            pressedOutside = isOutside(event);
        });
        dialogEl.addEventListener("pointerup", (event) => {
            if (pressedOutside && isOutside(event)) {
                if (typeof onClose === "function") {
                    onClose();
                } else if (dialogEl.open) {
                    dialogEl.close();
                }
            }
        });
    }

    function ensureLightbox() {
        if (lightboxElements) {
            return lightboxElements;
        }
        const overlay = document.createElement("dialog");
        overlay.className = "image-lightbox-overlay";
        overlay.setAttribute("closedby", "any");
        overlay.setAttribute("aria-label", "Xem ảnh phóng to");

        const dialog = document.createElement("div");
        dialog.className = "image-lightbox-content";

        const topbar = document.createElement("div");
        topbar.className = "image-lightbox-topbar";

        const counterEl = document.createElement("span");
        counterEl.className = "image-lightbox-counter";

        const closeBtn = document.createElement("button");
        closeBtn.type = "button";
        closeBtn.className = "image-lightbox-close";
        closeBtn.textContent = "×";
        closeBtn.setAttribute("aria-label", "Đóng ảnh phóng to");

        topbar.append(counterEl, closeBtn);

        const stage = document.createElement("div");
        stage.className = "image-lightbox-stage";

        const prevBtn = document.createElement("button");
        prevBtn.type = "button";
        prevBtn.className = "image-lightbox-nav image-lightbox-prev";
        prevBtn.textContent = "‹";
        prevBtn.setAttribute("aria-label", "Ảnh trước");
        prevBtn.title = "Ảnh trước (←)";

        const fullImg = document.createElement("img");
        fullImg.className = "image-lightbox-img";
        fullImg.alt = "";

        const nextBtn = document.createElement("button");
        nextBtn.type = "button";
        nextBtn.className = "image-lightbox-nav image-lightbox-next";
        nextBtn.textContent = "›";
        nextBtn.setAttribute("aria-label", "Ảnh tiếp theo");
        nextBtn.title = "Ảnh tiếp theo (→)";

        stage.append(prevBtn, fullImg, nextBtn);

        const metaBox = document.createElement("div");
        metaBox.className = "image-lightbox-meta";

        dialog.append(topbar, stage, metaBox);
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);

        function renderLightboxImage() {
            const images = lightboxState.images;
            const total = images.length;
            const idx = lightboxState.activeIndex;
            const item = images[idx];
            if (!item) {
                return;
            }

            let imgAlt = "Hình ảnh minh họa";
            if (item.caption) {
                imgAlt = item.caption;
            }
            fullImg.src = item.url;
            fullImg.alt = imgAlt;

            if (total <= 1) {
                counterEl.hidden = true;
                prevBtn.hidden = true;
                nextBtn.hidden = true;
            } else {
                counterEl.hidden = false;
                counterEl.textContent = "Ảnh " + (idx + 1) + " / " + total;
                prevBtn.hidden = false;
                nextBtn.hidden = false;
                prevBtn.disabled = idx <= 0;
                nextBtn.disabled = idx >= total - 1;
            }

            metaBox.replaceChildren();

            const badgesRow = document.createElement("div");
            badgesRow.className = "image-lightbox-badges";
            if (item.citation_label) {
                const citeBadge = document.createElement("span");
                citeBadge.className = "evidence-image-badge";
                citeBadge.textContent = item.citation_label;
                badgesRow.appendChild(citeBadge);
            }
            const typeBadge = document.createElement("span");
            typeBadge.className = "evidence-image-type";
            typeBadge.textContent = regionTypeLabel(item.region_type);
            badgesRow.appendChild(typeBadge);
            metaBox.appendChild(badgesRow);

            const titleEl = document.createElement("div");
            titleEl.className = "image-lightbox-title";
            titleEl.textContent = imgAlt;
            metaBox.appendChild(titleEl);

            const metaSummary = formatImageMetaSummary(item);
            if (metaSummary) {
                const subEl = document.createElement("div");
                subEl.className = "image-lightbox-sub";
                subEl.textContent = metaSummary;
                metaBox.appendChild(subEl);
            }
        }

        function stepLightbox(delta) {
            const nextIdx = lightboxState.activeIndex + delta;
            if (nextIdx >= 0 && nextIdx < lightboxState.images.length) {
                lightboxState.activeIndex = nextIdx;
                renderLightboxImage();
            }
        }

        function closeLightbox() {
            if (overlay.open) {
                overlay.close();
            }
            fullImg.src = "";
        }

        prevBtn.addEventListener("click", () => {
            stepLightbox(-1);
        });
        nextBtn.addEventListener("click", () => {
            stepLightbox(1);
        });
        closeBtn.addEventListener("click", closeLightbox);
        overlay.addEventListener("close", () => {
            fullImg.src = "";
        });
        attachDialogLightDismissFallback(overlay, closeLightbox);
        overlay.addEventListener("click", (event) => {
            if (event.target === overlay) {
                closeLightbox();
            }
        });
        document.addEventListener("keydown", (event) => {
            if (!overlay.open) {
                return;
            }
            if (event.key === "ArrowLeft") {
                stepLightbox(-1);
            } else if (event.key === "ArrowRight") {
                stepLightbox(1);
            }
        });

        lightboxElements = { overlay, fullImg, metaBox, renderLightboxImage };
        return lightboxElements;
    }

    function openImageLightbox(imagesList, startIndex = 0) {
        if (!Array.isArray(imagesList) || imagesList.length === 0) {
            return;
        }
        const box = ensureLightbox();
        let idx = startIndex;
        if (typeof idx !== "number" || idx < 0) {
            idx = 0;
        } else if (idx >= imagesList.length) {
            idx = imagesList.length - 1;
        }
        lightboxState = { images: imagesList, activeIndex: idx };
        box.renderLightboxImage();
        if (!box.overlay.open) {
            box.overlay.showModal();
        }
    }

    function regionTypeLabel(regionType) {
        if (regionType === "app_step") {
            return "Thao tác App";
        }
        if (regionType === "slide_visual") {
            return "Hướng dẫn trực quan";
        }
        if (regionType === "table") {
            return "Bảng biểu";
        }
        if (regionType === "visual_card") {
            return "Thẻ chỉ số";
        }
        return "Minh họa";
    }

    function buildEvidenceImagesElement(images) {
        if (!Array.isArray(images) || images.length === 0) {
            return null;
        }
        const validImages = [];
        for (const item of images) {
            if (item && item.url) {
                validImages.push(item);
            }
        }
        if (validImages.length === 0) {
            return null;
        }

        const firstItem = validImages[0];
        const total = validImages.length;

        const card = document.createElement("div");
        card.className = "evidence-images-card";

        const header = document.createElement("div");
        header.className = "evidence-images-header";

        const headerTitle = document.createElement("span");
        headerTitle.textContent = "Hình ảnh minh họa liên quan (" + total + ")";
        header.appendChild(headerTitle);
        card.appendChild(header);

        const singleWrap = document.createElement("div");
        singleWrap.className = "evidence-images-single";

        const tile = document.createElement("button");
        tile.type = "button";
        if (total > 1) {
            tile.className = "evidence-image-item has-more";
        } else {
            tile.className = "evidence-image-item";
        }

        const thumbWrap = document.createElement("div");
        thumbWrap.className = "evidence-image-thumb-wrap";

        const imgEl = document.createElement("img");
        imgEl.src = firstItem.url;
        imgEl.loading = "lazy";
        let imgAlt = "Hình ảnh minh họa";
        if (firstItem.caption) {
            imgAlt = firstItem.caption;
        }
        imgEl.alt = imgAlt;
        thumbWrap.appendChild(imgEl);

        if (total > 1) {
            const moreBadge = document.createElement("span");
            moreBadge.className = "evidence-image-more-badge";
            moreBadge.textContent = "+" + (total - 1) + " ảnh";
            thumbWrap.appendChild(moreBadge);
        }

        const metaDiv = document.createElement("div");
        metaDiv.className = "evidence-image-meta";

        const topRow = document.createElement("div");
        topRow.className = "evidence-image-badges";

        if (firstItem.citation_label) {
            const citeBadge = document.createElement("span");
            citeBadge.className = "evidence-image-badge";
            citeBadge.textContent = firstItem.citation_label;
            topRow.appendChild(citeBadge);
        }

        const typeBadge = document.createElement("span");
        typeBadge.className = "evidence-image-type";
        typeBadge.textContent = regionTypeLabel(firstItem.region_type);
        topRow.appendChild(typeBadge);

        if (total > 1) {
            const countPill = document.createElement("span");
            countPill.className = "evidence-image-count-pill";
            countPill.textContent = "1/" + total;
            topRow.appendChild(countPill);
        }

        const capEl = document.createElement("div");
        capEl.className = "evidence-image-caption";
        const metaSummary = formatImageMetaSummary(firstItem);
        if (metaSummary) {
            capEl.textContent = metaSummary;
        } else {
            capEl.textContent = imgAlt;
        }

        metaDiv.append(topRow, capEl);
        tile.append(thumbWrap, metaDiv);

        tile.addEventListener("click", () => {
            openImageLightbox(validImages, 0);
        });
        singleWrap.appendChild(tile);

        card.appendChild(singleWrap);
        return card;
    }

    let citationViewerElements = null;
    let citationDrawerState = { citations: [], activePos: 0, renderToken: 0, evidenceCollapsed: false };
    const docPreviewCache = new Map();
    const pdfLocateCache = new Map();

    function chunkTypeLabel(chunkType) {
        if (chunkType === "table") {
            return "Bảng biểu";
        }
        if (chunkType === "multimodal") {
            return "Đa phương tiện";
        }
        return "Văn bản";
    }

    function resolveCitationFileType(citation) {
        if (!citation) {
            return "";
        }
        if (citation.file_type) {
            const ft = String(citation.file_type).trim().toLowerCase();
            if (ft === "pdf" || ft === "docx" || ft === "xlsx") {
                return ft;
            }
        }
        const rawSource = String(citation.source || "").split("\\").join("/").trim().toLowerCase();
        if (rawSource.endsWith(".pdf")) {
            return "pdf";
        }
        if (rawSource.endsWith(".docx")) {
            return "docx";
        }
        if (rawSource.endsWith(".xlsx")) {
            return "xlsx";
        }
        return "";
    }

    function foldClientText(value) {
        const decomposed = String(value || "").toLowerCase().normalize("NFKD");
        let out = "";
        let prevSpace = true;
        for (let i = 0; i < decomposed.length; i++) {
            const code = decomposed.charCodeAt(i);
            if (code >= 0x0300 && code <= 0x036f) {
                continue;
            }
            const ch = decomposed[i];
            if (ch === "đ") {
                out += "d";
                prevSpace = false;
            } else if ((ch >= "a" && ch <= "z") || (ch >= "0" && ch <= "9")) {
                out += ch;
                prevSpace = false;
            } else if (!prevSpace) {
                out += " ";
                prevSpace = true;
            }
        }
        return out.trim();
    }

    function buildPdfPageUrl(citation, pageNumber) {
        if (!citation) {
            return "";
        }
        const rawSource = String(citation.source || "").split("\\").join("/").trim();
        let baseUrl = "";
        if (rawSource.toLowerCase().endsWith(".pdf")) {
            baseUrl = "/api/documents/pdf?source=" + encodeURIComponent(rawSource);
        } else if (citation.pdf_url) {
            const rawUrl = String(citation.pdf_url).trim();
            const hashIdx = rawUrl.indexOf("#");
            if (hashIdx >= 0) {
                baseUrl = rawUrl.slice(0, hashIdx);
            } else {
                baseUrl = rawUrl;
            }
        }
        if (!baseUrl) {
            return "";
        }
        let targetPage = 0;
        if (typeof pageNumber === "number" && pageNumber > 0) {
            targetPage = pageNumber;
        } else if (Array.isArray(citation.page_numbers) && citation.page_numbers.length > 0) {
            const first = Number(citation.page_numbers[0]);
            if (Number.isInteger(first) && first > 0) {
                targetPage = first;
            }
        }
        if (targetPage > 0) {
            return baseUrl + "#page=" + targetPage;
        }
        return baseUrl;
    }

    function buildRawFileUrl(citation) {
        if (!citation || !citation.source) {
            return "";
        }
        const rawSource = String(citation.source || "").split("\\").join("/").trim();
        if (!rawSource) {
            return "";
        }
        return "/api/documents/file?source=" + encodeURIComponent(rawSource);
    }

    function extractCitationXPaths(citation, fallbackXPaths) {
        const out = [];
        const seen = new Set();
        let prov = null;
        if (citation && citation.provenance) {
            prov = citation.provenance;
        }
        let locs = [];
        if (prov && Array.isArray(prov.locations)) {
            locs = prov.locations;
        }
        for (const loc of locs) {
            if (loc && loc.xpath) {
                const xp = String(loc.xpath);
                if (!seen.has(xp)) {
                    seen.add(xp);
                    out.push(xp);
                }
            }
        }
        if (Array.isArray(fallbackXPaths)) {
            for (const xp of fallbackXPaths) {
                if (xp && !seen.has(String(xp))) {
                    seen.add(String(xp));
                    out.push(String(xp));
                }
            }
        }
        return out;
    }

    function extractCitationExcelTarget(citation) {
        let sheetName = "";
        let rangeText = "";
        if (citation && citation.sheet_name) {
            sheetName = String(citation.sheet_name);
        }
        if (citation && Array.isArray(citation.cell_ranges) && citation.cell_ranges.length > 0) {
            rangeText = String(citation.cell_ranges[0]);
        }
        let prov = null;
        if (citation && citation.provenance) {
            prov = citation.provenance;
        }
        let locs = [];
        if (prov && Array.isArray(prov.locations)) {
            locs = prov.locations;
        }
        for (const loc of locs) {
            if (!loc) {
                continue;
            }
            if (!sheetName && loc.sheet_name) {
                sheetName = String(loc.sheet_name);
            }
            if (!rangeText && loc.cell_range) {
                rangeText = String(loc.cell_range);
            }
            if (sheetName && rangeText) {
                break;
            }
        }
        return { sheetName, rangeText };
    }

    function ensureCitationViewer() {
        if (citationViewerElements) {
            return citationViewerElements;
        }
        const dialog = document.getElementById("citationViewerDialog");
        const badgeEl = document.getElementById("citationViewerBadge");
        const typeEl = document.getElementById("citationViewerType");
        const fileKindEl = document.getElementById("citationViewerFileKind");
        const titleEl = document.getElementById("citationViewerTitle");
        const toggleEvidenceBtn = document.getElementById("citationToggleEvidenceBtn");
        const prevBtn = document.getElementById("citationPrevBtn");
        const nextBtn = document.getElementById("citationNextBtn");
        const tabsRow = document.getElementById("citationViewerTabs");
        const workspace = document.getElementById("citationViewerWorkspace");
        const evidencePane = document.getElementById("citationEvidencePane");
        const metaBox = document.getElementById("citationMetaBox");
        const copyBtn = document.getElementById("citationCopyBtn");
        const excerptEl = document.getElementById("citationExcerptEl");
        const imagesBox = document.getElementById("citationImagesBox");
        const docToolbar = document.getElementById("citationDocToolbar");
        const docStatus = document.getElementById("citationDocStatus");
        const docOutline = document.getElementById("citationDocOutline");
        const docViewport = document.getElementById("citationDocViewport");

        let activePdfObserver = null;
        let activeScrollCleanup = null;

        function clearActiveObserver() {
            if (activePdfObserver) {
                activePdfObserver.disconnect();
                activePdfObserver = null;
            }
            if (activeScrollCleanup) {
                activeScrollCleanup();
                activeScrollCleanup = null;
            }
        }

        function setDocStatus(message, isError = false) {
            if (!message) {
                docStatus.hidden = true;
                docStatus.textContent = "";
                docStatus.className = "citation-doc-status";
                return;
            }
            docStatus.hidden = false;
            docStatus.textContent = message;
            if (isError) {
                docStatus.className = "citation-doc-status is-error";
            } else {
                docStatus.className = "citation-doc-status";
            }
        }

        function renderEvidencePane(item) {
            metaBox.replaceChildren();
            const metaRows = [];
            let docTitle = item.document_name;
            if (!docTitle && item.source) {
                docTitle = sourceFileName(item.source);
            }
            if (docTitle) {
                metaRows.push(["Tài liệu", docTitle]);
            }
            if (item.source) {
                metaRows.push(["Tệp nguồn", item.source]);
            }
            let sectionPath = item.breadcrumb;
            if (!sectionPath) {
                sectionPath = item.section_title;
            } else if (item.section_title && item.breadcrumb.indexOf(item.section_title) < 0) {
                sectionPath = item.breadcrumb + " > " + item.section_title;
            }
            if (sectionPath) {
                metaRows.push(["Mục / Phân cấp", sectionPath]);
            }
            if (Array.isArray(item.page_numbers) && item.page_numbers.length > 0) {
                metaRows.push(["Trang", "Trang " + item.page_numbers.join(", ")]);
            }
            const xlTarget = extractCitationExcelTarget(item);
            if (xlTarget.sheetName) {
                let xlDisplay = xlTarget.sheetName;
                if (xlTarget.rangeText) {
                    xlDisplay += " (" + xlTarget.rangeText + ")";
                }
                metaRows.push(["Bảng tính", xlDisplay]);
            }
            for (const pair of metaRows) {
                const row = document.createElement("div");
                row.className = "citation-meta-row";
                const lbl = document.createElement("span");
                lbl.className = "citation-meta-label";
                lbl.textContent = pair[0];
                const val = document.createElement("span");
                val.className = "citation-meta-value";
                val.textContent = pair[1];
                row.append(lbl, val);
                metaBox.appendChild(row);
            }

            excerptEl.replaceChildren();
            const excerptText = String(item.excerpt || "").trim();
            if (excerptText) {
                excerptEl.innerHTML = parseMarkdown(excerptText);
                copyBtn.hidden = false;
                copyBtn.textContent = "Sao chép";
            } else {
                const emptyNote = document.createElement("p");
                emptyNote.className = "citation-empty-note";
                emptyNote.textContent = "Không có đoạn trích văn bản lưu kèm cho nguồn này.";
                excerptEl.appendChild(emptyNote);
                copyBtn.hidden = true;
            }

            imagesBox.replaceChildren();
            if (Array.isArray(item.images) && item.images.length > 0) {
                const imgGallery = buildEvidenceImagesElement(item.images);
                if (imgGallery) {
                    imagesBox.appendChild(imgGallery);
                    imagesBox.hidden = false;
                } else {
                    imagesBox.hidden = true;
                }
            } else {
                imagesBox.hidden = true;
            }
            evidencePane.scrollTop = 0;
        }

        async function renderPdfReader(item, rawSource, renderToken) {
            docOutline.hidden = true;
            docOutline.replaceChildren();
            setDocStatus("Đang tải cấu trúc trang PDF và định vị chính xác đoạn trích dẫn...");

            let manifest = docPreviewCache.get("pdf:" + rawSource);
            let manifestPromise;
            if (manifest) {
                manifestPromise = Promise.resolve(manifest);
            } else {
                manifestPromise = fetch("/api/documents/pdf/manifest?source=" + encodeURIComponent(rawSource)).then((res) => {
                    if (!res.ok) {
                        throw new Error("Không tải được manifest PDF (" + res.status + ")");
                    }
                    return res.json();
                });
            }

            const locateKey = rawSource + "|" + String(item.chunk_id || "") + "|" + String((item.page_numbers || []).join(","));
            let locateData = pdfLocateCache.get(locateKey);
            let locatePromise;
            if (locateData) {
                locatePromise = Promise.resolve(locateData);
            } else {
                let reqPages = [];
                if (Array.isArray(item.page_numbers)) {
                    reqPages = item.page_numbers;
                }
                let reqImages = [];
                if (Array.isArray(item.images)) {
                    reqImages = item.images;
                }
                locatePromise = fetch("/api/documents/pdf/locate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        source: rawSource,
                        page_numbers: reqPages,
                        chunk_id: String(item.chunk_id || ""),
                        excerpt: String(item.excerpt || ""),
                        section_title: String(item.section_title || ""),
                        images: reqImages
                    })
                }).then((res) => {
                    if (!res.ok) {
                        return { pages: {} };
                    }
                    return res.json();
                }).catch(() => {
                    return { pages: {} };
                });
            }

            let targetPages = [];
            if (Array.isArray(item.page_numbers) && item.page_numbers.length > 0) {
                for (const p of item.page_numbers) {
                    const num = Number(p);
                    if (Number.isInteger(num) && num > 0) {
                        targetPages.push(num);
                    }
                }
            }
            if (targetPages.length === 0) {
                targetPages = [1];
            }

            try {
                manifest = await manifestPromise;
            } catch (err) {
                if (renderToken !== citationDrawerState.renderToken) {
                    return;
                }
                setDocStatus(err.message || "Không thể mở tài liệu PDF.", true);
                return;
            }
            if (renderToken !== citationDrawerState.renderToken) {
                return;
            }
            docPreviewCache.set("pdf:" + rawSource, manifest);

            let fallbackPageCount = 0;
            if (Array.isArray(manifest.pages)) {
                fallbackPageCount = manifest.pages.length;
            }
            const totalPages = Number(manifest.total_pages || fallbackPageCount);
            const pageCards = new Map();
            let currentZoom = 1.0;
            let basePageWidth = 760;
            if (docViewport.clientWidth > 160) {
                basePageWidth = Math.min(880, Math.max(460, docViewport.clientWidth - 52));
            }

            const stack = document.createElement("div");
            stack.className = "pdf-pages-stack";

            function cardSizeForPage(pageMeta) {
                const pw = Math.max(1, Number(pageMeta.width || 595));
                const ph = Math.max(1, Number(pageMeta.height || 842));
                const wPx = Math.round(basePageWidth * currentZoom);
                const hPx = Math.round((wPx * ph) / pw);
                return { wPx, hPx };
            }

            function loadPageImage(pageNumber) {
                const rec = pageCards.get(pageNumber);
                if (!rec || rec.loaded) {
                    return;
                }
                rec.loaded = true;
                rec.img.src = "/api/documents/pdf/page?source=" + encodeURIComponent(rawSource) + "&page=" + pageNumber + "&dpi=144";
            }

            for (const pMeta of manifest.pages || []) {
                const pNum = Number(pMeta.page_number);
                const size = cardSizeForPage(pMeta);
                const card = document.createElement("div");
                card.className = "pdf-page-card";
                card.setAttribute("data-pdf-page", String(pNum));
                card.style.width = size.wPx + "px";
                card.style.height = size.hPx + "px";

                const badge = document.createElement("span");
                badge.className = "pdf-page-badge";
                badge.textContent = "Trang " + pNum + " / " + totalPages;

                const placeholder = document.createElement("div");
                placeholder.className = "pdf-page-placeholder";
                placeholder.textContent = "Đang tải trang " + pNum + "...";

                const img = document.createElement("img");
                img.className = "pdf-page-img";
                img.alt = "Trang " + pNum + " của " + (manifest.file_name || "PDF");
                img.addEventListener("load", () => {
                    placeholder.hidden = true;
                });

                const overlayLayer = document.createElement("div");
                overlayLayer.className = "pdf-overlay-layer";

                card.append(placeholder, img, overlayLayer, badge);
                stack.appendChild(card);
                pageCards.set(pNum, { card, img, overlayLayer, meta: pMeta, loaded: false });
            }

            docViewport.replaceChildren(stack);

            // Build PDF Reader Toolbar
            docToolbar.replaceChildren();
            const leftGroup = document.createElement("div");
            leftGroup.className = "doc-toolbar-group";

            const focusBtn = document.createElement("button");
            focusBtn.type = "button";
            focusBtn.className = "doc-tool-btn doc-tool-primary";
            focusBtn.textContent = "Tới vị trí trích dẫn";
            leftGroup.appendChild(focusBtn);

            if (targetPages.length > 1) {
                for (const tp of targetPages) {
                    const jumpPill = document.createElement("button");
                    jumpPill.type = "button";
                    jumpPill.className = "doc-tool-btn";
                    jumpPill.textContent = "Tr. " + tp;
                    jumpPill.addEventListener("click", () => {
                        scrollToPageHighlight(tp, null);
                    });
                    leftGroup.appendChild(jumpPill);
                }
            }

            const prevPageBtn = document.createElement("button");
            prevPageBtn.type = "button";
            prevPageBtn.className = "doc-tool-btn";
            prevPageBtn.textContent = "‹";
            prevPageBtn.title = "Trang trước";

            const pageInput = document.createElement("input");
            pageInput.type = "number";
            pageInput.min = "1";
            pageInput.max = String(totalPages);
            pageInput.value = String(targetPages[0]);
            pageInput.className = "doc-page-input";
            pageInput.setAttribute("aria-label", "Số trang PDF");

            const pageTotalLbl = document.createElement("span");
            pageTotalLbl.className = "doc-toolbar-label";
            pageTotalLbl.textContent = "/ " + totalPages + " trang";

            const nextPageBtn = document.createElement("button");
            nextPageBtn.type = "button";
            nextPageBtn.className = "doc-tool-btn";
            nextPageBtn.textContent = "›";
            nextPageBtn.title = "Trang tiếp theo";

            leftGroup.append(prevPageBtn, pageInput, pageTotalLbl, nextPageBtn);

            const rightGroup = document.createElement("div");
            rightGroup.className = "doc-toolbar-group";

            const zoomOutBtn = document.createElement("button");
            zoomOutBtn.type = "button";
            zoomOutBtn.className = "doc-tool-btn";
            zoomOutBtn.textContent = "−";
            zoomOutBtn.title = "Thu nhỏ";

            const zoomLabelBtn = document.createElement("button");
            zoomLabelBtn.type = "button";
            zoomLabelBtn.className = "doc-tool-btn";
            zoomLabelBtn.textContent = "100%";
            zoomLabelBtn.title = "Đặt lại thu phóng 100%";

            const zoomInBtn = document.createElement("button");
            zoomInBtn.type = "button";
            zoomInBtn.className = "doc-tool-btn";
            zoomInBtn.textContent = "+";
            zoomInBtn.title = "Phóng to";

            const searchWrap = document.createElement("div");
            searchWrap.className = "doc-search-wrap";
            const searchInput = document.createElement("input");
            searchInput.type = "search";
            searchInput.className = "doc-search-input";
            searchInput.placeholder = "Tìm trong PDF...";
            searchInput.setAttribute("aria-label", "Tìm kiếm nội dung trong PDF");
            const searchGoBtn = document.createElement("button");
            searchGoBtn.type = "button";
            searchGoBtn.className = "doc-tool-btn";
            searchGoBtn.textContent = "Tìm";
            const searchCountEl = document.createElement("span");
            searchCountEl.className = "doc-search-count";
            const searchNextBtn = document.createElement("button");
            searchNextBtn.type = "button";
            searchNextBtn.className = "doc-tool-btn";
            searchNextBtn.textContent = "↓";
            searchNextBtn.title = "Kết quả tiếp theo";
            searchNextBtn.hidden = true;
            searchWrap.append(searchInput, searchGoBtn, searchCountEl, searchNextBtn);

            const openRawLink = document.createElement("a");
            openRawLink.className = "doc-tool-btn";
            openRawLink.href = buildPdfPageUrl(item, targetPages[0]);
            openRawLink.target = "_blank";
            openRawLink.rel = "noopener noreferrer";
            openRawLink.textContent = "Mở PDF gốc ↗";

            rightGroup.append(zoomOutBtn, zoomLabelBtn, zoomInBtn, searchWrap, openRawLink);
            docToolbar.append(leftGroup, rightGroup);

            // Lazy-load visible pages + sync pageInput cleanly via scroll center
            clearActiveObserver();
            activePdfObserver = new IntersectionObserver((entries) => {
                for (const entry of entries) {
                    if (entry.isIntersecting) {
                        const pNum = Number(entry.target.getAttribute("data-pdf-page"));
                        loadPageImage(pNum);
                    }
                }
            }, { root: docViewport, rootMargin: "550px 0px", threshold: 0.01 });

            for (const rec of pageCards.values()) {
                activePdfObserver.observe(rec.card);
            }

            let ignoreScrollSyncUntil = 0;
            function syncPageInputFromScroll() {
                if (Date.now() < ignoreScrollSyncUntil) {
                    return;
                }
                const vpMid = docViewport.scrollTop + (docViewport.clientHeight * 0.42);
                let bestPage = 1;
                let bestDist = 1e12;
                for (const [pNum, rec] of pageCards.entries()) {
                    const top = rec.card.offsetTop;
                    const bottom = top + rec.card.offsetHeight;
                    if (vpMid >= top && vpMid <= bottom) {
                        bestPage = pNum;
                        break;
                    }
                    const dist = Math.abs(((top + bottom) * 0.5) - vpMid);
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestPage = pNum;
                    }
                }
                pageInput.value = String(bestPage);
            }
            docViewport.addEventListener("scroll", syncPageInputFromScroll, { passive: true });
            activeScrollCleanup = () => {
                docViewport.removeEventListener("scroll", syncPageInputFromScroll);
            };

            let activeLocatePages = {};
            let activeSearchMatches = [];
            let activeSearchPos = 0;

            function paintOverlaysOnPages(locatePagesMap, searchMatchesList) {
                for (const [pNum, rec] of pageCards.entries()) {
                    rec.overlayLayer.replaceChildren();
                    const pKey = String(pNum);
                    let locPage = null;
                    if (locatePagesMap && locatePagesMap[pKey]) {
                        locPage = locatePagesMap[pKey];
                    }
                    if (locPage && Array.isArray(locPage.boxes)) {
                        for (const bx of locPage.boxes) {
                            if (!bx || !Array.isArray(bx.bbox) || bx.bbox.length !== 4) {
                                continue;
                            }
                            const [x0, y0, x1, y1] = bx.bbox;
                            const el = document.createElement("div");
                            if (bx.kind === "region") {
                                el.className = "pdf-region-box";
                            } else {
                                el.className = "pdf-highlight-box";
                            }
                            el.style.left = (x0 * 100).toFixed(2) + "%";
                            el.style.top = (y0 * 100).toFixed(2) + "%";
                            el.style.width = (Math.max(0.02, x1 - x0) * 100).toFixed(2) + "%";
                            el.style.height = (Math.max(0.012, y1 - y0) * 100).toFixed(2) + "%";
                            rec.overlayLayer.appendChild(el);
                        }
                    }
                    if (locPage && Array.isArray(locPage.focus_box) && locPage.focus_box.length === 4) {
                        const [fx0, fy0, fx1, fy1] = locPage.focus_box;
                        const anchor = document.createElement("div");
                        anchor.className = "pdf-focus-anchor";
                        anchor.style.left = (Math.max(0, fx0 - 0.008) * 100).toFixed(2) + "%";
                        anchor.style.top = (Math.max(0, fy0 - 0.008) * 100).toFixed(2) + "%";
                        anchor.style.width = (Math.min(1, fx1 - fx0 + 0.016) * 100).toFixed(2) + "%";
                        anchor.style.height = (Math.min(1, fy1 - fy0 + 0.016) * 100).toFixed(2) + "%";
                        const pill = document.createElement("span");
                        pill.className = "pdf-anchor-pill";
                        if (item.label) {
                            pill.textContent = item.label;
                        } else if (item.index) {
                            pill.textContent = "Nguồn " + item.index;
                        } else {
                            pill.textContent = "Vị trí trích dẫn";
                        }
                        anchor.appendChild(pill);
                        rec.overlayLayer.appendChild(anchor);
                    }
                    if (Array.isArray(searchMatchesList)) {
                        for (const sm of searchMatchesList) {
                            if (Number(sm.page_number) !== pNum || !Array.isArray(sm.boxes)) {
                                continue;
                            }
                            for (const bx of sm.boxes) {
                                if (!bx || !Array.isArray(bx.bbox) || bx.bbox.length !== 4) {
                                    continue;
                                }
                                const [x0, y0, x1, y1] = bx.bbox;
                                const el = document.createElement("div");
                                el.className = "pdf-highlight-box is-search";
                                el.style.left = (x0 * 100).toFixed(2) + "%";
                                el.style.top = (y0 * 100).toFixed(2) + "%";
                                el.style.width = (Math.max(0.02, x1 - x0) * 100).toFixed(2) + "%";
                                el.style.height = (Math.max(0.012, y1 - y0) * 100).toFixed(2) + "%";
                                rec.overlayLayer.appendChild(el);
                            }
                        }
                    }
                }
            }

            function scrollToPageHighlight(pageNumber, overrideFocusBox, smoothScroll = false) {
                const clamped = Math.max(1, Math.min(totalPages, Number(pageNumber) || 1));
                const rec = pageCards.get(clamped);
                if (!rec) {
                    return;
                }
                loadPageImage(clamped);
                if (clamped > 1) {
                    loadPageImage(clamped - 1);
                }
                if (clamped < totalPages) {
                    loadPageImage(clamped + 1);
                }
                ignoreScrollSyncUntil = Date.now() + 500;
                pageInput.value = String(clamped);

                const pInfo = activeLocatePages[String(clamped)];
                let pFocusBox = null;
                if (pInfo && pInfo.focus_box) {
                    pFocusBox = pInfo.focus_box;
                }
                const fBox = overrideFocusBox || pFocusBox;
                const cardTop = rec.card.offsetTop;
                const cardHeight = rec.card.offsetHeight;
                let targetTop = Math.max(0, cardTop - 16);
                if (Array.isArray(fBox) && fBox.length === 4) {
                    const boxTop = fBox[1] * cardHeight;
                    const boxHeight = (fBox[3] - fBox[1]) * cardHeight;
                    if (boxHeight > docViewport.clientHeight * 0.62) {
                        targetTop = Math.max(0, cardTop + boxTop - 38);
                    } else {
                        const centerY = ((fBox[1] + fBox[3]) * 0.5) * cardHeight;
                        targetTop = Math.max(0, cardTop + centerY - Math.round(docViewport.clientHeight * 0.38));
                        if (cardTop + boxTop - targetTop < 32) {
                            targetTop = Math.max(0, cardTop + boxTop - 36);
                        }
                    }
                }
                let scrollBehavior = "auto";
                if (smoothScroll) {
                    scrollBehavior = "smooth";
                }
                docViewport.scrollTo({ top: targetTop, behavior: scrollBehavior });
            }

            function applyZoom(newZoom) {
                currentZoom = Math.max(0.55, Math.min(1.85, newZoom));
                zoomLabelBtn.textContent = Math.round(currentZoom * 100) + "%";
                const prevPage = Number(pageInput.value) || targetPages[0];
                for (const rec of pageCards.values()) {
                    const sz = cardSizeForPage(rec.meta);
                    rec.card.style.width = sz.wPx + "px";
                    rec.card.style.height = sz.hPx + "px";
                }
                scrollToPageHighlight(prevPage, null);
            }

            // Immediately jump to target page right away before OCR completes
            scrollToPageHighlight(targetPages[0], null);

            focusBtn.addEventListener("click", () => {
                scrollToPageHighlight(targetPages[0], null, true);
                paintOverlaysOnPages(activeLocatePages, activeSearchMatches);
            });
            prevPageBtn.addEventListener("click", () => {
                const curr = Number(pageInput.value) || 1;
                if (curr > 1) {
                    scrollToPageHighlight(curr - 1, null);
                }
            });
            nextPageBtn.addEventListener("click", () => {
                const curr = Number(pageInput.value) || 1;
                if (curr < totalPages) {
                    scrollToPageHighlight(curr + 1, null);
                }
            });
            pageInput.addEventListener("change", () => {
                scrollToPageHighlight(Number(pageInput.value) || 1, null);
            });
            zoomOutBtn.addEventListener("click", () => {
                applyZoom(currentZoom - 0.15);
            });
            zoomInBtn.addEventListener("click", () => {
                applyZoom(currentZoom + 0.15);
            });
            zoomLabelBtn.addEventListener("click", () => {
                applyZoom(1.0);
            });

            async function runPdfSearch() {
                const q = searchInput.value.trim();
                if (q.length < 2) {
                    activeSearchMatches = [];
                    searchCountEl.textContent = "";
                    searchNextBtn.hidden = true;
                    paintOverlaysOnPages(activeLocatePages, []);
                    return;
                }
                setDocStatus("Đang tìm \"" + q + "\" trong toàn bộ PDF...");
                try {
                    const resp = await fetch(
                        "/api/documents/pdf/search?source="
                        + encodeURIComponent(rawSource)
                        + "&q="
                        + encodeURIComponent(q)
                    );
                    if (!resp.ok) {
                        setDocStatus("Không thể tìm kiếm trong PDF.", true);
                        return;
                    }
                    const body = await resp.json();
                    if (renderToken !== citationDrawerState.renderToken) {
                        return;
                    }
                    if (Array.isArray(body.matches)) {
                        activeSearchMatches = body.matches;
                    } else {
                        activeSearchMatches = [];
                    }
                    activeSearchPos = 0;
                    paintOverlaysOnPages(activeLocatePages, activeSearchMatches);
                    if (activeSearchMatches.length === 0) {
                        searchCountEl.textContent = "0 kq";
                        searchNextBtn.hidden = true;
                        setDocStatus("Không tìm thấy cụm từ \"" + q + "\" trong tài liệu.");
                    } else {
                        searchCountEl.textContent = "1/" + activeSearchMatches.length;
                        searchNextBtn.hidden = activeSearchMatches.length <= 1;
                        const firstMatch = activeSearchMatches[0];
                        scrollToPageHighlight(firstMatch.page_number, firstMatch.focus_box);
                        setDocStatus("Tìm thấy trên " + activeSearchMatches.length + " trang.");
                    }
                } catch (_err) {
                    setDocStatus("Lỗi kết nối khi tìm kiếm PDF.", true);
                }
            }

            searchGoBtn.addEventListener("click", runPdfSearch);
            searchInput.addEventListener("keydown", (event) => {
                if (event.key === "Enter") {
                    event.preventDefault();
                    runPdfSearch();
                }
            });
            searchNextBtn.addEventListener("click", () => {
                if (activeSearchMatches.length === 0) {
                    return;
                }
                activeSearchPos = (activeSearchPos + 1) % activeSearchMatches.length;
                searchCountEl.textContent = (activeSearchPos + 1) + "/" + activeSearchMatches.length;
                const match = activeSearchMatches[activeSearchPos];
                scrollToPageHighlight(match.page_number, match.focus_box);
            });

            // Wait for Level-2 OCR location boxes and scroll straight to exact position
            locateData = await locatePromise;
            if (renderToken !== citationDrawerState.renderToken) {
                return;
            }
            pdfLocateCache.set(locateKey, locateData);
            if (locateData && locateData.pages) {
                activeLocatePages = locateData.pages;
            } else {
                activeLocatePages = {};
            }
            paintOverlaysOnPages(activeLocatePages, activeSearchMatches);

            const primaryInfo = activeLocatePages[String(targetPages[0])];
            if (primaryInfo && Array.isArray(primaryInfo.boxes) && primaryInfo.boxes.length > 0) {
                scrollToPageHighlight(targetPages[0], primaryInfo.focus_box);
                setDocStatus("");
            } else {
                setDocStatus("Đã mở trang " + targetPages[0] + " / " + totalPages + " của tài liệu PDF.");
            }
        }

        async function renderDocxReader(item, rawSource, renderToken) {
            setDocStatus("Đang tải toàn bộ nội dung file Word (.docx)...");
            const cacheKey = "docx:" + rawSource;
            let data = docPreviewCache.get(cacheKey);
            if (!data) {
                try {
                    const url = "/api/documents/docx/preview?source="
                        + encodeURIComponent(rawSource)
                        + "&chunk_id="
                        + encodeURIComponent(String(item.chunk_id || ""));
                    const resp = await fetch(url);
                    if (!resp.ok) {
                        throw new Error("Không tải được file Word (" + resp.status + ")");
                    }
                    data = await resp.json();
                    docPreviewCache.set(cacheKey, data);
                } catch (err) {
                    if (renderToken !== citationDrawerState.renderToken) {
                        return;
                    }
                    setDocStatus(err.message || "Không thể đọc file Word.", true);
                    return;
                }
            }
            if (renderToken !== citationDrawerState.renderToken) {
                return;
            }

            docViewport.innerHTML = data.html || "";

            // Build Heading Outline
            let outlineList = [];
            if (Array.isArray(data.outline)) {
                outlineList = data.outline;
            }
            docOutline.replaceChildren();
            if (outlineList.length > 0) {
                const outlineTitle = document.createElement("div");
                outlineTitle.className = "doc-outline-title";
                outlineTitle.textContent = "Mục lục văn bản";
                docOutline.appendChild(outlineTitle);
                for (const hd of outlineList) {
                    const btn = document.createElement("button");
                    btn.type = "button";
                    btn.className = "doc-outline-item lvl-" + Math.min(4, Math.max(1, Number(hd.level || 1)));
                    btn.textContent = hd.text;
                    btn.addEventListener("click", () => {
                        const target = findDocxNodeByXPath(hd.xpath);
                        if (target) {
                            target.scrollIntoView({ behavior: "smooth", block: "start" });
                        }
                    });
                    docOutline.appendChild(btn);
                }
                docOutline.hidden = false;
            } else {
                docOutline.hidden = true;
            }

            function findDocxNodeByXPath(xpathStr) {
                if (!xpathStr) {
                    return null;
                }
                const allNodes = docViewport.querySelectorAll("[data-xpath]");
                for (const node of allNodes) {
                    if (node.getAttribute("data-xpath") === xpathStr) {
                        return node;
                    }
                }
                // Strip run/drawing suffix down to parent w:p[...] or w:tbl[...]
                let trimmed = String(xpathStr);
                while (trimmed.indexOf("/") >= 0) {
                    const lastSlash = trimmed.lastIndexOf("/");
                    if (lastSlash <= 0) {
                        break;
                    }
                    trimmed = trimmed.slice(0, lastSlash);
                    for (const node of allNodes) {
                        if (node.getAttribute("data-xpath") === trimmed) {
                            return node;
                        }
                    }
                }
                return null;
            }

            // Locate & highlight exact XPaths from provenance (or fallback by excerpt matching)
            const xpaths = extractCitationXPaths(item, data.fallback_xpaths);
            const matchedElements = [];
            const seenEls = new Set();
            for (const xp of xpaths) {
                const node = findDocxNodeByXPath(xp);
                if (node && !seenEls.has(node)) {
                    seenEls.add(node);
                    node.classList.add("doc-cited-highlight");
                    matchedElements.push(node);
                }
            }

            if (matchedElements.length === 0 && item.excerpt) {
                const targetLines = String(item.excerpt)
                    .split("\n")
                    .map(foldClientText)
                    .filter((s) => s.length >= 16);
                const blocks = docViewport.querySelectorAll(".docx-block, .docx-table tr");
                for (const blk of blocks) {
                    const blkFold = foldClientText(blk.textContent || "");
                    if (blkFold.length < 10) {
                        continue;
                    }
                    let hit = false;
                    for (const tl of targetLines) {
                        if (blkFold.indexOf(tl.slice(0, 48)) >= 0 || tl.indexOf(blkFold.slice(0, 48)) >= 0) {
                            hit = true;
                            break;
                        }
                    }
                    if (hit && !seenEls.has(blk)) {
                        seenEls.add(blk);
                        blk.classList.add("doc-cited-highlight");
                        matchedElements.push(blk);
                    }
                }
            }

            // Build Word Toolbar
            docToolbar.replaceChildren();
            const leftGroup = document.createElement("div");
            leftGroup.className = "doc-toolbar-group";

            const focusBtn = document.createElement("button");
            focusBtn.type = "button";
            focusBtn.className = "doc-tool-btn doc-tool-primary";
            focusBtn.textContent = "Tới vị trí trích dẫn";
            focusBtn.disabled = matchedElements.length === 0;
            focusBtn.addEventListener("click", () => {
                if (matchedElements.length > 0) {
                    matchedElements[0].scrollIntoView({ behavior: "smooth", block: "center" });
                }
            });
            leftGroup.appendChild(focusBtn);

            if (outlineList.length > 0) {
                const outlineBtn = document.createElement("button");
                outlineBtn.type = "button";
                outlineBtn.className = "doc-tool-btn";
                outlineBtn.textContent = "Mục lục (" + outlineList.length + ")";
                outlineBtn.setAttribute("aria-pressed", "true");
                outlineBtn.addEventListener("click", () => {
                    docOutline.hidden = !docOutline.hidden;
                    if (docOutline.hidden) {
                        outlineBtn.setAttribute("aria-pressed", "false");
                    } else {
                        outlineBtn.setAttribute("aria-pressed", "true");
                    }
                });
                leftGroup.appendChild(outlineBtn);
            }

            const rightGroup = document.createElement("div");
            rightGroup.className = "doc-toolbar-group";

            const searchWrap = document.createElement("div");
            searchWrap.className = "doc-search-wrap";
            const searchInput = document.createElement("input");
            searchInput.type = "search";
            searchInput.className = "doc-search-input";
            searchInput.placeholder = "Tìm trong Word...";
            searchInput.setAttribute("aria-label", "Tìm kiếm trong file Word");
            const searchCount = document.createElement("span");
            searchCount.className = "doc-search-count";
            const searchNextBtn = document.createElement("button");
            searchNextBtn.type = "button";
            searchNextBtn.className = "doc-tool-btn";
            searchNextBtn.textContent = "↓";
            searchNextBtn.hidden = true;
            searchWrap.append(searchInput, searchCount, searchNextBtn);

            let wordSearchHits = [];
            let wordSearchIdx = 0;

            function runWordSearch() {
                for (const prevHit of docViewport.querySelectorAll(".doc-search-highlight")) {
                    prevHit.classList.remove("doc-search-highlight");
                }
                wordSearchHits = [];
                wordSearchIdx = 0;
                const qFold = foldClientText(searchInput.value);
                if (qFold.length < 2) {
                    searchCount.textContent = "";
                    searchNextBtn.hidden = true;
                    return;
                }
                for (const blk of docViewport.querySelectorAll(".docx-block, .docx-table tr")) {
                    if (foldClientText(blk.textContent || "").indexOf(qFold) >= 0) {
                        blk.classList.add("doc-search-highlight");
                        wordSearchHits.push(blk);
                    }
                }
                if (wordSearchHits.length === 0) {
                    searchCount.textContent = "0 kq";
                    searchNextBtn.hidden = true;
                } else {
                    searchCount.textContent = "1/" + wordSearchHits.length;
                    searchNextBtn.hidden = wordSearchHits.length <= 1;
                    wordSearchHits[0].scrollIntoView({ behavior: "smooth", block: "center" });
                }
            }

            searchInput.addEventListener("input", runWordSearch);
            searchNextBtn.addEventListener("click", () => {
                if (wordSearchHits.length === 0) {
                    return;
                }
                wordSearchIdx = (wordSearchIdx + 1) % wordSearchHits.length;
                searchCount.textContent = (wordSearchIdx + 1) + "/" + wordSearchHits.length;
                wordSearchHits[wordSearchIdx].scrollIntoView({ behavior: "smooth", block: "center" });
            });

            const downloadLink = document.createElement("a");
            downloadLink.className = "doc-tool-btn";
            downloadLink.href = buildRawFileUrl(item);
            downloadLink.target = "_blank";
            downloadLink.rel = "noopener noreferrer";
            downloadLink.textContent = "Tải file Word ↗";

            rightGroup.append(searchWrap, downloadLink);
            docToolbar.append(leftGroup, rightGroup);

            if (matchedElements.length > 0) {
                setDocStatus("");
                matchedElements[0].scrollIntoView({ behavior: "auto", block: "center" });
            } else {
                docViewport.scrollTop = 0;
                setDocStatus("Đã mở toàn bộ file Word (.docx).");
            }
        }

        async function renderXlsxReader(item, rawSource, renderToken) {
            docOutline.hidden = true;
            docOutline.replaceChildren();
            setDocStatus("Đang tải dữ liệu bảng tính Excel (.xlsx)...");

            const targetInfo = extractCitationExcelTarget(item);
            const url = "/api/documents/xlsx/preview?source="
                + encodeURIComponent(rawSource)
                + "&sheet="
                + encodeURIComponent(targetInfo.sheetName)
                + "&range="
                + encodeURIComponent(targetInfo.rangeText)
                + "&chunk_id="
                + encodeURIComponent(String(item.chunk_id || ""));

            let data = null;
            try {
                const resp = await fetch(url);
                if (!resp.ok) {
                    throw new Error("Không tải được file Excel (" + resp.status + ")");
                }
                data = await resp.json();
            } catch (err) {
                if (renderToken !== citationDrawerState.renderToken) {
                    return;
                }
                setDocStatus(err.message || "Không thể đọc file Excel.", true);
                return;
            }
            if (renderToken !== citationDrawerState.renderToken) {
                return;
            }

            let sheets = [];
            if (Array.isArray(data.sheets)) {
                sheets = data.sheets;
            }
            if (sheets.length === 0) {
                docViewport.replaceChildren();
                setDocStatus("File Excel không có sheet hiển thị.", true);
                return;
            }

            let activeSheetIdx = 0;
            const resolvedSheetName = String(data.target_sheet || targetInfo.sheetName || "").trim();
            if (resolvedSheetName) {
                for (let idx = 0; idx < sheets.length; idx++) {
                    if (sheets[idx].name === resolvedSheetName) {
                        activeSheetIdx = idx;
                        break;
                    }
                }
            }
            const targetRange = data.target_range || null;

            const shell = document.createElement("div");
            shell.className = "xlsx-viewer-shell";
            const sheetTabBar = document.createElement("div");
            sheetTabBar.className = "xlsx-sheet-tabs";
            sheetTabBar.setAttribute("role", "tablist");
            sheetTabBar.setAttribute("aria-label", "Danh sách Sheet Excel");

            const tableWrapper = document.createElement("div");
            tableWrapper.className = "xlsx-table-wrapper";
            tableWrapper.setAttribute("role", "region");
            tableWrapper.setAttribute("tabindex", "0");
            tableWrapper.setAttribute("aria-label", "Nội dung bảng tính Excel");

            shell.append(sheetTabBar, tableWrapper);
            docViewport.replaceChildren(shell);

            let firstCitedCellEl = null;

            function renderSheet(idx) {
                activeSheetIdx = idx;
                const sheet = sheets[idx];
                const tabBtns = sheetTabBar.querySelectorAll(".xlsx-sheet-tab");
                for (let t = 0; t < tabBtns.length; t++) {
                    if (t === idx) {
                        tabBtns[t].setAttribute("aria-selected", "true");
                        const targetLeft = Math.max(
                            0,
                            tabBtns[t].offsetLeft - Math.round((sheetTabBar.clientWidth - tabBtns[t].offsetWidth) * 0.5)
                        );
                        sheetTabBar.scrollLeft = targetLeft;
                    } else {
                        tabBtns[t].setAttribute("aria-selected", "false");
                    }
                }

                const mergeLookup = new Map();
                const skippedCells = new Set();
                for (const m of sheet.merges || []) {
                    const key = m.min_row + ":" + m.min_col;
                    mergeLookup.set(key, {
                        rowspan: (m.max_row - m.min_row) + 1,
                        colspan: (m.max_col - m.min_col) + 1
                    });
                    for (let mr = m.min_row; mr <= m.max_row; mr++) {
                        for (let mc = m.min_col; mc <= m.max_col; mc++) {
                            if (mr !== m.min_row || mc !== m.min_col) {
                                skippedCells.add(mr + ":" + mc);
                            }
                        }
                    }
                }

                const tbl = document.createElement("table");
                tbl.className = "xlsx-grid-table";

                const thead = document.createElement("thead");
                const headTr = document.createElement("tr");
                const cornerTh = document.createElement("th");
                cornerTh.className = "xlsx-corner-th";
                cornerTh.textContent = "#";
                headTr.appendChild(cornerTh);
                for (const ch of sheet.col_headers || []) {
                    const th = document.createElement("th");
                    th.textContent = ch.label;
                    headTr.appendChild(th);
                }
                thead.appendChild(headTr);
                tbl.appendChild(thead);

                const tbody = document.createElement("tbody");
                firstCitedCellEl = null;
                const matchesTargetSheet = !resolvedSheetName || sheet.name === resolvedSheetName;

                for (const rowObj of sheet.rows || []) {
                    const tr = document.createElement("tr");
                    const rowTh = document.createElement("th");
                    rowTh.className = "xlsx-row-th";
                    rowTh.textContent = String(rowObj.row_num);
                    tr.appendChild(rowTh);

                    for (const cellObj of rowObj.cells || []) {
                        const coordKey = cellObj.r + ":" + cellObj.c;
                        if (skippedCells.has(coordKey)) {
                            continue;
                        }
                        const td = document.createElement("td");
                        const cellInner = document.createElement("div");
                        cellInner.className = "xlsx-cell-inner";
                        cellInner.textContent = cellObj.v;
                        td.appendChild(cellInner);
                        td.setAttribute("data-a1", cellObj.a1);
                        td.title = sheet.name + "!" + cellObj.a1;
                        if (cellObj.b) {
                            td.classList.add("is-bold");
                        }
                        if (cellObj.al === "right") {
                            td.classList.add("align-right");
                        } else if (cellObj.al === "center") {
                            td.classList.add("align-center");
                        }
                        const mInfo = mergeLookup.get(coordKey);
                        if (mInfo) {
                            if (mInfo.rowspan > 1) {
                                td.rowSpan = mInfo.rowspan;
                            }
                            if (mInfo.colspan > 1) {
                                td.colSpan = mInfo.colspan;
                            }
                        }
                        if (matchesTargetSheet && targetRange) {
                            if (
                                cellObj.r >= targetRange.min_row
                                && cellObj.r <= targetRange.max_row
                                && cellObj.c >= targetRange.min_col
                                && cellObj.c <= targetRange.max_col
                            ) {
                                td.classList.add("xlsx-cell-cited");
                                if (!firstCitedCellEl) {
                                    firstCitedCellEl = td;
                                }
                            }
                        }
                        tr.appendChild(td);
                    }
                    tbody.appendChild(tr);
                }
                tbl.appendChild(tbody);
                tableWrapper.replaceChildren(tbl);

                if (firstCitedCellEl) {
                    firstCitedCellEl.scrollIntoView({ behavior: "auto", block: "nearest", inline: "nearest" });
                } else {
                    tableWrapper.scrollTop = 0;
                }
            }

            for (let idx = 0; idx < sheets.length; idx++) {
                const sBtn = document.createElement("button");
                sBtn.type = "button";
                sBtn.role = "tab";
                sBtn.className = "xlsx-sheet-tab";
                sBtn.textContent = sheets[idx].name;
                if (idx === activeSheetIdx) {
                    sBtn.setAttribute("aria-selected", "true");
                } else {
                    sBtn.setAttribute("aria-selected", "false");
                }
                sBtn.addEventListener("click", () => {
                    renderSheet(idx);
                });
                sheetTabBar.appendChild(sBtn);
            }

            // Build Excel Toolbar
            docToolbar.replaceChildren();
            const leftGroup = document.createElement("div");
            leftGroup.className = "doc-toolbar-group";

            const focusBtn = document.createElement("button");
            focusBtn.type = "button";
            focusBtn.className = "doc-tool-btn doc-tool-primary";
            focusBtn.textContent = "Tới ô trích dẫn";
            focusBtn.addEventListener("click", () => {
                if (resolvedSheetName) {
                    for (let idx = 0; idx < sheets.length; idx++) {
                        if (sheets[idx].name === resolvedSheetName) {
                            renderSheet(idx);
                            break;
                        }
                    }
                }
                if (firstCitedCellEl) {
                    firstCitedCellEl.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
                }
            });
            leftGroup.appendChild(focusBtn);

            if (targetInfo.rangeText) {
                const rangeBadge = document.createElement("span");
                rangeBadge.className = "doc-toolbar-label";
                let sheetPrefix = "";
                if (resolvedSheetName) {
                    sheetPrefix = resolvedSheetName + "!";
                }
                rangeBadge.textContent = "Vùng: " + sheetPrefix + targetInfo.rangeText;
                leftGroup.appendChild(rangeBadge);
            }

            const rightGroup = document.createElement("div");
            rightGroup.className = "doc-toolbar-group";

            const searchWrap = document.createElement("div");
            searchWrap.className = "doc-search-wrap";
            const searchInput = document.createElement("input");
            searchInput.type = "search";
            searchInput.className = "doc-search-input";
            searchInput.placeholder = "Tìm trong Sheet...";
            searchInput.setAttribute("aria-label", "Tìm kiếm trong Sheet Excel");
            const searchCount = document.createElement("span");
            searchCount.className = "doc-search-count";
            const searchNextBtn = document.createElement("button");
            searchNextBtn.type = "button";
            searchNextBtn.className = "doc-tool-btn";
            searchNextBtn.textContent = "↓";
            searchNextBtn.hidden = true;
            searchWrap.append(searchInput, searchCount, searchNextBtn);

            let xlHits = [];
            let xlHitIdx = 0;

            function runExcelSearch() {
                for (const prev of tableWrapper.querySelectorAll(".xlsx-cell-search")) {
                    prev.classList.remove("xlsx-cell-search");
                }
                xlHits = [];
                xlHitIdx = 0;
                const qFold = foldClientText(searchInput.value);
                if (qFold.length < 2) {
                    searchCount.textContent = "";
                    searchNextBtn.hidden = true;
                    return;
                }
                for (const td of tableWrapper.querySelectorAll("tbody td")) {
                    if (foldClientText(td.textContent || "").indexOf(qFold) >= 0) {
                        td.classList.add("xlsx-cell-search");
                        xlHits.push(td);
                    }
                }
                if (xlHits.length === 0) {
                    searchCount.textContent = "0 kq";
                    searchNextBtn.hidden = true;
                } else {
                    searchCount.textContent = "1/" + xlHits.length;
                    searchNextBtn.hidden = xlHits.length <= 1;
                    xlHits[0].scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
                }
            }

            searchInput.addEventListener("input", runExcelSearch);
            searchNextBtn.addEventListener("click", () => {
                if (xlHits.length === 0) {
                    return;
                }
                xlHitIdx = (xlHitIdx + 1) % xlHits.length;
                searchCount.textContent = (xlHitIdx + 1) + "/" + xlHits.length;
                xlHits[xlHitIdx].scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
            });

            const downloadLink = document.createElement("a");
            downloadLink.className = "doc-tool-btn";
            downloadLink.href = buildRawFileUrl(item);
            downloadLink.target = "_blank";
            downloadLink.rel = "noopener noreferrer";
            downloadLink.textContent = "Tải file Excel ↗";

            rightGroup.append(searchWrap, downloadLink);
            docToolbar.append(leftGroup, rightGroup);

            renderSheet(activeSheetIdx);
            setDocStatus("");
        }

        function renderActiveCitation() {
            const citations = citationDrawerState.citations;
            const pos = citationDrawerState.activePos;
            const item = citations[pos];
            if (!item) {
                return;
            }
            citationDrawerState.renderToken += 1;
            const renderToken = citationDrawerState.renderToken;
            clearActiveObserver();

            let badgeText = "Nguồn trích dẫn";
            if (item.label) {
                badgeText = item.label;
            } else if (item.index) {
                badgeText = "[Nguồn " + item.index + "]";
            }
            badgeEl.textContent = badgeText;
            typeEl.textContent = chunkTypeLabel(item.chunk_type);

            let docTitle = item.document_name;
            if (!docTitle && item.source) {
                docTitle = sourceFileName(item.source);
            }
            if (!docTitle) {
                docTitle = "Chi tiết tài liệu trích dẫn";
            }
            titleEl.textContent = docTitle;

            const fileType = resolveCitationFileType(item);
            fileKindEl.className = "citation-file-kind-pill";
            if (fileType === "pdf") {
                fileKindEl.textContent = "PDF";
                fileKindEl.classList.add("kind-pdf");
                fileKindEl.hidden = false;
            } else if (fileType === "docx") {
                fileKindEl.textContent = "WORD";
                fileKindEl.classList.add("kind-docx");
                fileKindEl.hidden = false;
            } else if (fileType === "xlsx") {
                fileKindEl.textContent = "EXCEL";
                fileKindEl.classList.add("kind-xlsx");
                fileKindEl.hidden = false;
            } else {
                fileKindEl.hidden = true;
            }

            prevBtn.disabled = pos <= 0;
            nextBtn.disabled = pos >= citations.length - 1;
            const hideNav = citations.length <= 1;
            prevBtn.hidden = hideNav;
            nextBtn.hidden = hideNav;

            tabsRow.replaceChildren();
            if (citations.length > 1) {
                tabsRow.hidden = false;
                for (let idx = 0; idx < citations.length; idx++) {
                    const tabItem = citations[idx];
                    const tabBtn = document.createElement("button");
                    tabBtn.type = "button";
                    tabBtn.role = "tab";
                    tabBtn.className = "citation-viewer-tab";
                    if (idx === pos) {
                        tabBtn.setAttribute("aria-selected", "true");
                        tabBtn.setAttribute("focusgroupstart", "");
                    } else {
                        tabBtn.setAttribute("aria-selected", "false");
                    }
                    if (tabItem.label) {
                        tabBtn.textContent = tabItem.label;
                    } else if (tabItem.index) {
                        tabBtn.textContent = "[Nguồn " + tabItem.index + "]";
                    } else {
                        tabBtn.textContent = "Nguồn " + (idx + 1);
                    }
                    tabBtn.addEventListener("click", () => {
                        citationDrawerState.activePos = idx;
                        renderActiveCitation();
                    });
                    tabsRow.appendChild(tabBtn);
                }
            } else {
                tabsRow.hidden = true;
            }

            renderEvidencePane(item);

            const rawSource = String(item.source || "").split("\\").join("/").trim();
            docToolbar.replaceChildren();
            docViewport.replaceChildren();
            docOutline.hidden = true;
            docOutline.replaceChildren();

            const legalSource = legalCitationUrl(item.source_url);
            if (legalSource) {
                const link = document.createElement("a");
                link.className = "doc-tool-btn";
                link.href = legalSource;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                link.textContent = "Mở văn bản trên Thư Viện Pháp Luật ↗";
                docToolbar.append(link);
                setDocStatus("Đoạn trích từ kết quả tìm kiếm web.");
                return;
            }
            if (!rawSource || !fileType) {
                setDocStatus("Nguồn trích dẫn này không gắn với tệp PDF, Word hoặc Excel gốc.");
                return;
            }
            if (fileType === "pdf") {
                renderPdfReader(item, rawSource, renderToken);
            } else if (fileType === "docx") {
                renderDocxReader(item, rawSource, renderToken);
            } else if (fileType === "xlsx") {
                renderXlsxReader(item, rawSource, renderToken);
            }
        }

        toggleEvidenceBtn.addEventListener("click", () => {
            citationDrawerState.evidenceCollapsed = !citationDrawerState.evidenceCollapsed;
            if (citationDrawerState.evidenceCollapsed) {
                workspace.classList.add("evidence-collapsed");
                toggleEvidenceBtn.setAttribute("aria-pressed", "true");
                toggleEvidenceBtn.textContent = "Hiện đoạn Chunk";
            } else {
                workspace.classList.remove("evidence-collapsed");
                toggleEvidenceBtn.setAttribute("aria-pressed", "false");
                toggleEvidenceBtn.textContent = "Mở rộng tài liệu";
            }
        });

        prevBtn.addEventListener("click", () => {
            if (citationDrawerState.activePos > 0) {
                citationDrawerState.activePos = citationDrawerState.activePos - 1;
                renderActiveCitation();
            }
        });
        nextBtn.addEventListener("click", () => {
            if (citationDrawerState.activePos < citationDrawerState.citations.length - 1) {
                citationDrawerState.activePos = citationDrawerState.activePos + 1;
                renderActiveCitation();
            }
        });

        copyBtn.addEventListener("click", () => {
            const active = citationDrawerState.citations[citationDrawerState.activePos];
            if (!active || !active.excerpt) {
                return;
            }
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(active.excerpt).then(() => {
                    copyBtn.textContent = "Đã sao chép";
                    setTimeout(() => {
                        copyBtn.textContent = "Sao chép";
                    }, 1800);
                }).catch(() => {});
            }
        });

        dialog.addEventListener("close", () => {
            clearActiveObserver();
        });
        attachDialogLightDismissFallback(dialog);

        citationViewerElements = { dialog, renderActiveCitation };
        return citationViewerElements;
    }

    function openCitationDrawer(citationsList, activePos) {
        if (!Array.isArray(citationsList) || citationsList.length === 0) {
            return;
        }
        const viewer = ensureCitationViewer();
        let pos = activePos;
        if (typeof pos !== "number" || pos < 0 || pos >= citationsList.length) {
            pos = 0;
        }
        citationDrawerState.citations = citationsList;
        citationDrawerState.activePos = pos;
        if (!viewer.dialog.open) {
            viewer.dialog.showModal();
        }
        viewer.renderActiveCitation();
    }

    function sourceFileName(source) {
        return String(source || "").split("\\").join("/").split("/").pop();
    }

    function legalCitationUrl(source) {
        if (!source) {
            return "";
        }
        try {
            const url = new URL(source);
            if (url.origin === "https://thuvienphapluat.vn" && !url.username && !url.password
                && url.pathname.toLowerCase().startsWith("/van-ban/") && url.pathname.toLowerCase().endsWith(".aspx")) {
                return url.href;
            }
        } catch {
            return "";
        }
        return "";
    }

    function getMessageReferences(meta) {
        const citations = [];
        const fallbackSources = [];
        for (const citation of meta?.citations || []) {
            if (typeof citation === "string") {
                fallbackSources.push(citation);
            } else if (citation && (citation.source || citation.index || citation.label || citation.document_name)) {
                citations.push(citation);
            }
        }
        if (citations.length === 0) {
            fallbackSources.push(...(meta?.sources || []));
        }
        return { citations, fallbackSources };
    }

    function formatAnswerWithReferences(answer, { citations, fallbackSources }) {
        const entries = citations.map(citation => {
            let label = citation.label || "";
            if (!label && citation.index) {
                label = "[Nguồn " + citation.index + "]";
            }
            const name = citation.document_name || sourceFileName(citation.source) || "Tài liệu";
            let entry = [label, name].filter(Boolean).join(" ");
            if (Array.isArray(citation.page_numbers) && citation.page_numbers.length > 0) {
                entry += " · Trang " + citation.page_numbers.join(", ");
            }
            const legalSource = legalCitationUrl(citation.source_url);
            if (legalSource) {
                entry += " · " + legalSource;
            }
            return entry;
        });
        if (entries.length === 0) {
            entries.push(...fallbackSources.map(sourceFileName).filter(Boolean));
        }
        if (entries.length === 0) {
            return answer;
        }
        return answer + "\n\nTài liệu trích dẫn\n- " + Array.from(new Set(entries)).join("\n- ");
    }

    function buildSourcesCardElement(citationsList, fallbackSources) {
        if (citationsList.length === 0 && (!Array.isArray(fallbackSources) || fallbackSources.length === 0)) {
            return null;
        }
        const card = document.createElement("div");
        card.className = "sources-card";

        const header = document.createElement("div");
        header.className = "sources-header";
        if (citationsList.length > 0) {
            header.textContent = "Tài liệu đã trích dẫn (nhấn để xem đoạn gốc)";
        } else {
            header.textContent = "Tài liệu đã trích dẫn";
        }
        card.appendChild(header);

        const pills = document.createElement("div");
        pills.className = "sources-pills";

        if (citationsList.length > 0) {
            for (let pos = 0; pos < citationsList.length; pos++) {
                const cit = citationsList[pos];
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = "source-tag source-tag-clickable";

                let docDisplay = cit.document_name;
                if (!docDisplay && cit.source) {
                    docDisplay = sourceFileName(cit.source);
                }
                if (!docDisplay) {
                    docDisplay = "Tài liệu";
                }
                if (docDisplay.length > 46) {
                    docDisplay = docDisplay.slice(0, 43) + "...";
                }
                let labelPrefix = "";
                if (cit.label) {
                    labelPrefix = cit.label + " ";
                } else if (cit.index) {
                    labelPrefix = "[Nguồn " + cit.index + "] ";
                }
                let pageSuffix = "";
                if (Array.isArray(cit.page_numbers) && cit.page_numbers.length > 0) {
                    pageSuffix = " · Tr. " + cit.page_numbers.join(", ");
                }
                btn.textContent = labelPrefix + docDisplay + pageSuffix;
                btn.title = "Nhấn để xem chi tiết " + labelPrefix + docDisplay;
                btn.addEventListener("click", () => {
                    openCitationDrawer(citationsList, pos);
                });
                pills.appendChild(btn);
            }
        } else {
            const uniqueSources = Array.from(new Set(fallbackSources)).slice(0, 6);
            for (const src of uniqueSources) {
                const span = document.createElement("span");
                span.className = "source-tag";
                span.textContent = String(src);
                pills.appendChild(span);
            }
        }

        card.appendChild(pills);
        return card;
    }

    function openTraceInInspector(traceId) {
        if (traceId && window.ABankDebugInspector) {
            window.ABankDebugInspector.openTrace(String(traceId));
        }
    }

    function appendMessage(sender, content, meta = null) {
        exitWelcomeState();
        const msgDiv = document.createElement("div");
        msgDiv.className = "message " + sender + "-message";

        if (sender === "user") {
            const bubbleEl = document.createElement("div");
            bubbleEl.className = "bubble";

            if (meta && Array.isArray(meta.userAttachments) && meta.userAttachments.length > 0) {
                const thumbsGrid = document.createElement("div");
                thumbsGrid.className = "user-attachments-grid";
                const lightboxItems = [];
                for (let idx = 0; idx < meta.userAttachments.length; idx++) {
                    const att = meta.userAttachments[idx];
                    if (!att || !att.data_url) {
                        continue;
                    }
                    let attName = "Ảnh đính kèm " + (idx + 1);
                    if (att.name) {
                        attName = att.name;
                    }
                    lightboxItems.push({
                        url: att.data_url,
                        caption: attName,
                        region_type: "attachment"
                    });
                }
                for (let idx = 0; idx < lightboxItems.length; idx++) {
                    const lbItem = lightboxItems[idx];
                    const thumbBtn = document.createElement("button");
                    thumbBtn.type = "button";
                    thumbBtn.className = "user-attachment-thumb";
                    thumbBtn.title = "Phóng to " + lbItem.caption;
                    const thumbImg = document.createElement("img");
                    thumbImg.src = lbItem.url;
                    thumbImg.alt = lbItem.caption;
                    thumbBtn.appendChild(thumbImg);
                    thumbBtn.addEventListener("click", () => {
                        openImageLightbox(lightboxItems, idx);
                    });
                    thumbsGrid.appendChild(thumbBtn);
                }
                if (thumbsGrid.childElementCount > 0) {
                    bubbleEl.appendChild(thumbsGrid);
                }
            }

            if (content) {
                const textPara = document.createElement("p");
                textPara.textContent = content;
                bubbleEl.appendChild(textPara);
            }
            msgDiv.appendChild(bubbleEl);
        } else {
            const references = getMessageReferences(meta);
            const { citations: citationList, fallbackSources } = references;
            const citationByIndex = new Map();
            citationList.forEach((citation, position) => {
                if (typeof citation.index === "number") {
                    citationByIndex.set(citation.index, position);
                }
            });

            const bubbleEl = document.createElement("div");
            bubbleEl.className = "bubble";
            bubbleEl.innerHTML = parseMarkdown(content, citationByIndex);

            if (citationList.length > 0) {
                bubbleEl.addEventListener("click", (event) => {
                    const btn = event.target.closest(".inline-citation-btn");
                    if (!btn) {
                        return;
                    }
                    const citeIdx = Number(btn.getAttribute("data-cite-idx"));
                    if (citationByIndex.has(citeIdx)) {
                        openCitationDrawer(citationList, citationByIndex.get(citeIdx));
                    }
                });
            }

            if (meta && Array.isArray(meta.images) && meta.images.length > 0) {
                const galleryEl = buildEvidenceImagesElement(meta.images);
                if (galleryEl) {
                    bubbleEl.appendChild(galleryEl);
                }
            }

            const sourcesCardEl = buildSourcesCardElement(citationList, fallbackSources);
            if (sourcesCardEl) {
                bubbleEl.appendChild(sourcesCardEl);
            }

            msgDiv.appendChild(bubbleEl);
            if (content && meta && meta.audit_status !== "ERROR") {
                const copyBtn = document.createElement("button");
                copyBtn.type = "button";
                copyBtn.className = "message-copy-btn";
                copyBtn.textContent = "Sao chép câu trả lời";
                copyBtn.addEventListener("click", async () => {
                    try {
                        await navigator.clipboard.writeText(formatAnswerWithReferences(content, references));
                        showComposerStatus("Đã sao chép câu trả lời.", "info", 2500);
                    } catch {
                        showComposerStatus("Không thể sao chép. Vui lòng kiểm tra quyền truy cập bộ nhớ tạm.", "error", 4000);
                    }
                });
                msgDiv.appendChild(copyBtn);
            }
        }

        chatMessages.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    function setBusy(value) {
        busy = value;
        const locked = busy || Boolean(pendingCard);
        userInput.disabled = locked;
        sendBtn.disabled = locked;
        if (attachImageBtn) {
            attachImageBtn.disabled = locked;
        }
        if (voiceInputBtn) {
            voiceInputBtn.disabled = locked;
        }
        if (locked && isRecording) {
            stopVoiceRecording();
        }
        if (newSessionBtn) {
            newSessionBtn.disabled = busy;
        }
    }

    function createCardSvg(className, paths) {
        const ns = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(ns, "svg");
        svg.setAttribute("class", className);
        svg.setAttribute("viewBox", "0 0 24 24");
        svg.setAttribute("fill", "none");
        svg.setAttribute("stroke", "currentColor");
        svg.setAttribute("stroke-width", "1.9");
        svg.setAttribute("stroke-linecap", "round");
        svg.setAttribute("stroke-linejoin", "round");
        svg.setAttribute("aria-hidden", "true");
        for (const item of paths) {
            const node = document.createElementNS(ns, item.tag);
            for (const [key, val] of Object.entries(item.attrs)) {
                node.setAttribute(key, val);
            }
            svg.appendChild(node);
        }
        return svg;
    }

    function renderInteraction(data) {
        const card = data.interaction;
        if (pendingCard && pendingCard.id === card.id) {
            return;
        }
        if (pendingCard) {
            pendingCard.element.remove();
        }
        if (nodeStepper) {
            nodeStepper.style.display = "block";
        }
        renderPlan(data.plan, false);
        finishProgressForStatus("AWAITING_INPUT", "awaiting_input");
        exitWelcomeState();

        const isMulti = card.multi_select === true;
        const element = document.createElement("div");
        element.className = "message bot-message bot-message-interaction";

        const bubble = document.createElement("div");
        bubble.className = "bubble interaction-card";

        const header = document.createElement("div");
        header.className = "interaction-header";
        const headerIcon = createCardSvg("interaction-header-icon", [
            {
                tag: "path",
                attrs: {
                    d: "M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"
                }
            }
        ]);
        const title = document.createElement("h3");
        title.className = "interaction-title";
        title.id = "interaction-title-" + card.id;
        title.textContent = card.message;
        header.append(headerIcon, title);
        bubble.appendChild(header);

        const subtitle = document.createElement("p");
        subtitle.className = "interaction-subtitle";
        if (card.kind === "confirm") {
            subtitle.textContent = "Vui lòng xem phạm vi và xác nhận phương án bên dưới.";
        } else if (isMulti) {
            subtitle.textContent = "Vui lòng chọn một hoặc nhiều phương án bên dưới.";
        } else {
            subtitle.textContent = "Vui lòng chọn một trong các phương án bên dưới.";
        }
        bubble.appendChild(subtitle);

        if (Array.isArray(card.scope) && card.scope.length) {
            const scopeList = document.createElement("ol");
            scopeList.className = "interaction-scope";
            for (const item of card.scope) {
                const entry = document.createElement("li");
                entry.textContent = item.task;
                scopeList.appendChild(entry);
            }
            bubble.appendChild(scopeList);
        }

        const form = document.createElement("form");
        form.className = "interaction-form";
        form.setAttribute("aria-labelledby", title.id);
        form.noValidate = true;

        const fieldset = document.createElement("fieldset");
        fieldset.className = "interaction-fieldset";
        const legend = document.createElement("legend");
        legend.className = "visually-hidden";
        legend.textContent = "Danh sách lựa chọn";
        fieldset.appendChild(legend);

        const optionRows = [];
        let cancelOption = null;
        let hasOtherOption = false;

        for (const option of card.options) {
            if (option.id === "cancel") {
                cancelOption = option;
                continue;
            }
            if (option.id === "other") {
                hasOtherOption = true;
            }
            const inputId = "opt-" + card.id + "-" + option.id;
            const row = document.createElement("label");
            row.className = "interaction-option-row";
            row.htmlFor = inputId;

            const control = document.createElement("input");
            if (isMulti) {
                control.type = "checkbox";
            } else {
                control.type = "radio";
            }
            control.id = inputId;
            control.name = "interaction-choice-" + card.id;
            control.value = option.id;
            control.className = "interaction-control";

            const textSpan = document.createElement("span");
            textSpan.className = "interaction-option-text";
            textSpan.textContent = option.label;

            row.append(control, textSpan);
            fieldset.appendChild(row);
            optionRows.push({ option, row, control });
        }
        form.appendChild(fieldset);

        let otherWrap = null;
        let otherInput = null;
        if (hasOtherOption) {
            otherWrap = document.createElement("div");
            otherWrap.className = "interaction-other-wrap";
            otherWrap.hidden = true;

            const otherId = "other-" + card.id;
            const otherLabel = document.createElement("label");
            otherLabel.className = "visually-hidden";
            otherLabel.htmlFor = otherId;
            otherLabel.textContent = "Nhập lựa chọn khác";

            otherInput = document.createElement("input");
            otherInput.type = "text";
            otherInput.id = otherId;
            otherInput.className = "interaction-other-input";
            otherInput.maxLength = 500;
            otherInput.placeholder = "Nhập lựa chọn của bạn...";
            otherInput.addEventListener("keydown", event => {
                if (event.key === "Enter" && event.isComposing) {
                    event.preventDefault();
                }
            });

            otherWrap.append(otherLabel, otherInput);
            form.appendChild(otherWrap);
        }

        const divider = document.createElement("hr");
        divider.className = "interaction-divider";
        form.appendChild(divider);

        const summary = document.createElement("div");
        summary.className = "interaction-summary";
        summary.setAttribute("aria-live", "polite");
        summary.hidden = true;

        const summaryHead = document.createElement("div");
        summaryHead.className = "interaction-summary-head";
        const summaryIcon = createCardSvg("interaction-summary-icon", [
            { tag: "circle", attrs: { cx: "12", cy: "12", r: "9" } },
            { tag: "polyline", attrs: { points: "9 12 11 14 15 10" } }
        ]);
        const summaryTitle = document.createElement("strong");
        summaryHead.append(summaryIcon, summaryTitle);

        const summaryNote = document.createElement("p");
        summaryNote.className = "interaction-summary-note";
        summary.append(summaryHead, summaryNote);
        form.appendChild(summary);

        const footer = document.createElement("div");
        footer.className = "interaction-footer";

        const submitBtn = document.createElement("button");
        submitBtn.type = "submit";
        submitBtn.className = "interaction-submit-pill";
        submitBtn.textContent = "Xác nhận lựa chọn";
        footer.appendChild(submitBtn);

        const feedback = document.createElement("p");
        feedback.setAttribute("role", "status");

        if (cancelOption) {
            const cancelBtn = document.createElement("button");
            cancelBtn.type = "button";
            cancelBtn.className = "interaction-cancel-btn";
            cancelBtn.textContent = cancelOption.label;
            cancelBtn.addEventListener("click", () => {
                let payload;
                if (isMulti) {
                    payload = { interaction_id: card.id, choice_ids: ["cancel"] };
                } else {
                    payload = { interaction_id: card.id, choice_id: "cancel" };
                }
                submitInteractionReply(
                    card,
                    payload,
                    cancelOption.label,
                    false,
                    bubble,
                    feedback
                );
            });
            footer.appendChild(cancelBtn);
        }
        form.appendChild(footer);

        const syncSelectionState = () => {
            const selectedLabels = [];
            let otherChecked = false;
            for (const item of optionRows) {
                const isChecked = item.control.checked;
                item.row.classList.toggle("is-selected", isChecked);
                if (!isChecked) {
                    continue;
                }
                if (item.option.id === "other") {
                    otherChecked = true;
                    let customText = "";
                    if (otherInput) {
                        customText = otherInput.value.trim();
                    }
                    if (customText) {
                        selectedLabels.push("Khác: " + customText);
                    } else {
                        selectedLabels.push(item.option.label);
                    }
                } else {
                    selectedLabels.push(item.option.label);
                }
            }
            if (otherWrap && otherInput) {
                otherWrap.hidden = !otherChecked;
                if (!otherChecked) {
                    otherInput.removeAttribute("aria-invalid");
                }
            }
            if (selectedLabels.length === 0) {
                summary.hidden = true;
                summaryTitle.textContent = "";
                summaryNote.textContent = "";
            } else {
                summary.hidden = false;
                summaryTitle.textContent = "Đã chọn: " + selectedLabels.join("; ");
                if (otherChecked && otherInput && !otherInput.value.trim()) {
                    summaryNote.textContent = "Vui lòng nhập nội dung cho lựa chọn Khác.";
                } else if (isMulti && selectedLabels.length > 1) {
                    summaryNote.textContent = "Đã chọn " + selectedLabels.length + " phương án. Nhấn Xác nhận lựa chọn để tiếp tục.";
                } else {
                    summaryNote.textContent = "Nhấn Xác nhận lựa chọn để tiếp tục xử lý yêu cầu.";
                }
            }
            if (selectedLabels.length > 0 && (!otherChecked || (otherInput && otherInput.value.trim()))) {
                feedback.textContent = "";
            }
        };

        for (const item of optionRows) {
            item.control.addEventListener("change", () => {
                syncSelectionState();
                if (item.option.id === "other" && item.control.checked && otherInput) {
                    otherInput.focus();
                }
            });
        }
        if (otherInput) {
            otherInput.addEventListener("input", () => {
                if (otherInput.value.trim()) {
                    otherInput.removeAttribute("aria-invalid");
                }
                syncSelectionState();
            });
        }

        form.addEventListener("submit", event => {
            event.preventDefault();
            const selectedIds = [];
            const selectedLabels = [];
            for (const item of optionRows) {
                if (!item.control.checked) {
                    continue;
                }
                selectedIds.push(item.option.id);
                selectedLabels.push(item.option.label);
            }
            if (!selectedIds.length) {
                feedback.textContent = "Vui lòng chọn ít nhất một phương án bên dưới.";
                if (optionRows.length) {
                    optionRows[0].control.focus();
                }
                return;
            }
            const hasOther = selectedIds.includes("other");
            let customVal = "";
            if (otherInput) {
                customVal = otherInput.value.trim();
            }
            if (hasOther && !customVal) {
                feedback.textContent = "Vui lòng nhập nội dung cho lựa chọn Khác.";
                if (otherInput) {
                    otherInput.setAttribute("aria-invalid", "true");
                    otherInput.focus();
                }
                return;
            }
            if (otherInput) {
                otherInput.removeAttribute("aria-invalid");
            }
            const displayLabels = [];
            for (let idx = 0; idx < selectedIds.length; idx++) {
                if (selectedIds[idx] === "other") {
                    displayLabels.push(customVal);
                } else {
                    displayLabels.push(selectedLabels[idx]);
                }
            }
            let payload;
            if (isMulti) {
                payload = { interaction_id: card.id, choice_ids: selectedIds };
            } else {
                payload = { interaction_id: card.id, choice_id: selectedIds[0] };
            }
            if (hasOther) {
                payload.text = customVal;
            }
            const isOtherOnly = selectedIds.length === 1 && selectedIds[0] === "other";
            submitInteractionReply(
                card,
                payload,
                displayLabels.join("; "),
                isOtherOnly,
                bubble,
                feedback
            );
        });

        bubble.append(form, feedback);
        element.appendChild(bubble);
        pendingCard = { id: card.id, element };
        chatMessages.appendChild(element);
        setBusy(busy);
        scrollToBottom();
    }

    function renderAgentResult(data) {
        if (data && data.trace_id) {
            dagUiState.latestTraceId = String(data.trace_id);
            if (window.ABankDebugInspector) {
                window.ABankDebugInspector.onTraceCompleted(data.trace_id);
            }
        }
        if (data && data.dag_telemetry) {
            if (nodeStepper) {
                nodeStepper.style.display = "block";
            }
            applyDagTelemetry(data.dag_telemetry, false);
        }
        if (data && data.session_title) {
            applyResolvedSessionTitle(currentSessionId, data.session_title);
        }
        if (data.status === "awaiting_input") {
            finishProgressForStatus(data.audit_status || "AWAITING_INPUT", "awaiting_input", data.answer || "");
            renderInteraction(data);
            fetchSessions();
            return;
        }
        const isPassDone = Boolean(data && data.audit_status === "PASS");
        if (nodeStepper && (data.dag_telemetry || (Array.isArray(data.plan) && data.plan.length > 0) || data.answer)) {
            nodeStepper.style.display = "block";
        }
        renderPlan(data.plan, isPassDone);
        finishProgressForStatus(data && data.audit_status, data && data.status, data && data.answer);
        appendMessage("bot", data.answer || "Không nhận được phản hồi từ hệ thống.", data);
        fetchSessions();
    }

    function postAgent(path, payload) {
        return fetch(path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: currentSessionId, ...payload })
        });
    }

    async function recoverInteraction(interactionId = null, restoreResult = false) {
        try {
            const response = await postAgent("/api/agent/status", {});
            if (!response.ok) {
                return false;
            }
            const data = await response.json();
            if (data && data.session_title) {
                applyResolvedSessionTitle(currentSessionId, data.session_title);
            }
            if (data.status === "awaiting_input") {
                renderInteraction(data);
                return data.interaction.id !== interactionId;
            }
            if (data.answer && (restoreResult || (interactionId && data.accepted_interaction_id === interactionId))) {
                if (pendingCard) {
                    pendingCard.element.remove();
                    pendingCard = null;
                }
                renderAgentResult(data);
                return true;
            }
        } catch {
            return false;
        }
        return false;
    }

    function createStreamingBubbleElement(initialText = "Đang kết nối agent...") {
        const botMsgDiv = document.createElement("div");
        botMsgDiv.className = "message bot-message";
        const bubbleEl = document.createElement("div");
        bubbleEl.className = "bubble";
        const statusSpan = document.createElement("span");
        statusSpan.style.color = "#94a3b8";
        statusSpan.style.fontStyle = "italic";
        statusSpan.textContent = initialText;
        bubbleEl.appendChild(statusSpan);
        botMsgDiv.appendChild(bubbleEl);
        return { botMsgDiv, streamingBubble: bubbleEl };
    }

    async function consumeAgentSseResponse(response, streamingBubble, onProgressOrToken = null) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        let accumulatedText = "";
        let finalDoneMeta = null;
        let activeStreamTraceId = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                break;
            }
            buffer = buffer + decoder.decode(value, { stream: true });
            const parts = buffer.split("\n\n");
            buffer = parts.pop();

            for (const part of parts) {
                const trimmedPart = part.trim();
                if (!trimmedPart.startsWith("data: ")) {
                    continue;
                }
                const jsonStr = trimmedPart.substring(6).trim();
                try {
                    const eventData = JSON.parse(jsonStr);
                    const eventType = eventData.event_type;

                    if (eventType === "node_progress") {
                        if (onProgressOrToken) {
                            onProgressOrToken(eventType, eventData);
                        }
                        if (eventData.trace_id) {
                            dagUiState.latestTraceId = String(eventData.trace_id);
                        }
                        if (eventData.trace_id && eventData.trace_id !== activeStreamTraceId) {
                            activeStreamTraceId = eventData.trace_id;
                            if (window.ABankDebugInspector) {
                                window.ABankDebugInspector.onLiveTraceStart(activeStreamTraceId);
                            }
                        }
                        if (eventData.dag_telemetry) {
                            applyDagTelemetry(eventData.dag_telemetry, true);
                        } else if (eventData.dag_node) {
                            applyDagNodeEvent(eventData.dag_node);
                        } else if (eventData.node) {
                            advanceDagFromNodeName(eventData.node, eventData.description);
                        }
                        updateStepperProgress(eventData.description, eventData.action);
                        renderPlan(eventData.plan, false);
                    } else if (eventType === "awaiting_input") {
                        finalDoneMeta = eventData;
                        if (eventData.trace_id) {
                            dagUiState.latestTraceId = String(eventData.trace_id);
                        }
                        if (eventData.dag_telemetry) {
                            applyDagTelemetry(eventData.dag_telemetry, false);
                        }
                        finishProgressForStatus("AWAITING_INPUT", "awaiting_input");
                    } else if (eventType === "token") {
                        if (onProgressOrToken) {
                            onProgressOrToken(eventType, eventData);
                        }
                        accumulatedText = accumulatedText + eventData.content;
                        if (streamingBubble) {
                            streamingBubble.innerHTML = parseMarkdown(accumulatedText);
                            scrollToBottom();
                        }
                    } else if (eventType === "done") {
                        finalDoneMeta = eventData;
                        if (eventData.trace_id) {
                            dagUiState.latestTraceId = String(eventData.trace_id);
                        }
                        if (eventData.dag_telemetry) {
                            applyDagTelemetry(eventData.dag_telemetry, false);
                        }
                        const isPassDone = eventData.audit_status === "PASS";
                        renderPlan(eventData.plan, isPassDone);
                        finishProgressForStatus(
                            eventData.audit_status,
                            eventData.status,
                            eventData.full_answer || accumulatedText
                        );
                    } else if (eventType === "error") {
                        finalDoneMeta = eventData;
                        if (eventData.status_code === 422) {
                            finishProgressForStatus("AWAITING_INPUT", "awaiting_input");
                        } else {
                            finishProgressForStatus("ERROR", "error");
                        }
                    }
                } catch (e) {
                    console.warn("Parse SSE event error:", e);
                }
            }
        }

        return { finalDoneMeta, accumulatedText, activeStreamTraceId };
    }

    function renderStreamTerminalResult(finalDoneMeta, accumulatedText, activeStreamTraceId) {
        if (finalDoneMeta.event_type === "error") {
            let errMsg = "Đã xảy ra lỗi khi xử lý yêu cầu.";
            if (finalDoneMeta.message) {
                errMsg = finalDoneMeta.message;
            }
            finishProgressForStatus("ERROR", "error");
            appendMessage("bot", "**Rất tiếc:** " + errMsg, {
                trace_id: finalDoneMeta.trace_id,
                audit_status: "ERROR"
            });
            if (window.ABankDebugInspector) {
                window.ABankDebugInspector.onTraceCompleted(finalDoneMeta.trace_id || activeStreamTraceId);
            }
            fetchSessions();
            return;
        }
        if (finalDoneMeta.session_title) {
            applyResolvedSessionTitle(currentSessionId, finalDoneMeta.session_title);
        }
        if (finalDoneMeta.status === "awaiting_input") {
            if (window.ABankDebugInspector) {
                window.ABankDebugInspector.onTraceCompleted(finalDoneMeta.trace_id || activeStreamTraceId);
            }
            renderInteraction(finalDoneMeta);
            fetchSessions();
            return;
        }
        let finalText = accumulatedText;
        if (finalDoneMeta.full_answer) {
            finalText = finalDoneMeta.full_answer;
        }
        appendMessage("bot", finalText, {
            citations: finalDoneMeta.citations,
            sources: finalDoneMeta.sources,
            images: finalDoneMeta.images,
            trace_id: finalDoneMeta.trace_id,
            turn_id: finalDoneMeta.turn_id,
            audit_status: finalDoneMeta.audit_status,
            latency_ms: finalDoneMeta.latency_ms
        });
        if (window.ABankDebugInspector) {
            window.ABankDebugInspector.onTraceCompleted(finalDoneMeta.trace_id || activeStreamTraceId);
        }
        fetchSessions();
    }

    async function submitInteractionReply(card, payload, displayLabel, isOtherOnly, bubble, feedback) {
        if (busy || !pendingCard || pendingCard.id !== card.id) {
            return;
        }
        setBusy(true);
        const controls = bubble.querySelectorAll("button, input");
        controls.forEach(control => { control.disabled = true; });
        feedback.textContent = "Đang gửi lựa chọn…";
        if (nodeStepper) {
            nodeStepper.style.display = "block";
        }
        resetDagState(true, "workers_tools");
        updateStepperProgress("Đang gửi lựa chọn và tiếp tục xử lý...", {
            kind: "interaction",
            name: "resume",
            status: "RUNNING"
        });

        let userBubbleCommitted = false;
        let streamBubbleRefs = null;
        let streamStarted = false;

        const commitUserSelectionToChat = () => {
            if (userBubbleCommitted) {
                return;
            }
            userBubbleCommitted = true;
            pendingCard = null;
            bubble.querySelectorAll(".interaction-subtitle, .interaction-scope, .interaction-form").forEach(group => group.remove());
            feedback.remove();
            if (agentActions) {
                for (const li of Array.from(agentActions.children)) {
                    if (li.textContent === "Đang gửi lựa chọn và tiếp tục xử lý...") {
                        li.textContent = "Đã nhận phản hồi lựa chọn, đang xử lý...";
                    }
                }
            }
            if (isOtherOnly) {
                appendMessage("user", "Đã gửi câu trả lời khác.");
            } else {
                appendMessage("user", displayLabel);
            }
            if (streamBubbleRefs && !streamBubbleRefs.botMsgDiv.parentNode) {
                chatMessages.appendChild(streamBubbleRefs.botMsgDiv);
            }
            scrollToBottom();
        };

        try {
            const streamResp = await postAgent("/api/agent/resume/stream", payload);
            if (streamResp.ok && streamResp.body) {
                streamStarted = true;
                streamBubbleRefs = createStreamingBubbleElement("Đang tổng hợp phản hồi...");

                const { finalDoneMeta, accumulatedText, activeStreamTraceId } = await consumeAgentSseResponse(
                    streamResp,
                    streamBubbleRefs.streamingBubble,
                    (evtType) => {
                        if (!userBubbleCommitted && (evtType === "node_progress" || evtType === "token")) {
                            commitUserSelectionToChat();
                        }
                    }
                );

                if (streamBubbleRefs && streamBubbleRefs.botMsgDiv.parentNode) {
                    streamBubbleRefs.botMsgDiv.remove();
                }

                if (finalDoneMeta && finalDoneMeta.event_type === "error" && finalDoneMeta.status_code === 422 && !userBubbleCommitted) {
                    feedback.textContent = finalDoneMeta.message || "Vui lòng kiểm tra lại lựa chọn.";
                    finishProgressForStatus("AWAITING_INPUT", "awaiting_input");
                    return;
                }

                if (finalDoneMeta) {
                    if (!userBubbleCommitted && finalDoneMeta.event_type !== "error") {
                        commitUserSelectionToChat();
                        if (streamBubbleRefs && streamBubbleRefs.botMsgDiv.parentNode) {
                            streamBubbleRefs.botMsgDiv.remove();
                        }
                    }
                    renderStreamTerminalResult(finalDoneMeta, accumulatedText, activeStreamTraceId);
                    return;
                }
                if (window.ABankDebugInspector && activeStreamTraceId) {
                    window.ABankDebugInspector.onTraceCompleted(activeStreamTraceId);
                }
                if (!await recoverInteraction(card.id)) {
                    finishProgress("Kết nối bị gián đoạn");
                    if (userBubbleCommitted) {
                        appendMessage("bot", "Kết nối bị gián đoạn trước khi nhận được phản hồi.");
                    } else {
                        feedback.textContent = "Chưa nhận được kết quả. Bạn có thể thử lại.";
                    }
                }
                fetchSessions();
                return;
            }

            const response = await postAgent("/api/agent/resume", payload);
            if (response.status === 422) {
                const data = await response.json();
                feedback.textContent = "Vui lòng kiểm tra lại lựa chọn.";
                if (typeof data.detail === "string") {
                    feedback.textContent = data.detail;
                }
                finishProgressForStatus("AWAITING_INPUT", "awaiting_input");
                return;
            }
            if (!response.ok) {
                throw new Error("Không nhận được kết quả tiếp tục.");
            }
            const data = await response.json();
            pendingCard = null;
            bubble.querySelectorAll(".interaction-subtitle, .interaction-scope, .interaction-form").forEach(group => group.remove());
            feedback.remove();
            if (agentActions) {
                for (const li of Array.from(agentActions.children)) {
                    if (li.textContent === "Đang gửi lựa chọn và tiếp tục xử lý...") {
                        li.textContent = "Đã nhận phản hồi lựa chọn và hoàn tất xử lý";
                    }
                }
            }
            if (isOtherOnly) {
                appendMessage("user", "Đã gửi câu trả lời khác.");
            } else {
                appendMessage("user", displayLabel);
            }
            renderAgentResult(data);
        } catch {
            if (streamBubbleRefs && streamBubbleRefs.botMsgDiv.parentNode) {
                streamBubbleRefs.botMsgDiv.remove();
            }
            if (!await recoverInteraction(card.id)) {
                if (streamStarted && userBubbleCommitted) {
                    finishProgress("Kết nối bị gián đoạn");
                    appendMessage("bot", "Kết nối bị gián đoạn trước khi nhận được phản hồi.");
                } else {
                    feedback.textContent = "Chưa nhận được kết quả. Bạn có thể thử lại.";
                }
            }
        } finally {
            if (pendingCard && pendingCard.id === card.id) {
                controls.forEach(control => { control.disabled = false; });
            }
            setBusy(false);
        }
    }

    async function sendChoice(card, choiceId, label, bubble, feedback) {
        const payload = { interaction_id: card.id, choice_id: choiceId };
        if (choiceId === "other") {
            payload.text = label;
        }
        return submitInteractionReply(card, payload, label, choiceId === "other", bubble, feedback);
    }

    function showComposerStatus(message, mode = "info", autoHideMs = 0) {
        if (!composerStatusBanner) {
            return;
        }
        if (statusBannerTimer) {
            clearTimeout(statusBannerTimer);
            statusBannerTimer = null;
        }
        composerStatusBanner.className = "composer-status-banner";
        if (mode === "recording") {
            composerStatusBanner.classList.add("is-recording");
        } else if (mode === "error") {
            composerStatusBanner.classList.add("is-error");
        }
        composerStatusBanner.textContent = message;
        composerStatusBanner.hidden = false;
        if (autoHideMs > 0) {
            statusBannerTimer = setTimeout(() => {
                clearComposerStatus();
            }, autoHideMs);
        }
    }

    function clearComposerStatus() {
        if (!composerStatusBanner) {
            return;
        }
        if (statusBannerTimer) {
            clearTimeout(statusBannerTimer);
            statusBannerTimer = null;
        }
        composerStatusBanner.textContent = "";
        composerStatusBanner.className = "composer-status-banner";
        composerStatusBanner.hidden = true;
    }

    function formatFileSize(bytes) {
        if (typeof bytes !== "number" || bytes <= 0) {
            return "";
        }
        if (bytes < 1024) {
            return bytes + " B";
        }
        if (bytes < 1024 * 1024) {
            return Math.round(bytes / 1024) + " KB";
        }
        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }

    function isAllowedImageFile(file) {
        if (!file) {
            return false;
        }
        const mime = String(file.type || "").toLowerCase();
        if (ALLOWED_IMAGE_TYPES.includes(mime)) {
            return true;
        }
        if ((!mime || mime === "application/octet-stream") && file.name) {
            const lowerName = String(file.name).toLowerCase();
            for (const ext of ALLOWED_IMAGE_EXTS) {
                if (lowerName.endsWith(ext)) {
                    return true;
                }
            }
        }
        return false;
    }

    function inferImageMimeSubFromName(fileName) {
        const lower = String(fileName || "").toLowerCase();
        if (lower.endsWith(".png")) {
            return "png";
        }
        if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
            return "jpeg";
        }
        if (lower.endsWith(".webp")) {
            return "webp";
        }
        if (lower.endsWith(".gif")) {
            return "gif";
        }
        return "";
    }

    function normalizeImageDataUrl(dataUrl, fileName) {
        const raw = String(dataUrl || "");
        if (raw.startsWith("data:image/pjpeg;base64,")) {
            return "data:image/jpeg;base64," + raw.slice("data:image/pjpeg;base64,".length);
        }
        if (raw.startsWith("data:image/x-png;base64,")) {
            return "data:image/png;base64," + raw.slice("data:image/x-png;base64,".length);
        }
        const octetPrefix = "data:application/octet-stream;base64,";
        if (raw.startsWith(octetPrefix)) {
            const sub = inferImageMimeSubFromName(fileName);
            if (sub) {
                return "data:image/" + sub + ";base64," + raw.slice(octetPrefix.length);
            }
        }
        const emptyMimePrefix = "data:;base64,";
        if (raw.startsWith(emptyMimePrefix)) {
            const sub = inferImageMimeSubFromName(fileName);
            if (sub) {
                return "data:image/" + sub + ";base64," + raw.slice(emptyMimePrefix.length);
            }
        }
        return raw;
    }

    function readFileAsDataUrl(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                resolve(String(reader.result || ""));
            };
            reader.onerror = () => {
                reject(new Error("Không thể đọc tệp hình ảnh."));
            };
            reader.readAsDataURL(file);
        });
    }

    function renderAttachmentPreviewBar() {
        if (!attachmentPreviewBar) {
            return;
        }
        attachmentPreviewBar.replaceChildren();
        if (pendingAttachments.length === 0) {
            attachmentPreviewBar.hidden = true;
            return;
        }
        attachmentPreviewBar.hidden = false;
        const previewLightboxItems = [];
        for (let idx = 0; idx < pendingAttachments.length; idx++) {
            const att = pendingAttachments[idx];
            previewLightboxItems.push({
                url: att.data_url,
                caption: att.name,
                region_type: "attachment"
            });
        }
        for (let idx = 0; idx < pendingAttachments.length; idx++) {
            const att = pendingAttachments[idx];
            const card = document.createElement("div");
            card.className = "attachment-thumb-card";

            const thumbBtn = document.createElement("button");
            thumbBtn.type = "button";
            thumbBtn.className = "attachment-thumb-btn";
            thumbBtn.title = "Xem trước " + att.name;
            const img = document.createElement("img");
            img.src = att.data_url;
            img.alt = att.name;
            thumbBtn.appendChild(img);
            thumbBtn.addEventListener("click", () => {
                openImageLightbox(previewLightboxItems, idx);
            });

            const meta = document.createElement("div");
            meta.className = "attachment-thumb-meta";
            const nameEl = document.createElement("span");
            nameEl.className = "attachment-thumb-name";
            nameEl.textContent = att.name;
            const sizeEl = document.createElement("span");
            sizeEl.className = "attachment-thumb-size";
            sizeEl.textContent = formatFileSize(att.size);
            meta.append(nameEl, sizeEl);

            const removeBtn = document.createElement("button");
            removeBtn.type = "button";
            removeBtn.className = "attachment-remove-btn";
            removeBtn.textContent = "×";
            removeBtn.setAttribute("aria-label", "Xóa ảnh " + att.name);
            removeBtn.addEventListener("click", () => {
                pendingAttachments.splice(idx, 1);
                renderAttachmentPreviewBar();
            });

            card.append(thumbBtn, meta, removeBtn);
            attachmentPreviewBar.appendChild(card);
        }
    }

    function addImageFiles(fileList) {
        const files = Array.from(fileList || []);
        if (files.length === 0) {
            return Promise.resolve();
        }
        const runTask = async () => {
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                if (pendingAttachments.length >= MAX_ATTACHMENTS) {
                    showComposerStatus("Chỉ đính kèm tối đa " + MAX_ATTACHMENTS + " ảnh cho mỗi lượt gửi.", "error", 4000);
                    break;
                }
                if (!isAllowedImageFile(file)) {
                    showComposerStatus("Định dạng không hỗ trợ (" + (file.name || "tệp") + "). Vui lòng chọn PNG, JPEG, WEBP hoặc GIF.", "error", 4500);
                    continue;
                }
                if (file.size > MAX_IMAGE_BYTES) {
                    showComposerStatus("Ảnh " + (file.name || "") + " vượt quá giới hạn 5 MB.", "error", 4500);
                    continue;
                }
                try {
                    const rawDataUrl = await readFileAsDataUrl(file);
                    const dataUrl = normalizeImageDataUrl(rawDataUrl, file.name);
                    if (!dataUrl.startsWith("data:image/")) {
                        showComposerStatus("Không đọc được dữ liệu hình ảnh hợp lệ.", "error", 4000);
                        continue;
                    }
                    let displayName = file.name;
                    if (!displayName) {
                        displayName = "anh-dinh-kem-" + (pendingAttachments.length + 1) + ".png";
                    }
                    pendingAttachments.push({
                        name: displayName,
                        size: file.size,
                        data_url: dataUrl
                    });
                } catch {
                    showComposerStatus("Không thể tải ảnh " + (file.name || "") + ".", "error", 4000);
                }
            }
            renderAttachmentPreviewBar();
        };
        activeImageReadPromise = activeImageReadPromise.then(runTask, runTask);
        return activeImageReadPromise;
    }

    function setVoiceRecordingUi(active) {
        isRecording = active;
        if (!voiceInputBtn) {
            return;
        }
        voiceInputBtn.classList.toggle("is-recording", active);
        if (active) {
            voiceInputBtn.setAttribute("aria-pressed", "true");
            voiceInputBtn.title = "Nhấn để dừng ghi âm";
        } else {
            voiceInputBtn.setAttribute("aria-pressed", "false");
            voiceInputBtn.title = "Nhập bằng giọng nói tiếng Việt";
        }
    }

    function stopVoiceRecording() {
        if (speechRecognition && isRecording) {
            try {
                speechRecognition.stop();
            } catch {
                setVoiceRecordingUi(false);
            }
        }
    }

    function initVoiceInput() {
        if (!voiceInputBtn) {
            return;
        }
        const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognitionCtor) {
            voiceInputBtn.addEventListener("click", () => {
                showComposerStatus("Trình duyệt hiện tại chưa hỗ trợ nhận diện giọng nói. Vui lòng dùng Chrome hoặc Edge.", "error", 5000);
            });
            return;
        }

        voiceInputBtn.addEventListener("click", () => {
            if (busy || pendingCard) {
                return;
            }
            if (isRecording && speechRecognition) {
                stopVoiceRecording();
                return;
            }

            const recognition = new SpeechRecognitionCtor();
            recognition.lang = "vi-VN";
            recognition.interimResults = true;
            recognition.continuous = false;
            recognition.maxAlternatives = 1;
            speechRecognition = recognition;
            voiceBaseText = userInput.value.trim();
            voiceCapturedText = "";

            recognition.onstart = () => {
                setVoiceRecordingUi(true);
                showComposerStatus("Đang nghe giọng nói tiếng Việt… Hãy nói câu hỏi của Quý khách.", "recording");
            };

            recognition.onresult = (event) => {
                let finalPart = "";
                let interimPart = "";
                for (let i = 0; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        finalPart = finalPart + transcript + " ";
                    } else {
                        interimPart = interimPart + transcript;
                    }
                }
                const combinedVoice = (finalPart + interimPart).trim();
                voiceCapturedText = combinedVoice;
                if (voiceBaseText && combinedVoice) {
                    userInput.value = voiceBaseText + " " + combinedVoice;
                } else if (combinedVoice) {
                    userInput.value = combinedVoice;
                } else {
                    userInput.value = voiceBaseText;
                }
            };

            recognition.onerror = (event) => {
                setVoiceRecordingUi(false);
                const errCode = String((event && event.error) || "");
                let errText = "Không thể nhận diện giọng nói lúc này.";
                if (errCode === "not-allowed" || errCode === "service-not-allowed") {
                    errText = "Chưa cấp quyền micro. Vui lòng cho phép truy cập micro trong trình duyệt.";
                } else if (errCode === "no-speech") {
                    errText = "Chưa phát hiện giọng nói. Quý khách vui lòng nhấn micro và thử lại.";
                } else if (errCode === "audio-capture") {
                    errText = "Không tìm thấy thiết bị micro khả dụng.";
                } else if (errCode === "network") {
                    errText = "Lỗi kết nối mạng khi nhận diện giọng nói.";
                }
                showComposerStatus(errText, "error", 5000);
            };

            recognition.onend = () => {
                const hadSpeech = Boolean(voiceCapturedText);
                setVoiceRecordingUi(false);
                if (hadSpeech) {
                    showComposerStatus("Đã ghi nhận nội dung giọng nói vào ô nhập.", "info", 3000);
                    userInput.focus();
                } else if (composerStatusBanner && composerStatusBanner.classList.contains("is-recording")) {
                    clearComposerStatus();
                }
            };

            try {
                recognition.start();
            } catch {
                setVoiceRecordingUi(false);
                showComposerStatus("Không thể khởi động micro. Vui lòng thử lại.", "error", 4000);
            }
        });
    }

    async function submitQueryFallback(cleanQuery, attachmentsPayload) {
        try {
            const reqBody = {
                message: cleanQuery,
                session_id: currentSessionId
            };
            if (Array.isArray(attachmentsPayload) && attachmentsPayload.length > 0) {
                reqBody.attachments = attachmentsPayload;
            }
            const res = await fetch("/api/agent/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(reqBody)
            });
            if (!res.ok) {
                const errData = await res.json().catch(() => ({ detail: "Lỗi kết nối máy chủ dịch vụ" }));
                let detailMsg = "Không thể xử lý yêu cầu lúc này.";
                if (errData && errData.detail) {
                    detailMsg = errData.detail;
                }
                finishProgressForStatus("ERROR", "error");
                appendMessage("bot", "**Rất tiếc:** " + detailMsg);
            } else {
                const data = await res.json();
                renderAgentResult(data);
            }
        } catch (err) {
            finishProgressForStatus("ERROR", "error");
            appendMessage("bot", "**Lỗi kết nối:** Không thể liên lạc với máy chủ (" + err.message + ").");
        }
    }

    async function submitQuery(queryText) {
        if (busy || pendingCard) {
            return;
        }
        await activeImageReadPromise;
        if (busy || pendingCard) {
            return;
        }
        let cleanQuery = "";
        if (queryText) {
            cleanQuery = queryText.trim();
        }
        const attachmentsToSend = pendingAttachments.slice();
        if (cleanQuery === "" && attachmentsToSend.length === 0) {
            return;
        }

        if (isRecording) {
            stopVoiceRecording();
        }
        clearComposerStatus();

        const attachmentsPayload = [];
        for (const att of attachmentsToSend) {
            attachmentsPayload.push({ name: att.name, data_url: att.data_url });
        }

        let userBubbleText = cleanQuery;
        if (!userBubbleText && attachmentsToSend.length > 0) {
            userBubbleText = "Đã gửi " + attachmentsToSend.length + " hình ảnh đính kèm để phân tích.";
        }

        exitWelcomeState();
        appendMessage("user", userBubbleText, { userAttachments: attachmentsToSend });
        upsertOptimisticSessionInSidebar(userBubbleText, "running");
        userInput.value = "";
        pendingAttachments = [];
        renderAttachmentPreviewBar();
        setBusy(true);

        resetStepper();
        resetDagState(true, "input_guardrail");
        if (nodeStepper) {
            nodeStepper.style.display = "block";
        }

        // Tạo bubble tạm thời để stream tokens vào
        const { botMsgDiv, streamingBubble } = createStreamingBubbleElement("Đang kết nối agent...");
        chatMessages.appendChild(botMsgDiv);
        scrollToBottom();

        let streamStarted = false;

        try {
            const streamBody = {
                message: cleanQuery,
                session_id: currentSessionId
            };
            if (attachmentsPayload.length > 0) {
                streamBody.attachments = attachmentsPayload;
            }
            const response = await fetch("/api/agent/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(streamBody)
            });

            if (!response.ok || !response.body) {
                botMsgDiv.remove();
                if (!await recoverInteraction() && response.status < 500) {
                    await submitQueryFallback(cleanQuery, attachmentsPayload);
                } else if (!pendingCard) {
                    finishProgressForStatus("ERROR", "error");
                    appendMessage("bot", "Không nhận được phản hồi từ hệ thống.");
                }
                return;
            }

            streamStarted = true;
            const { finalDoneMeta, accumulatedText, activeStreamTraceId } = await consumeAgentSseResponse(
                response,
                streamingBubble
            );

            if (finalDoneMeta) {
                botMsgDiv.remove();
                renderStreamTerminalResult(finalDoneMeta, accumulatedText, activeStreamTraceId);
            } else {
                if (window.ABankDebugInspector && activeStreamTraceId) {
                    window.ABankDebugInspector.onTraceCompleted(activeStreamTraceId);
                }
                botMsgDiv.remove();
                if (!await recoverInteraction()) {
                    finishProgress("Kết nối bị gián đoạn");
                    appendMessage("bot", "Kết nối bị gián đoạn trước khi nhận được phản hồi.");
                }
                fetchSessions();
            }

        } catch (err) {
            botMsgDiv.remove();
            if (await recoverInteraction()) {
                return;
            }
            if (streamStarted) {
                finishProgress("Kết nối bị gián đoạn");
                appendMessage("bot", "Kết nối bị gián đoạn trước khi nhận được phản hồi.");
            } else {
                await submitQueryFallback(cleanQuery, attachmentsPayload);
            }
        } finally {
            setBusy(false);
            userInput.focus();
            scrollToBottom();
        }
    }

    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        submitQuery(userInput.value);
    });

    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && e.isComposing) {
            e.preventDefault();
        }
    });

    userInput.addEventListener("paste", (e) => {
        if (busy || pendingCard || !e.clipboardData || !e.clipboardData.items) {
            return;
        }
        const pastedImages = [];
        for (const item of e.clipboardData.items) {
            if (item.kind === "file" && item.type && item.type.startsWith("image/")) {
                const file = item.getAsFile();
                if (file) {
                    pastedImages.push(file);
                }
            }
        }
        if (pastedImages.length > 0) {
            e.preventDefault();
            addImageFiles(pastedImages);
        }
    });

    if (attachImageBtn && imageUploadInput) {
        attachImageBtn.addEventListener("click", () => {
            if (busy || pendingCard) {
                return;
            }
            imageUploadInput.click();
        });
        imageUploadInput.addEventListener("change", () => {
            const selectedFiles = Array.from(imageUploadInput.files || []);
            imageUploadInput.value = "";
            if (selectedFiles.length > 0) {
                addImageFiles(selectedFiles);
            }
        });
    }

    if (chatComposer) {
        chatComposer.addEventListener("dragover", (e) => {
            e.preventDefault();
            if (!busy && !pendingCard) {
                chatComposer.classList.add("is-dragover");
            }
        });
        chatComposer.addEventListener("dragleave", () => {
            chatComposer.classList.remove("is-dragover");
        });
        chatComposer.addEventListener("drop", (e) => {
            e.preventDefault();
            chatComposer.classList.remove("is-dragover");
            if (busy || pendingCard) {
                return;
            }
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                const droppedFiles = Array.from(e.dataTransfer.files);
                addImageFiles(droppedFiles);
            }
        });
    }

    if (newSessionBtn) {
        newSessionBtn.addEventListener("click", () => {
            createNewSession(true);
        });
    }

    if (sessionSearchInput) {
        sessionSearchInput.addEventListener("input", () => {
            sessionUiState.searchQuery = sessionSearchInput.value;
            if (sessionUiState.searchTimer !== null) {
                window.clearTimeout(sessionUiState.searchTimer);
            }
            sessionUiState.searchTimer = window.setTimeout(() => {
                fetchSessions();
            }, 180);
        });
    }

    initSidebarToggle();
    resetDagState(false);
    initVoiceInput();

    setBusy(true);
    fetchSessions();
    (async () => {
        if (storedSessionBeforeLoad) {
            const restored = await switchToSession(currentSessionId);
            if (!restored) {
                await recoverInteraction(null, true);
            }
        } else {
            await recoverInteraction(null, true);
        }
        await fetchSessions();
    })().finally(() => {
        setBusy(false);
        userInput.focus();
    });
});
