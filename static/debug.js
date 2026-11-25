/**
 * A-Bank Debug & Tracing Inspector UI
 * Tuân thủ 100%: 0% Regex Law & Zero Ternary Law.
 */

(function () {
    "use strict";

    const RUN_TYPE_COLORS = {
        chain: "chain",
        llm: "llm",
        tool: "tool",
        retriever: "retriever",
        parser: "parser",
        embedding: "embedding"
    };

    const state = {
        initialized: false,
        sessionId: "",
        scope: "session",
        searchQuery: "",
        spanFilterType: "all",
        spanSearchQuery: "",
        jsonFilterQuery: "",
        traces: [],
        activeCount: 0,
        totalStored: 0,
        systemConfig: null,
        selectedTraceId: null,
        selectedDetail: null,
        selectedSpanId: null,
        liveTraceId: null,
        pollTimer: null,
        replayTimer: null,
        replayPlaying: false,
        replayStepIndex: -1,
        replaySpeedMs: 1400,
        replayTraceId: null,
        activeTabId: "debugTabWaterfall",
        loadingList: false,
        loadingDetail: false,
        pendingReload: false,
        pendingAutoSelect: false
    };

    const dom = {};

    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) {
            node.className = className;
        }
        if (text !== undefined && text !== null) {
            node.textContent = String(text);
        }
        return node;
    }

    function svgEl(tag, attrs, text) {
        const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
        if (attrs && typeof attrs === "object") {
            const keys = Object.keys(attrs);
            for (let i = 0; i < keys.length; i++) {
                const k = keys[i];
                if (attrs[k] !== undefined && attrs[k] !== null) {
                    node.setAttribute(k, String(attrs[k]));
                }
            }
        }
        if (text !== undefined && text !== null) {
            node.textContent = String(text);
        }
        return node;
    }

    function formatMs(val) {
        if (typeof val !== "number" || Number.isNaN(val)) {
            return "—";
        }
        if (val >= 1000) {
            return (val / 1000).toFixed(2) + " s (" + val.toFixed(1) + " ms)";
        }
        return val.toFixed(2) + " ms";
    }

    function formatShortMs(val) {
        if (typeof val !== "number" || Number.isNaN(val)) {
            return "—";
        }
        if (val >= 1000) {
            return (val / 1000).toFixed(2) + "s";
        }
        return val.toFixed(1) + "ms";
    }

    function formatTimeLabel(isoStr) {
        if (!isoStr) {
            return "—";
        }
        const dt = new Date(isoStr);
        if (Number.isNaN(dt.getTime())) {
            return String(isoStr);
        }
        const hh = String(dt.getHours()).padStart(2, "0");
        const mm = String(dt.getMinutes()).padStart(2, "0");
        const ss = String(dt.getSeconds()).padStart(2, "0");
        return hh + ":" + mm + ":" + ss;
    }

    function shortId(idStr) {
        if (!idStr) {
            return "—";
        }
        const s = String(idStr);
        if (s.length <= 12) {
            return s;
        }
        return s.slice(0, 8) + "…" + s.slice(-4);
    }

    function runTypeClass(runType) {
        const clean = String(runType || "chain").toLowerCase();
        if (Object.prototype.hasOwnProperty.call(RUN_TYPE_COLORS, clean)) {
            return "type-" + RUN_TYPE_COLORS[clean];
        }
        return "type-chain";
    }

    function statusBadgeClass(statusVal) {
        const lower = String(statusVal || "").toLowerCase();
        if (lower === "error" || lower === "fail" || lower === "blocked" || lower === "failed" || lower === "rejected") {
            return "status-error";
        }
        if (lower === "running") {
            return "status-running";
        }
        if (lower === "suspended" || lower === "awaiting_input" || lower === "needs_input" || lower === "retry") {
            return "status-warn";
        }
        if (lower === "ok" || lower === "pass" || lower === "completed" || lower === "success" || lower === "passed") {
            return "status-ok";
        }
        return "status-neutral";
    }

    function fallbackCopyViaTextarea(textStr, btn, defaultLabel, onSuccess) {
        const ta = document.createElement("textarea");
        ta.value = textStr;
        ta.style.position = "fixed";
        ta.style.left = "-9999px";
        ta.style.top = "0";
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        let copied = false;
        try {
            copied = document.execCommand("copy");
        } catch (_err) {
            copied = false;
        }
        ta.remove();
        if (copied) {
            if (typeof onSuccess === "function") {
                onSuccess();
            } else if (btn) {
                btn.textContent = "Đã sao chép ✓";
                window.setTimeout(() => {
                    btn.textContent = defaultLabel;
                }, 1500);
            }
        }
    }

    function copyText(text, btn, defaultLabel, onSuccess) {
        if (!text) {
            return;
        }
        const textStr = String(text);
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(textStr).then(() => {
                if (typeof onSuccess === "function") {
                    onSuccess();
                } else if (btn) {
                    btn.textContent = "Đã sao chép ✓";
                    window.setTimeout(() => {
                        btn.textContent = defaultLabel;
                    }, 1500);
                }
            }).catch(() => {
                fallbackCopyViaTextarea(textStr, btn, defaultLabel, onSuccess);
            });
        } else {
            fallbackCopyViaTextarea(textStr, btn, defaultLabel, onSuccess);
        }
    }

    function buildCopyIconBtn(getTextFn, titleText) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "dag-io-copy-btn";
        btn.title = titleText;
        btn.setAttribute("aria-label", titleText);

        function drawClipboardIcon() {
            btn.replaceChildren();
            btn.classList.remove("is-copied");
            const s = svgEl("svg", {
                viewBox: "0 0 24 24",
                width: "13",
                height: "13",
                fill: "none",
                stroke: "currentColor",
                "stroke-width": "2",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
                "aria-hidden": "true"
            });
            s.appendChild(svgEl("rect", { x: "9", y: "9", width: "13", height: "13", rx: "2", ry: "2" }));
            s.appendChild(svgEl("path", { d: "M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" }));
            btn.appendChild(s);
        }

        function drawCheckIcon() {
            btn.replaceChildren();
            btn.classList.add("is-copied");
            const s = svgEl("svg", {
                viewBox: "0 0 24 24",
                width: "13",
                height: "13",
                fill: "none",
                stroke: "currentColor",
                "stroke-width": "2.4",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
                "aria-hidden": "true"
            });
            s.appendChild(svgEl("polyline", { points: "20 6 9 17 4 12" }));
            btn.appendChild(s);
        }

        drawClipboardIcon();
        btn.addEventListener("click", () => {
            const val = getTextFn();
            copyText(val, null, "", () => {
                drawCheckIcon();
                window.setTimeout(() => {
                    drawClipboardIcon();
                }, 1400);
            });
        });
        return btn;
    }

    function syncHeaderIndicators() {
        if (dom.traceCountBadge) {
            dom.traceCountBadge.textContent = String(state.totalStored);
        }
        if (dom.liveDot) {
            const isLive = Boolean(state.liveTraceId) || state.activeCount > 0;
            dom.liveDot.hidden = !isLive;
        }
    }

    function syncScopeButtons() {
        if (dom.scopeSessionBtn) {
            const isSession = state.scope === "session";
            dom.scopeSessionBtn.classList.toggle("active", isSession);
            dom.scopeSessionBtn.setAttribute("aria-pressed", String(isSession));
        }
        if (dom.scopeAllBtn) {
            const isAll = state.scope === "all";
            dom.scopeAllBtn.classList.toggle("active", isAll);
            dom.scopeAllBtn.setAttribute("aria-pressed", String(isAll));
        }
    }

    function syncTopSystemBadges() {
        const cfg = state.systemConfig;
        if (!cfg || !dom.langsmithBadge) {
            return;
        }
        if (cfg.langsmith_active) {
            dom.langsmithBadge.textContent = "LangSmith: ON";
            dom.langsmithBadge.className = "debug-pill debug-pill-ok";
        } else if (cfg.tracing_enabled) {
            dom.langsmithBadge.textContent = "LangSmith: LOCAL";
            dom.langsmithBadge.className = "debug-pill debug-pill-info";
        } else {
            dom.langsmithBadge.textContent = "Tracing: OFF";
            dom.langsmithBadge.className = "debug-pill debug-pill-muted";
        }

        if (dom.projectBadge) {
            if (cfg.langsmith_project) {
                dom.projectBadge.textContent = cfg.langsmith_project;
                dom.projectBadge.hidden = false;
            } else {
                dom.projectBadge.hidden = true;
            }
        }
    }

    function syncActionButtons(detail) {
        const hasDetail = Boolean(detail);
        if (dom.copyJsonBtn) {
            dom.copyJsonBtn.disabled = !hasDetail;
        }
        if (dom.exportJsonBtn) {
            dom.exportJsonBtn.disabled = !hasDetail;
        }
        if (dom.langsmithLink) {
            if (detail && detail.langsmith_run_url) {
                dom.langsmithLink.href = detail.langsmith_run_url;
                dom.langsmithLink.hidden = false;
            } else {
                dom.langsmithLink.removeAttribute("href");
                dom.langsmithLink.hidden = true;
            }
        }
    }

    function buildKeyValueTable(rows) {
        const wrap = el("div", "debug-kv-table");
        for (const pair of rows) {
            const row = el("div", "debug-kv-row");
            const keyEl = el("div", "debug-kv-key", pair[0]);
            const valEl = el("div", "debug-kv-val");
            if (pair[1] instanceof HTMLElement) {
                valEl.appendChild(pair[1]);
            } else if (typeof pair[1] === "object" && pair[1] !== null) {
                const pre = el("pre", "debug-mini-pre", JSON.stringify(pair[1], null, 2));
                valEl.appendChild(pre);
            } else if (pair[1] === undefined || pair[1] === null || pair[1] === "") {
                valEl.textContent = "—";
                valEl.classList.add("is-muted");
            } else {
                valEl.textContent = String(pair[1]);
            }
            row.append(keyEl, valEl);
            wrap.appendChild(row);
        }
        return wrap;
    }

    function buildAccordionItem(summaryText, contentNode, openByDefault) {
        const det = document.createElement("details");
        det.className = "debug-accordion-item";
        if (openByDefault) {
            det.open = true;
        }
        const sum = document.createElement("summary");
        sum.className = "debug-accordion-summary";
        sum.textContent = summaryText;
        const body = el("div", "debug-accordion-body");
        if (contentNode instanceof HTMLElement) {
            body.appendChild(contentNode);
        } else {
            body.textContent = String(contentNode || "");
        }
        det.append(sum, body);
        return det;
    }

    function syncExpandAllButton(container, toggleBtn) {
        if (!container || !toggleBtn) {
            return;
        }
        const items = container.querySelectorAll("details.debug-accordion-item");
        if (items.length === 0) {
            toggleBtn.disabled = true;
            toggleBtn.setAttribute("aria-expanded", "false");
            return;
        }
        toggleBtn.disabled = false;
        let allOpen = true;
        for (const item of items) {
            if (!item.open) {
                allOpen = false;
                break;
            }
        }
        toggleBtn.setAttribute("aria-expanded", String(allOpen));
    }

    function activateTab(tabId, moveFocus) {
        if (!dom.tabs || !dom.panels) {
            return;
        }
        let activeTab = null;
        for (const tab of dom.tabs) {
            const isMatch = tab.id === tabId;
            tab.setAttribute("aria-selected", String(isMatch));
            if (isMatch) {
                tab.setAttribute("focusgroupstart", "");
                activeTab = tab;
            } else {
                tab.removeAttribute("focusgroupstart");
            }
        }

        for (const panel of dom.panels) {
            const ownerId = panel.getAttribute("aria-labelledby");
            if (ownerId === tabId) {
                panel.removeAttribute("hidden");
                panel.setAttribute("tabindex", "0");
            } else {
                panel.hidden = "until-found";
                panel.removeAttribute("tabindex");
            }
        }

        state.activeTabId = tabId;
        if (activeTab) {
            activeTab.scrollIntoView({ block: "nearest", inline: "nearest" });
            if (moveFocus) {
                activeTab.focus();
            }
        }
    }

    function renderTraceList() {
        if (!dom.traceList || !dom.sidebarMeta) {
            return;
        }
        const queryLower = state.searchQuery.trim().toLowerCase();
        const filtered = [];
        for (const item of state.traces) {
            if (!queryLower) {
                filtered.push(item);
                continue;
            }
            const haystackParts = [
                item.trace_id || "",
                item.turn_id || "",
                item.query_preview || "",
                item.intent || "",
                item.audit_status || "",
                item.entrypoint || "",
                item.outcome || ""
            ];
            if (Array.isArray(item.models_used)) {
                haystackParts.push(item.models_used.join(" "));
            }
            const haystack = haystackParts.join(" ").toLowerCase();
            if (haystack.includes(queryLower)) {
                filtered.push(item);
            }
        }

        let scopeLabel = "Phiên hiện tại";
        if (state.scope === "all") {
            scopeLabel = "Toàn hệ thống";
        }
        dom.sidebarMeta.textContent = scopeLabel + " · " + filtered.length + "/" + state.traces.length + " trace (Đang chạy: " + state.activeCount + ")";

        dom.traceList.replaceChildren();
        if (filtered.length === 0) {
            const emptyNote = el("div", "debug-sidebar-empty", "Chưa có trace nào phù hợp bộ lọc.");
            dom.traceList.appendChild(emptyNote);
            return;
        }

        for (const item of filtered) {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "debug-trace-item";
            if (item.trace_id === state.selectedTraceId) {
                btn.classList.add("selected");
            }
            if (item.is_active) {
                btn.classList.add("is-active");
            }

            const topRow = el("div", "debug-trace-item-top");
            const leftBadges = el("div", "debug-trace-item-badges");
            if (item.is_active) {
                const liveBadge = el("span", "debug-status-badge status-running", "LIVE");
                leftBadges.appendChild(liveBadge);
            }
            const entryBadge = el("span", "debug-entry-badge", item.entrypoint || "agent");
            leftBadges.appendChild(entryBadge);

            let badgeText = item.audit_status || item.outcome || "completed";
            const statBadge = el("span", "debug-status-badge " + statusBadgeClass(badgeText), badgeText);
            leftBadges.appendChild(statBadge);

            const durSpan = el("span", "debug-trace-item-dur", formatShortMs(item.duration_ms));
            topRow.append(leftBadges, durSpan);

            let titleText = item.query_preview;
            if (!titleText) {
                titleText = "[" + (item.entrypoint || "trace") + "] " + shortId(item.trace_id);
            }
            const titleEl = el("div", "debug-trace-item-query", titleText);

            const metaRow = el("div", "debug-trace-item-meta");
            let tokenLabel = "0 tok";
            if (item.token_usage && typeof item.token_usage.total_tokens === "number") {
                tokenLabel = item.token_usage.total_tokens + " tok";
            }
            const summaryLine = [
                shortId(item.trace_id),
                (item.span_count || 0) + " spans",
                (item.llm_call_count || 0) + " LLM",
                tokenLabel,
                formatTimeLabel(item.timestamp)
            ].join(" · ");
            metaRow.textContent = summaryLine;

            btn.append(topRow, titleEl, metaRow);
            btn.addEventListener("click", () => {
                selectTrace(item.trace_id);
            });
            dom.traceList.appendChild(btn);
        }
    }

    function renderKpiAndIdentity(detail) {
        if (!dom.kpiStrip || !dom.identityBar) {
            return;
        }
        const summary = detail.summary || {};
        const tok = summary.token_usage || {};
        const inTok = tok.input_tokens || 0;
        const outTok = tok.output_tokens || 0;
        const totalTok = tok.total_tokens || 0;

        let statusText = detail.outcome || "completed";
        if (summary.audit_status) {
            statusText = summary.audit_status;
        }
        if (detail.is_active) {
            statusText = "RUNNING";
        }

        let modelsLabel = "Chưa gọi LLM";
        if (Array.isArray(summary.models_used) && summary.models_used.length > 0) {
            modelsLabel = summary.models_used.join(", ");
        }
        const kpis = [
            { label: "Tổng độ trễ", value: formatMs(detail.duration_ms), sub: "Entrypoint: " + (detail.entrypoint || "—") },
            { label: "Tổng Spans", value: String(summary.span_count || 0), sub: "Graph nodes: " + (summary.graph_node_count || 0) },
            { label: "LLM Calls", value: String(summary.llm_call_count || 0), sub: modelsLabel },
            { label: "Tokens (In / Out)", value: String(totalTok), sub: inTok + " prompt / " + outTok + " completion" },
            { label: "Tools & RAG", value: (summary.tool_call_count || 0) + " / " + (summary.retriever_call_count || 0), sub: "Parsers: " + (summary.parser_call_count || 0) + " · Citations: " + (summary.citation_count || 0) },
            { label: "Trạng thái / Lỗi", value: String(statusText), sub: "Span lỗi: " + (summary.error_count || 0) }
        ];

        dom.kpiStrip.replaceChildren();

        for (const kpi of kpis) {
            const card = el("div", "debug-kpi-card");
            card.append(
                el("div", "debug-kpi-label", kpi.label),
                el("div", "debug-kpi-value", kpi.value),
                el("div", "debug-kpi-sub", kpi.sub)
            );
            dom.kpiStrip.appendChild(card);
        }

        dom.identityBar.replaceChildren();
        const idItems = [
            { label: "Trace ID", val: detail.trace_id || "—", copyable: detail.trace_id },
            { label: "Turn ID", val: detail.turn_id || "—", copyable: detail.turn_id },
            { label: "Run ID", val: detail.run_id || "—", copyable: detail.run_id },
            { label: "Session Ref", val: detail.session_ref || "—" },
            { label: "Intent", val: summary.intent || "—" },
            { label: "Thời điểm", val: detail.timestamp || "—" }
        ];

        for (const item of idItems) {
            const pill = el("div", "debug-id-pill");
            pill.append(el("span", "debug-id-key", item.label + ":"));
            const valSpan = el("span", "debug-id-val", item.val);
            pill.appendChild(valSpan);
            if (item.copyable) {
                const cBtn = el("button", "debug-mini-copy", "Copy");
                cBtn.type = "button";
                cBtn.addEventListener("click", () => {
                    copyText(item.copyable, cBtn, "Copy");
                });
                pill.appendChild(cBtn);
            }
            dom.identityBar.appendChild(pill);
        }
    }

    function renderSelectedSpanInspector(container, span, detail) {
        container.replaceChildren();
        if (!span) {
            container.appendChild(el("div", "debug-inspector-empty", "Nhấn vào một span trong Waterfall ở trên để xem toàn bộ thông số, thuộc tính, inputs/outputs và thời gian thực thi."));
            return;
        }

        const hdr = el("div", "debug-inspector-header");
        const titleRow = el("div", "debug-inspector-title-row");
        titleRow.append(
            el("span", "debug-runtype-badge " + runTypeClass(span.run_type), span.run_type),
            el("strong", "debug-inspector-name", span.name),
            el("span", "debug-status-badge " + statusBadgeClass(span.status), span.status)
        );

        const rightBtns = el("div", "debug-inspector-actions");
        const expandAllBtn = el("button", "debug-mini-btn", "");
        expandAllBtn.type = "button";
        expandAllBtn.setAttribute("aria-expanded", "false");
        expandAllBtn.append(
            el("span", "debug-expand-label", "Mở rộng tất cả"),
            el("span", "debug-collapse-label", "Thu gọn tất cả")
        );
        const copySpanBtn = el("button", "debug-mini-btn", "Copy Span JSON");
        copySpanBtn.type = "button";
        copySpanBtn.addEventListener("click", () => {
            copyText(JSON.stringify(span, null, 2), copySpanBtn, "Copy Span JSON");
        });
        rightBtns.append(expandAllBtn, copySpanBtn);
        hdr.append(titleRow, rightBtns);
        container.appendChild(hdr);

        const timingRows = [
            ["Span ID", span.span_id],
            ["Run ID", span.run_id],
            ["Parent Span ID", span.parent_span_id || "(root)"],
            ["Effective Parent", span.effective_parent_id || "(root)"],
            ["Depth / Children", "Depth " + span.depth + " · " + (span.children_count || 0) + " child spans"],
            ["Start Offset", formatMs(span.offset_ms)],
            ["End Offset", formatMs(span.end_ms)],
            ["Total Duration", formatMs(span.duration_ms)],
            ["Self Duration", formatMs(span.self_duration_ms)]
        ];
        if (span.tokens) {
            timingRows.push([
                "Token Usage",
                "In: " + span.tokens.input_tokens + " · Out: " + span.tokens.output_tokens + " · Total: " + span.tokens.total_tokens
            ]);
        }
        container.appendChild(buildKeyValueTable(timingRows));

        const attrs = span.attributes || {};
        const attrKeys = Object.keys(attrs);
        const accordionSection = el("div", "debug-accordion-group");

        if (attrKeys.length === 0) {
            accordionSection.appendChild(el("div", "debug-muted-note", "Span này không có thuộc tính bổ sung."));
        } else {
            const scalarRows = [];
            const nestedEntries = [];
            for (const key of attrKeys) {
                const val = attrs[key];
                if (val !== null && typeof val === "object") {
                    nestedEntries.push([key, val]);
                } else {
                    scalarRows.push([key, val]);
                }
            }

            if (scalarRows.length > 0) {
                accordionSection.appendChild(
                    buildAccordionItem(
                        "Thuộc tính cơ bản (" + scalarRows.length + " trường)",
                        buildKeyValueTable(scalarRows),
                        true
                    )
                );
            }

            for (const entry of nestedEntries) {
                const keyName = entry[0];
                const nestedVal = entry[1];
                let countSuffix = "";
                if (Array.isArray(nestedVal)) {
                    countSuffix = " (" + nestedVal.length + " mục)";
                } else {
                    countSuffix = " (" + Object.keys(nestedVal).length + " khóa)";
                }
                const pre = el("pre", "debug-code-pre", JSON.stringify(nestedVal, null, 2));
                accordionSection.appendChild(
                    buildAccordionItem(keyName + countSuffix, pre, keyName === "decision_summary" || keyName === "params")
                );
            }
        }

        accordionSection.appendChild(
            buildAccordionItem(
                "Toàn bộ Raw Span JSON",
                el("pre", "debug-code-pre", JSON.stringify(span, null, 2)),
                false
            )
        );

        container.appendChild(accordionSection);

        expandAllBtn.addEventListener("click", () => {
            const shouldOpen = expandAllBtn.getAttribute("aria-expanded") !== "true";
            const items = accordionSection.querySelectorAll("details.debug-accordion-item");
            for (const det of items) {
                det.open = shouldOpen;
            }
            syncExpandAllButton(accordionSection, expandAllBtn);
        });
        accordionSection.addEventListener("toggle", () => {
            syncExpandAllButton(accordionSection, expandAllBtn);
        }, { capture: true });
        syncExpandAllButton(accordionSection, expandAllBtn);
    }

    function renderWaterfallTab(detail) {
        const panel = dom.panelWaterfallInner;
        if (!panel) {
            return;
        }
        panel.replaceChildren();

        let spans = [];
        if (Array.isArray(detail.spans)) {
            spans = detail.spans;
        }
        if (spans.length === 0) {
            panel.appendChild(el("div", "debug-muted-note", "Trace này chưa ghi nhận span nào."));
            return;
        }

        let maxMs = Number(detail.duration_ms) || 0;
        for (const sp of spans) {
            if (typeof sp.end_ms === "number" && sp.end_ms > maxMs) {
                maxMs = sp.end_ms;
            }
        }
        if (maxMs <= 0) {
            maxMs = 1.0;
        }

        const toolbar = el("div", "debug-waterfall-toolbar");
        const pillsWrap = el("div", "debug-filter-pills");
        const filterOptions = [
            { id: "all", label: "Tất cả (" + spans.length + ")" },
            { id: "chain", label: "chain" },
            { id: "llm", label: "llm" },
            { id: "tool", label: "tool" },
            { id: "retriever", label: "retriever" },
            { id: "parser", label: "parser" },
            { id: "embedding", label: "embedding" },
            { id: "error", label: "error" }
        ];

        for (const opt of filterOptions) {
            const pillBtn = el("button", "debug-filter-pill", opt.label);
            pillBtn.type = "button";
            if (state.spanFilterType === opt.id) {
                pillBtn.classList.add("active");
            }
            pillBtn.addEventListener("click", () => {
                state.spanFilterType = opt.id;
                renderWaterfallTab(detail);
            });
            pillsWrap.appendChild(pillBtn);
        }

        const searchInput = document.createElement("input");
        searchInput.type = "search";
        searchInput.className = "debug-span-search";
        searchInput.placeholder = "Lọc tên span hoặc thuộc tính...";
        searchInput.value = state.spanSearchQuery;
        searchInput.addEventListener("input", () => {
            state.spanSearchQuery = searchInput.value;
            renderWaterfallListOnly(listWrap, detail, maxMs, inspectorBox);
        });

        toolbar.append(pillsWrap, searchInput);
        panel.appendChild(toolbar);

        const listWrap = el("div", "debug-waterfall-list");
        const inspectorBox = el("div", "debug-span-inspector");

        renderWaterfallListOnly(listWrap, detail, maxMs, inspectorBox);
        panel.append(listWrap, inspectorBox);
    }

    function matchesSpanFilter(sp) {
        if (state.spanFilterType === "error") {
            const hasErr = sp.status === "error" || (sp.attributes && (sp.attributes.result === "failed" || Boolean(sp.attributes.error_type)));
            if (!hasErr) {
                return false;
            }
        } else if (state.spanFilterType !== "all") {
            if (String(sp.run_type || "").toLowerCase() !== state.spanFilterType) {
                return false;
            }
        }
        const q = state.spanSearchQuery.trim().toLowerCase();
        if (q) {
            const hay = (String(sp.name || "") + " " + String(sp.run_type || "") + " " + JSON.stringify(sp.attributes || {})).toLowerCase();
            if (!hay.includes(q)) {
                return false;
            }
        }
        return true;
    }

    function renderWaterfallListOnly(listWrap, detail, maxMs, inspectorBox) {
        listWrap.replaceChildren();
        let spans = [];
        if (Array.isArray(detail.spans)) {
            spans = detail.spans;
        }
        const visibleSpans = [];
        for (const sp of spans) {
            if (matchesSpanFilter(sp)) {
                visibleSpans.push(sp);
            }
        }

        const headerRow = el("div", "debug-wf-header");
        headerRow.append(
            el("div", "debug-wf-col-name", "Span Hierarchy"),
            el("div", "debug-wf-col-type", "Type"),
            el("div", "debug-wf-col-bar", "Timeline (0 → " + formatShortMs(maxMs) + ")"),
            el("div", "debug-wf-col-dur", "Duration / Self")
        );
        listWrap.appendChild(headerRow);

        if (visibleSpans.length === 0) {
            listWrap.appendChild(el("div", "debug-muted-note", "Không có span nào khớp bộ lọc."));
        } else {
            let activeSpanStillVisible = false;
            for (const sp of visibleSpans) {
                if (sp.span_id === state.selectedSpanId) {
                    activeSpanStillVisible = true;
                    break;
                }
            }
            if (!activeSpanStillVisible) {
                state.selectedSpanId = visibleSpans[0].span_id;
            }

            for (const sp of visibleSpans) {
                const rowBtn = document.createElement("button");
                rowBtn.type = "button";
                rowBtn.className = "debug-wf-row";
                if (sp.span_id === state.selectedSpanId) {
                    rowBtn.classList.add("selected");
                }

                const nameCol = el("div", "debug-wf-col-name");
                const indentPx = Math.min(Number(sp.depth || 0), 8) * 16;
                nameCol.style.paddingLeft = indentPx + "px";
                if (sp.depth > 0) {
                    nameCol.appendChild(el("span", "debug-wf-branch", "└─"));
                }
                nameCol.appendChild(el("span", "debug-wf-span-name", sp.name));
                if (sp.status && sp.status !== "ok") {
                    nameCol.appendChild(el("span", "debug-status-badge " + statusBadgeClass(sp.status), sp.status));
                }
                if (sp.tokens && typeof sp.tokens.total_tokens === "number" && sp.tokens.total_tokens > 0) {
                    nameCol.appendChild(el("span", "debug-token-chip", sp.tokens.total_tokens + " tok"));
                }

                const typeCol = el("div", "debug-wf-col-type");
                typeCol.appendChild(el("span", "debug-runtype-badge " + runTypeClass(sp.run_type), sp.run_type));

                const barCol = el("div", "debug-wf-col-bar");
                const track = el("div", "debug-wf-track");
                const fill = el("div", "debug-wf-fill " + runTypeClass(sp.run_type));
                if (sp.status === "error") {
                    fill.classList.add("is-error");
                }
                const leftPct = Math.min(98, Math.max(0, ((Number(sp.offset_ms || 0)) / maxMs) * 100));
                const rawWidthPct = ((Number(sp.duration_ms || 0)) / maxMs) * 100;
                const widthPct = Math.min(100 - leftPct, Math.max(1.8, rawWidthPct));
                fill.style.left = leftPct.toFixed(2) + "%";
                fill.style.width = widthPct.toFixed(2) + "%";
                fill.title = sp.name + " · start +" + formatShortMs(sp.offset_ms) + " · dur " + formatShortMs(sp.duration_ms);
                track.appendChild(fill);
                barCol.appendChild(track);

                const durCol = el("div", "debug-wf-col-dur");
                durCol.append(
                    el("span", "debug-wf-dur-total", formatShortMs(sp.duration_ms)),
                    el("span", "debug-wf-dur-self", "self " + formatShortMs(sp.self_duration_ms))
                );

                rowBtn.append(nameCol, typeCol, barCol, durCol);
                rowBtn.addEventListener("click", () => {
                    state.selectedSpanId = sp.span_id;
                    const allRows = listWrap.querySelectorAll(".debug-wf-row");
                    for (const r of allRows) {
                        r.classList.remove("selected");
                    }
                    rowBtn.classList.add("selected");
                    renderSelectedSpanInspector(inspectorBox, sp, detail);
                });
                listWrap.appendChild(rowBtn);
            }
        }

        let selectedSpan = null;
        for (const sp of spans) {
            if (sp.span_id === state.selectedSpanId) {
                selectedSpan = sp;
                break;
            }
        }
        renderSelectedSpanInspector(inspectorBox, selectedSpan, detail);
    }

    function renderLlmTab(detail) {
        const panel = dom.panelLlmInner;
        if (!panel) {
            return;
        }
        panel.replaceChildren();

        let calls = [];
        if (Array.isArray(detail.llm_calls)) {
            calls = detail.llm_calls;
        }
        if (calls.length === 0) {
            panel.appendChild(el("div", "debug-muted-note", "Chưa ghi nhận lần gọi model LLM nào trong trace này."));
            return;
        }

        const summary = detail.summary || {};
        const tok = summary.token_usage || {};
        let successCount = 0;
        let failCount = 0;
        for (const c of calls) {
            if (c.result === "failed" || c.status === "error" || Boolean(c.error_type)) {
                failCount += 1;
            } else if (c.run_type === "llm") {
                successCount += 1;
            }
        }

        const summaryStrip = el("div", "debug-section-summary");
        summaryStrip.append(
            el("span", "debug-summary-chip", "Tổng events: " + calls.length),
            el("span", "debug-summary-chip status-ok", "Thành công: " + successCount),
            el("span", "debug-summary-chip status-error", "Thất bại / Fallback: " + failCount),
            el("span", "debug-summary-chip", "Tokens: " + (tok.input_tokens || 0) + " in / " + (tok.output_tokens || 0) + " out (" + (tok.total_tokens || 0) + " total)")
        );
        panel.appendChild(summaryStrip);

        for (let idx = 0; idx < calls.length; idx++) {
            const c = calls[idx];
            const card = el("div", "debug-detail-card");

            const cardHdr = el("div", "debug-detail-card-header");
            const left = el("div", "debug-detail-card-title");
            left.append(
                el("span", "debug-runtype-badge " + runTypeClass(c.run_type), c.name),
                el("strong", "", c.model || "(no model)"),
                el("span", "debug-pill debug-pill-muted", "Role: " + (c.role || c.configured_role || "—"))
            );
            if (c.alias) {
                left.appendChild(el("span", "debug-pill debug-pill-info", "Profile: " + c.alias));
            }
            if (c.reason) {
                left.appendChild(el("span", "debug-pill debug-pill-muted", "Reason: " + c.reason));
            }

            const right = el("div", "debug-detail-card-meta");
            let statusVal = c.result || c.status || "ok";
            right.append(
                el("span", "debug-status-badge " + statusBadgeClass(statusVal), statusVal),
                el("span", "debug-mono-chip", formatMs(c.duration_ms)),
                el("span", "debug-mono-chip", "+" + formatShortMs(c.offset_ms))
            );
            cardHdr.append(left, right);
            card.appendChild(cardHdr);

            const metricsGrid = el("div", "debug-metrics-grid");
            const tokObj = c.tokens || {};
            let attemptLabel = "—";
            if (c.attempt !== undefined && c.attempt !== null) {
                attemptLabel = "#" + c.attempt;
            }
            let inTokLabel = "—";
            if (tokObj.input_tokens !== undefined) {
                inTokLabel = String(tokObj.input_tokens);
            }
            let outTokLabel = "—";
            if (tokObj.output_tokens !== undefined) {
                outTokLabel = String(tokObj.output_tokens);
            }
            let totTokLabel = "—";
            if (tokObj.total_tokens !== undefined) {
                totTokLabel = String(tokObj.total_tokens);
            }
            let tempLabel = "—";
            if (c.temperature !== undefined && c.temperature !== null) {
                tempLabel = String(c.temperature);
            }
            let maxTokLabel = "—";
            if (c.output_tokens_limit) {
                maxTokLabel = String(c.output_tokens_limit);
            } else if (c.max_tokens) {
                maxTokLabel = String(c.max_tokens);
            } else if (c.max_completion_tokens) {
                maxTokLabel = String(c.max_completion_tokens);
            }
            let ctxWinLabel = "—";
            if (c.context_window) {
                ctxWinLabel = String(c.context_window);
            }
            let repairLabel = "Không";
            if (c.repair) {
                repairLabel = "Có";
            }

            const cleanMetrics = [
                ["Attempt", attemptLabel],
                ["Protocol", c.protocol || "—"],
                ["Prompt Tokens", inTokLabel],
                ["Completion Tokens", outTokLabel],
                ["Total Tokens", totTokLabel],
                ["Temperature", tempLabel],
                ["Reasoning Effort", c.reasoning_effort || "—"],
                ["Max Tokens", maxTokLabel],
                ["Context Window", ctxWinLabel],
                ["JSON Repair", repairLabel]
            ];

            for (const m of cleanMetrics) {
                const cell = el("div", "debug-metric-cell");
                cell.append(el("div", "debug-metric-key", m[0]), el("div", "debug-metric-val", m[1]));
                metricsGrid.appendChild(cell);
            }
            card.appendChild(metricsGrid);

            if (typeof c.input_bytes === "number" && typeof c.input_budget_bytes === "number" && c.input_budget_bytes > 0) {
                const pct = Math.min(100, Math.round((c.input_bytes / c.input_budget_bytes) * 100));
                const budgetWrap = el("div", "debug-budget-wrap");
                budgetWrap.appendChild(
                    el("div", "debug-budget-label", "Input Bytes / Budget: " + c.input_bytes + " B / " + c.input_budget_bytes + " B (" + pct + "%)")
                );
                const barTrack = el("div", "debug-budget-track");
                const barFill = el("div", "debug-budget-fill");
                if (pct >= 85) {
                    barFill.classList.add("is-high");
                }
                barFill.style.width = pct + "%";
                barTrack.appendChild(barFill);
                budgetWrap.appendChild(barTrack);
                card.appendChild(budgetWrap);
            }

            if (c.error_type || c.http_status) {
                let errText = "Lỗi: " + (c.error_type || "Error");
                if (c.http_status) {
                    errText += " (HTTP " + c.http_status + ")";
                }
                card.appendChild(el("div", "debug-error-banner", errText));
            }

            card.appendChild(
                buildAccordionItem(
                    "Chi tiết Attributes & Payload Metadata",
                    el("pre", "debug-code-pre", JSON.stringify(c.attributes || {}, null, 2)),
                    false
                )
            );
            panel.appendChild(card);
        }
    }

    function renderRagTable(items, mode) {
        const wrap = el("div", "debug-table-wrap");
        const table = document.createElement("table");
        table.className = "debug-table";

        const thead = document.createElement("thead");
        const trHead = document.createElement("tr");
        let headers = [];
        if (mode === "rerank") {
            headers = ["#", "Tài liệu / Mục", "Trang", "Rerank Score", "RRF Score", "Method"];
        } else {
            headers = ["#", "Tài liệu / Mục", "Loại", "RRF Score", "Dense (#)", "BM25 (#)"];
        }
        for (const h of headers) {
            trHead.appendChild(el("th", "", h));
        }
        thead.appendChild(trHead);
        table.appendChild(thead);

        const tbody = document.createElement("tbody");
        for (let i = 0; i < items.length; i++) {
            const it = items[i];
            const tr = document.createElement("tr");

            let rankVal = it.rank;
            if (rankVal === undefined || rankVal === null) {
                rankVal = i + 1;
            }
            tr.appendChild(el("td", "debug-td-num", "#" + rankVal));

            const docCell = el("td", "debug-td-doc");
            const docName = String(it.source || "—").split("\\").join("/").split("/").pop();
            docCell.appendChild(el("div", "debug-td-title", docName));
            if (it.section_title || it.breadcrumb) {
                docCell.appendChild(el("div", "debug-td-sub", it.section_title || it.breadcrumb));
            }
            const chunkText = String(it.content || it.snippet || it.excerpt || it.excerpt_preview || "").trim();
            if (chunkText) {
                const openFirstThree = i < 3;
                const charLabel = String(it.char_count || chunkText.length) + " ký tự";
                docCell.appendChild(
                    buildAccordionItem(
                        "Nội dung chunk (" + (it.chunk_id || "chunk") + " · " + charLabel + ")",
                        el("div", "debug-snippet-text", chunkText),
                        openFirstThree
                    )
                );
            }
            tr.appendChild(docCell);

            if (mode === "rerank") {
                let pageLabel = "—";
                if (Array.isArray(it.page_numbers) && it.page_numbers.length > 0) {
                    pageLabel = "Tr. " + it.page_numbers.join(", ");
                }
                tr.appendChild(el("td", "", pageLabel));

                let rrScore = "—";
                if (typeof it.rerank_score === "number") {
                    rrScore = it.rerank_score.toFixed(4);
                }
                tr.appendChild(el("td", "debug-td-score", rrScore));

                let rrfScore = "—";
                if (typeof it.rrf_score === "number") {
                    rrfScore = it.rrf_score.toFixed(4);
                }
                tr.appendChild(el("td", "debug-td-num", rrfScore));
                tr.appendChild(el("td", "", it.rerank_method || "—"));
            } else {
                tr.appendChild(el("td", "", it.chunk_type || "text"));

                let rrfScore = "—";
                if (typeof it.rrf_score === "number") {
                    rrfScore = it.rrf_score.toFixed(4);
                }
                tr.appendChild(el("td", "debug-td-score", rrfScore));

                let denseLabel = "—";
                if (typeof it.dense_score === "number") {
                    denseLabel = it.dense_score.toFixed(4);
                    if (it.dense_rank) {
                        denseLabel += " (#" + it.dense_rank + ")";
                    }
                }
                tr.appendChild(el("td", "debug-td-num", denseLabel));

                let bm25Label = "—";
                if (typeof it.bm25_score === "number") {
                    bm25Label = it.bm25_score.toFixed(2);
                    if (it.bm25_rank) {
                        bm25Label += " (#" + it.bm25_rank + ")";
                    }
                }
                tr.appendChild(el("td", "debug-td-num", bm25Label));
            }
            tbody.appendChild(tr);
        }
        table.appendChild(tbody);
        wrap.appendChild(table);
        return wrap;
    }

    function renderPillsGroup(items) {
        if (!Array.isArray(items) || items.length === 0) {
            return el("span", "is-muted", "—");
        }
        const wrap = el("div", "debug-inline-pills");
        for (const it of items) {
            let label = "";
            if (typeof it === "object" && it !== null) {
                label = JSON.stringify(it);
            } else {
                label = String(it);
            }
            wrap.appendChild(el("span", "debug-pill debug-pill-info", label));
        }
        return wrap;
    }

    function renderRagTab(detail) {
        const panel = dom.panelRagInner;
        if (!panel) {
            return;
        }
        panel.replaceChildren();

        const stages = detail.rag_stages || {};
        let parsers = [];
        if (Array.isArray(stages.query_parsers)) {
            parsers = stages.query_parsers;
        }
        let retrievals = [];
        if (Array.isArray(stages.retrievals)) {
            retrievals = stages.retrievals;
        }
        let rerankers = [];
        if (Array.isArray(stages.rerankers)) {
            rerankers = stages.rerankers;
        }
        let embeddings = [];
        if (Array.isArray(stages.embeddings)) {
            embeddings = stages.embeddings;
        }
        const snap = detail.turn_snapshot || {};
        let finalChunks = [];
        if (Array.isArray(snap.retrieved_chunks)) {
            finalChunks = snap.retrieved_chunks;
        }
        if (finalChunks.length === 0) {
            for (const rr of rerankers) {
                const rra = rr.attributes || {};
                if (Array.isArray(rra.reranked_preview) && rra.reranked_preview.length > 0) {
                    finalChunks = rra.reranked_preview;
                    break;
                }
            }
        }
        if (finalChunks.length === 0) {
            for (const rt of retrievals) {
                const rta = rt.attributes || {};
                if (Array.isArray(rta.candidates_preview) && rta.candidates_preview.length > 0) {
                    finalChunks = rta.candidates_preview;
                    break;
                }
            }
        }

        if (parsers.length === 0 && retrievals.length === 0 && rerankers.length === 0 && embeddings.length === 0 && finalChunks.length === 0) {
            const turnIntent = String(snap.intent || (detail.summary && detail.summary.intent) || "");
            let emptyText = "Turn này không thực hiện bước truy xuất RAG nào";
            if (turnIntent) {
                emptyText = emptyText + " (Intent: " + turnIntent + ").";
            } else {
                emptyText = emptyText + ".";
            }
            emptyText = emptyText + " Hãy chọn một trace có trạng thái RAG_SEARCH ở danh sách bên trái để xem nội dung các chunk đã truy xuất.";
            panel.appendChild(el("div", "debug-muted-note", emptyText));
            return;
        }

        if (finalChunks.length > 0) {
            const evHeadingRow = el("div", "debug-section-header-row");
            evHeadingRow.appendChild(
                el("h4", "debug-section-heading", "1. Nội dung Chunks trả về từ RAG (" + finalChunks.length + " chunks)")
            );
            const acc = el("div", "debug-accordion-group");
            let allExpanded = false;
            const toggleAllBtn = el("button", "debug-mini-btn", "Mở rộng tất cả");
            toggleAllBtn.type = "button";
            toggleAllBtn.addEventListener("click", () => {
                allExpanded = !allExpanded;
                for (const detItem of acc.querySelectorAll("details.debug-accordion-item")) {
                    detItem.open = allExpanded;
                }
                if (allExpanded) {
                    toggleAllBtn.textContent = "Thu gọn tất cả";
                } else {
                    toggleAllBtn.textContent = "Mở rộng tất cả";
                }
            });
            evHeadingRow.appendChild(toggleAllBtn);
            panel.appendChild(evHeadingRow);

            for (let idx = 0; idx < finalChunks.length; idx++) {
                const ch = finalChunks[idx];
                const docName = String(ch.source || "—").split("\\").join("/").split("/").pop();
                let rankNum = idx + 1;
                if (ch.index !== undefined && ch.index !== null) {
                    rankNum = ch.index;
                } else if (ch.rank !== undefined && ch.rank !== null) {
                    rankNum = ch.rank;
                }
                let title = "[" + rankNum + "] " + docName;
                if (ch.section_title) {
                    title += " · " + ch.section_title;
                }
                if (Array.isArray(ch.page_numbers) && ch.page_numbers.length > 0) {
                    title += " (Tr. " + ch.page_numbers.join(", ") + ")";
                }
                const body = el("div", "debug-chunk-body");
                const metaBar = el("div", "debug-chunk-meta-bar");
                metaBar.append(
                    el("span", "debug-mono-chip", "ID: " + (ch.chunk_id || "—")),
                    el("span", "debug-pill debug-pill-info", String(ch.chunk_type || "text"))
                );
                if (Array.isArray(ch.page_numbers) && ch.page_numbers.length > 0) {
                    metaBar.appendChild(el("span", "debug-pill debug-pill-muted", "Trang " + ch.page_numbers.join(", ")));
                }
                const chunkContentText = String(ch.content || ch.snippet || ch.excerpt || ch.excerpt_preview || "").trim();
                const charsLen = Number(ch.char_count || chunkContentText.length || 0);
                metaBar.appendChild(el("span", "debug-mono-chip", charsLen + " ký tự"));
                body.appendChild(metaBar);

                if (ch.source || ch.breadcrumb) {
                    body.appendChild(buildKeyValueTable([
                        ["Source", ch.source],
                        ["Breadcrumb", ch.breadcrumb]
                    ]));
                }
                const contentPre = el("pre", "debug-chunk-content-pre");
                if (chunkContentText) {
                    contentPre.textContent = chunkContentText;
                } else {
                    contentPre.textContent = "— Không có nội dung văn bản —";
                }
                body.appendChild(contentPre);
                acc.appendChild(buildAccordionItem(title, body, idx < 5));
            }
            panel.appendChild(acc);
        }

        if (parsers.length > 0) {
            panel.appendChild(el("h4", "debug-section-heading", "2. Query Parser & Chuẩn hóa truy vấn (" + parsers.length + ")"));
            for (const p of parsers) {
                const a = p.attributes || {};
                const card = el("div", "debug-detail-card");
                const hdr = el("div", "debug-detail-card-header");
                hdr.append(
                    el("strong", "", "Query Parser"),
                    el("span", "debug-mono-chip", formatMs(p.duration_ms))
                );
                card.appendChild(hdr);
                card.appendChild(buildKeyValueTable([
                    ["Raw Query", a.raw_query],
                    ["Normalized Query", a.normalized_query],
                    ["Expanded Query", a.expanded_query],
                    ["Parsed Intent / Domain", [a.parsed_intent, a.parsed_sub_intent, a.parsed_domain].filter(Boolean).join(" · ")],
                    ["Keywords", renderPillsGroup(a.keywords)],
                    ["Detected Acronyms", renderPillsGroup(a.detected_acronyms || a.acronyms)],
                    ["Legal References", renderPillsGroup(a.legal_references)],
                    ["Financial Amounts", renderPillsGroup(a.financial_amounts)],
                    ["Banks Mentioned", renderPillsGroup(a.banks_mentioned)],
                    ["Suggested Filters", a.suggested_filters],
                    ["Exact / Source Filter", [a.exact_filter, a.source_filter].filter(Boolean).join(" | ")]
                ]));
                panel.appendChild(card);
            }
        }

        if (retrievals.length > 0) {
            panel.appendChild(el("h4", "debug-section-heading", "3. Hybrid Retrieval — Dense Vector + BM25 + RRF (" + retrievals.length + ")"));
            for (const r of retrievals) {
                const a = r.attributes || {};
                const card = el("div", "debug-detail-card");
                const hdr = el("div", "debug-detail-card-header");
                hdr.append(
                    el("strong", "", a.query || r.name),
                    el("div", "debug-detail-card-meta", "")
                );
                let candCnt = 0;
                if (a.candidate_count !== undefined) {
                    candCnt = a.candidate_count;
                } else if (a.retrieved_candidates !== undefined) {
                    candCnt = a.retrieved_candidates;
                }
                const meta = hdr.lastChild;
                meta.append(
                    el("span", "debug-pill debug-pill-info", "Candidates: " + candCnt),
                    el("span", "debug-pill debug-pill-muted", "top_k=" + (a.top_k || "—") + " · candidate_k=" + (a.candidate_k || "—")),
                    el("span", "debug-mono-chip", formatMs(r.duration_ms))
                );
                card.appendChild(hdr);

                let relaxedText = "Không";
                if (a.relaxed_filters) {
                    relaxedText = "Có (Fallback unfiltered)";
                }
                card.appendChild(buildKeyValueTable([
                    ["Filters", a.filters],
                    ["Relaxed Filters", relaxedText],
                    ["Offset", a.offset]
                ]));

                if (Array.isArray(a.candidates_preview) && a.candidates_preview.length > 0) {
                    card.appendChild(renderRagTable(a.candidates_preview, "retrieval"));
                }
                panel.appendChild(card);
            }
        }

        if (rerankers.length > 0) {
            panel.appendChild(el("h4", "debug-section-heading", "4. Cross-Encoder Reranker (" + rerankers.length + ")"));
            for (const rr of rerankers) {
                const a = rr.attributes || {};
                const card = el("div", "debug-detail-card");
                const hdr = el("div", "debug-detail-card-header");
                hdr.append(el("strong", "", a.query || rr.name));
                const meta = el("div", "debug-detail-card-meta");
                meta.append(
                    el("span", "debug-pill debug-pill-ok", (a.input_count || 0) + " → " + (a.reranked_count || a.top_reranked_count || 0) + " chunks"),
                    el("span", "debug-mono-chip", formatMs(rr.duration_ms))
                );
                hdr.appendChild(meta);
                card.appendChild(hdr);
                if (Array.isArray(a.reranked_preview) && a.reranked_preview.length > 0) {
                    card.appendChild(renderRagTable(a.reranked_preview, "rerank"));
                }
                panel.appendChild(card);
            }
        }

        if (embeddings.length > 0) {
            panel.appendChild(el("h4", "debug-section-heading", "5. Dense Embeddings (" + embeddings.length + ")"));
            for (const em of embeddings) {
                const card = el("div", "debug-detail-card");
                card.appendChild(buildKeyValueTable(Object.entries(em.attributes || {})));
                panel.appendChild(card);
            }
        }
    }

    const LANGGRAPH_CANONICAL_NODES = [
        {
            id: "input_guardrail",
            label: "Input Guardrail",
            groupId: "init",
            role: "Kiểm duyệt an toàn đầu vào",
            desc: "Kiểm tra an toàn câu hỏi của người dùng, phát hiện prompt injection và từ khóa nhạy cảm."
        },
        {
            id: "context_prepare",
            label: "Context Prepare",
            groupId: "init",
            role: "Chuẩn bị ngữ cảnh & bộ nhớ",
            desc: "Nén lịch sử hội thoại, đồng bộ checkpoint và nạp bộ nhớ dài hạn của phiên."
        },
        {
            id: "deep_decide",
            label: "Deep Decide (Coordinator)",
            groupId: "coordinator",
            role: "Điều phối suy luận & ra quyết định",
            desc: "Bộ điều phối trung tâm: phân tích intent, chọn profile mô hình, lập kế hoạch ReAct và quyết định gọi tool hay trả lời."
        },
        {
            id: "react_action",
            label: "ReAct Action (Tools)",
            groupId: "actions",
            role: "Thực thi công cụ nghiệp vụ",
            desc: "Gọi các công cụ ngân hàng: tra cứu kiến thức RAG, tính lãi suất tiết kiệm, tra cứu biểu phí thẻ, web search."
        },
        {
            id: "skill_inspect",
            label: "Skill Inspect",
            groupId: "actions",
            role: "Nạp kỹ năng chuyên sâu",
            desc: "Nạp và kiểm tra hướng dẫn kỹ năng nghiệp vụ chuyên biệt cho các tình huống phức tạp."
        },
        {
            id: "plan_update",
            label: "Plan Update",
            groupId: "actions",
            role: "Cập nhật kế hoạch ReAct",
            desc: "Cập nhật các bước giải quyết câu hỏi động khi thu thập thêm dữ kiện mới."
        },
        {
            id: "subagent",
            label: "Subagent Delegation",
            groupId: "actions",
            role: "Ủy quyền Subagent",
            desc: "Ủy thác các nhiệm vụ con chuyên sâu cho subagent độc lập xử lý."
        },
        {
            id: "memory_update",
            label: "Memory Update",
            groupId: "actions",
            role: "Cập nhật bộ nhớ dài hạn",
            desc: "Lưu trữ thông tin quan trọng vào bộ nhớ dài hạn để duy trì ngữ cảnh các lượt sau."
        },
        {
            id: "answer_review",
            label: "Answer Review (Audit)",
            groupId: "review",
            role: "Kiểm toán ảo giác & căn cứ",
            desc: "Kiểm toán câu trả lời dự thảo: nếu phát hiện ảo giác hoặc thiếu căn cứ tài liệu thì kích hoạt vòng RETRY về deep_decide."
        },
        {
            id: "prepare_interaction",
            label: "Prepare Interaction",
            groupId: "review",
            role: "Chuẩn bị tương tác người dùng",
            desc: "Thiết lập trạng thái chờ người dùng xác nhận hoặc làm rõ câu hỏi khi thiếu thông tin."
        },
        {
            id: "await_interaction",
            label: "Await Interaction",
            groupId: "review",
            role: "Chờ người dùng phản hồi",
            desc: "Tạm dừng luồng LangGraph đợi người dùng gửi câu trả lời qua checkpoint persistent."
        },
        {
            id: "output_guardrail",
            label: "Output Guardrail",
            groupId: "exit",
            role: "Kiểm duyệt đầu ra & Trích dẫn",
            desc: "Kiểm duyệt an toàn câu trả lời cuối cùng, đối chiếu và định dạng trích dẫn nguồn tài liệu chuẩn."
        }
    ];

    function debugDagProfileShort(profileVal) {
        const upper = String(profileVal || "DETERMINISTIC").toUpperCase();
        if (upper === "DEEP") {
            return "DEEP";
        }
        if (upper === "STANDARD") {
            return "STD";
        }
        return "RULE";
    }

    function resolveLangGraphExecutionState(detail, dag) {
        const nodeMap = {};
        for (let i = 0; i < LANGGRAPH_CANONICAL_NODES.length; i++) {
            const def = LANGGRAPH_CANONICAL_NODES[i];
            nodeMap[def.id] = {
                id: def.id,
                label: def.label,
                groupId: def.groupId,
                role: def.role,
                desc: def.desc,
                executed: false,
                call_count: 0,
                latency_ms: 0,
                status: "IDLE",
                steps: [],
                executions: [],
                input_summary: {},
                output_summary: {}
            };
        }

        const executionTrail = [];

        let execPath = [];
        if (detail && Array.isArray(detail.langgraph_execution_path)) {
            execPath = detail.langgraph_execution_path;
        }

        for (let i = 0; i < execPath.length; i++) {
            const item = execPath[i];
            const nid = String(item.node || "");
            if (nodeMap[nid]) {
                const target = nodeMap[nid];
                target.executed = true;
                target.call_count = target.call_count + 1;
                target.latency_ms = Number((target.latency_ms + Number(item.duration_ms || 0)).toFixed(2));
                const stepVal = Number(item.step || (i + 1));
                target.steps.push(stepVal);
                const attrs = item.attributes || {};
                if (nid === "input_guardrail") {
                    if (attrs.is_safe === false) {
                        target.status = "BLOCKED";
                    } else {
                        target.status = "PASS";
                    }
                } else if (item.status && item.status !== "ok" && String(item.status).toUpperCase() !== "PENDING") {
                    target.status = String(item.status);
                } else if (target.status === "IDLE" || target.status === "PENDING") {
                    target.status = "PASS";
                }
                target.output_summary = attrs;
                target.executions.push(item);
                executionTrail.push({
                    step: stepVal,
                    nodeId: nid,
                    label: target.label,
                    duration_ms: Number(item.duration_ms || 0),
                    status: target.status,
                    span_id: item.span_id || null,
                    attributes: attrs
                });
            }
        }

        if (executionTrail.length === 0 && detail) {
            let spansToCheck = [];
            if (Array.isArray(detail.graph_nodes) && detail.graph_nodes.length > 0) {
                spansToCheck = detail.graph_nodes;
            } else if (Array.isArray(detail.spans)) {
                spansToCheck = detail.spans;
            }

            for (let i = 0; i < spansToCheck.length; i++) {
                const sp = spansToCheck[i];
                let sName = String(sp.name || "");
                if (sName.startsWith("graph.")) {
                    sName = sName.slice(6);
                }
                if (nodeMap[sName]) {
                    const target = nodeMap[sName];
                    target.executed = true;
                    target.call_count = target.call_count + 1;
                    target.latency_ms = Number((target.latency_ms + Number(sp.duration_ms || 0)).toFixed(2));
                    const stepNum = executionTrail.length + 1;
                    target.steps.push(stepNum);
                    const attrs = sp.attributes || {};
                    if (sName === "input_guardrail") {
                        if (attrs.is_safe === false) {
                            target.status = "BLOCKED";
                        } else {
                            target.status = "PASS";
                        }
                    } else if (attrs.audit_status && String(attrs.audit_status).toUpperCase() !== "PENDING") {
                        target.status = String(attrs.audit_status);
                    } else if (sp.status === "error") {
                        target.status = "ERROR";
                    } else if (sp.status === "suspended") {
                        target.status = "AWAITING_INPUT";
                    } else if (target.status === "IDLE" || target.status === "PENDING") {
                        target.status = "PASS";
                    }
                    target.output_summary = attrs;
                    target.executions.push({
                        step: stepNum,
                        duration_ms: Number(sp.duration_ms || 0),
                        attributes: attrs,
                        status: target.status
                    });
                    executionTrail.push({
                        step: stepNum,
                        nodeId: sName,
                        label: target.label,
                        duration_ms: Number(sp.duration_ms || 0),
                        status: target.status,
                        span_id: sp.span_id || null,
                        attributes: attrs
                    });
                }
            }
        }

        if (detail) {
            const snap = detail.turn_snapshot || {};
            if (Array.isArray(detail.tool_calls) && detail.tool_calls.length > 0 && !nodeMap["react_action"].executed) {
                nodeMap["react_action"].executed = true;
                nodeMap["react_action"].call_count = detail.tool_calls.length;
                nodeMap["react_action"].status = "PASS";
                nodeMap["react_action"].steps.push(executionTrail.length + 1);
            }
            if (snap.interaction && !nodeMap["prepare_interaction"].executed) {
                nodeMap["prepare_interaction"].executed = true;
                nodeMap["prepare_interaction"].status = "AWAITING_INPUT";
                nodeMap["await_interaction"].executed = true;
                nodeMap["await_interaction"].status = "AWAITING_INPUT";
            }
            if ((snap.review_outcomes || snap.audit_status) && !nodeMap["answer_review"].executed) {
                nodeMap["answer_review"].executed = true;
                nodeMap["answer_review"].status = String(snap.audit_status || "PASS");
            }
            if (snap.output_guard && !nodeMap["output_guardrail"].executed) {
                nodeMap["output_guardrail"].executed = true;
                nodeMap["output_guardrail"].status = "PASS";
            }
            if (snap.input_guard && !nodeMap["input_guardrail"].executed) {
                nodeMap["input_guardrail"].executed = true;
                nodeMap["input_guardrail"].status = "PASS";
            }
            if (snap.query && !nodeMap["deep_decide"].executed) {
                nodeMap["deep_decide"].executed = true;
                nodeMap["deep_decide"].status = "PASS";
            }
            if (snap.query && !nodeMap["context_prepare"].executed) {
                nodeMap["context_prepare"].executed = true;
                nodeMap["context_prepare"].status = "PASS";
            }
        }

        enrichTrailStepsWithIO(executionTrail, nodeMap, detail);

        let executedCount = 0;
        for (let i = 0; i < LANGGRAPH_CANONICAL_NODES.length; i++) {
            const def = LANGGRAPH_CANONICAL_NODES[i];
            if (nodeMap[def.id].executed) {
                executedCount = executedCount + 1;
            }
        }

        return {
            nodes: nodeMap,
            trail: executionTrail,
            executedCount: executedCount,
            totalCount: LANGGRAPH_CANONICAL_NODES.length
        };
    }

    function collectDescendantSpans(allSpans, rootSpanId) {
        const result = [];
        if (!rootSpanId || !Array.isArray(allSpans)) {
            return result;
        }
        const idSet = {};
        idSet[String(rootSpanId)] = true;
        let added = true;
        while (added) {
            added = false;
            for (let i = 0; i < allSpans.length; i++) {
                const sp = allSpans[i];
                const sid = String(sp.span_id || "");
                const pid = String(sp.parent_span_id || "");
                if (sid && pid && idSet[pid] && !idSet[sid]) {
                    idSet[sid] = true;
                    result.push(sp);
                    added = true;
                }
            }
        }
        return result;
    }

    function truncatePreviewStr(txt, maxLen) {
        const s = String(txt || "").trim();
        if (!s) {
            return "";
        }
        if (s.length <= maxLen) {
            return s;
        }
        return s.slice(0, maxLen - 3) + "...";
    }

    function enrichTrailStepsWithIO(executionTrail, nodeMap, detail) {
        if (!Array.isArray(executionTrail) || !detail) {
            return;
        }
        const snap = detail.turn_snapshot || {};
        const summary = detail.summary || {};
        let allSpans = [];
        if (Array.isArray(detail.spans)) {
            allSpans = detail.spans;
        }
        const queryText = truncatePreviewStr(snap.query || summary.query_preview || "", 260);
        const answerText = truncatePreviewStr(snap.final_answer || "", 260);

        for (let i = 0; i < executionTrail.length; i++) {
            const cur = executionTrail[i];
            let prev = null;
            if (i > 0) {
                prev = executionTrail[i - 1];
            }
            const attrs = cur.attributes || {};
            const childSpans = collectDescendantSpans(allSpans, cur.span_id);
            const llmModels = [];
            let stepTokens = 0;
            const toolsExecuted = [];
            let parserSummary = null;
            let retrievalCount = null;
            let rerankCount = null;

            for (let cIdx = 0; cIdx < childSpans.length; cIdx++) {
                const cs = childSpans[cIdx];
                const csAttrs = cs.attributes || {};
                if (cs.run_type === "llm") {
                    if (csAttrs.model && !llmModels.includes(csAttrs.model)) {
                        llmModels.push(csAttrs.model);
                    }
                    const usage = csAttrs.usage || {};
                    stepTokens = stepTokens + Number(usage.total_tokens || 0);
                } else if (cs.run_type === "tool") {
                    toolsExecuted.push({
                        name: csAttrs.tool_name || csAttrs.action_name || cs.name,
                        params: csAttrs.params || {},
                        status: csAttrs.status || cs.status,
                        latency_ms: cs.duration_ms
                    });
                } else if (cs.run_type === "parser" || cs.name === "query_parser") {
                    parserSummary = csAttrs;
                } else if (cs.name === "top_k_hybrid_retrieval") {
                    retrievalCount = csAttrs.candidate_count;
                } else if (cs.name === "top_k_cross_encoder_reranker") {
                    rerankCount = csAttrs.reranked_count;
                }
            }

            let stepIn = {};
            let stepOut = {};
            let caption = cur.label + " thực thi hoàn tất (" + formatShortMs(cur.duration_ms) + ").";

            if (cur.nodeId === "input_guardrail") {
                stepIn = {
                    nguon_vao: "START (Người dùng gửi câu hỏi)",
                    truy_van: queryText,
                    session_ref: detail.session_ref || ""
                };
                stepOut = {
                    ket_luan_an_toan: attrs.is_safe,
                    trang_thai: cur.status,
                    co_canh_bao: attrs.flags || []
                };
                if (attrs.is_safe === false) {
                    caption = "Phát hiện nội dung cần chặn tại Input Guardrail → Chuyển thẳng tới Output Guardrail.";
                } else {
                    caption = "Kiểm duyệt đầu vào an toàn (PASS) → Chuyển sang Context Prepare.";
                }
            } else if (cur.nodeId === "context_prepare") {
                stepIn = {
                    tu_node: "input_guardrail (is_safe = true)",
                    truy_van: queryText,
                    intent: snap.intent || summary.intent || "UNKNOWN"
                };
                stepOut = {
                    so_tin_nhan_lich_su: attrs.history_count,
                    so_lan_nen: attrs.compactions,
                    trang_thai_bo_nho: attrs.memory_status || {}
                };
                let histCnt = 0;
                if (typeof attrs.history_count === "number") {
                    histCnt = attrs.history_count;
                }
                caption = "Nạp " + histCnt + " tin nhắn lịch sử & trạng thái bộ nhớ dài hạn → Chuyển tới Deep Decide.";
            } else if (cur.nodeId === "deep_decide") {
                let prevSummary = "Context Prepare (Ngữ cảnh phiên sẵn sàng)";
                const prevPayload = {};
                if (prev) {
                    prevPayload.tu_node = prev.nodeId + " (Bước " + prev.step + ")";
                    if (prev.nodeId === "react_action" && prev.attributes) {
                        prevSummary = "Kết quả từ " + prev.label + " (Bước " + prev.step + ")";
                        prevPayload.ket_qua_tool = prev.attributes.action_name || prev.attributes.action_kind || "";
                        prevPayload.so_bang_chung = prev.attributes.evidence_count;
                    } else if (prev.nodeId === "answer_review") {
                        prevSummary = "Yêu cầu sửa lại từ Answer Review (RETRY)";
                        prevPayload.trang_thai_kiem_toan = prev.status;
                    } else if (prev.nodeId === "context_prepare" && prev.attributes) {
                        prevPayload.so_tin_nhan_lich_su = prev.attributes.history_count;
                    }
                }
                stepIn = {
                    nguon_vao: prevSummary,
                    du_lieu_nhan: prevPayload,
                    truy_van: queryText,
                    model_suy_luan: llmModels.join(", ") || (summary.models_used && summary.models_used[0]) || "deterministic"
                };
                stepOut = {
                    quyet_dinh: attrs.decision_kind || "final",
                    chi_tiet_quyet_dinh: attrs.decision_summary || {},
                    chuyen_toi_node: attrs.next_node || "",
                    tokens_su_dung: stepTokens
                };
                const decSum = attrs.decision_summary || {};
                if (attrs.decision_kind === "tool" && decSum.name) {
                    caption = "Deep Decide phân tích xong → Yêu cầu gọi tool \"" + decSum.name + "\" tại " + (attrs.next_node || "react_action") + ".";
                } else if (attrs.next_node === "answer_review") {
                    caption = "Deep Decide đã đủ dữ kiện & soạn nháp câu trả lời → Gửi qua Answer Review kiểm toán.";
                } else if (attrs.next_node === "prepare_interaction") {
                    caption = "Deep Decide thấy cần hỏi thêm khách hàng → Chuyển sang luồng HITL.";
                } else {
                    caption = "Deep Decide quyết định hướng đi tiếp → " + (attrs.next_node || "bước kế tiếp") + ".";
                }
            } else if (cur.nodeId === "react_action" || cur.nodeId === "skill_inspect" || cur.nodeId === "plan_update" || cur.nodeId === "subagent" || cur.nodeId === "memory_update") {
                let reqSummary = {};
                let prevStepLabel = "0";
                if (prev) {
                    prevStepLabel = String(prev.step);
                    if (prev.attributes && prev.attributes.decision_summary) {
                        reqSummary = prev.attributes.decision_summary;
                    }
                }
                stepIn = {
                    tu_node: "deep_decide (Bước " + prevStepLabel + ")",
                    lenh_thuc_thi: reqSummary,
                    cong_cu_da_goi: toolsExecuted
                };
                const chunksSummary = [];
                if (Array.isArray(snap.retrieved_chunks)) {
                    for (let chIdx = 0; chIdx < Math.min(3, snap.retrieved_chunks.length); chIdx++) {
                        const ch = snap.retrieved_chunks[chIdx];
                        chunksSummary.push({
                            source: ch.source,
                            section: ch.section_title,
                            pages: ch.page_numbers
                        });
                    }
                }
                stepOut = {
                    loai_hanh_dong: attrs.action_kind || cur.nodeId,
                    ten_hanh_dong: attrs.action_name || "",
                    trang_thai: attrs.action_status || cur.status,
                    so_bang_chung_hien_co: attrs.evidence_count,
                    rag_ung_vien: retrievalCount,
                    rag_rerank: rerankCount,
                    parser: parserSummary,
                    tai_lieu_trich_dan: chunksSummary
                };
                let actTitle = attrs.action_name || cur.label;
                caption = "Thực thi \"" + actTitle + "\" (" + (attrs.action_status || cur.status) + ") → Thu được " + Number(attrs.evidence_count || 0) + " bằng chứng, trả về Deep Decide.";
            } else if (cur.nodeId === "answer_review") {
                let reviewSource = "deep_decide";
                if (prev) {
                    reviewSource = prev.nodeId + " (Bước " + prev.step + ")";
                }
                stepIn = {
                    tu_node: reviewSource,
                    truy_van: queryText,
                    du_thao_tra_loi: answerText,
                    model_kiem_toan: llmModels.join(", ") || "rule-based"
                };
                stepOut = {
                    ket_qua_kiem_toan: attrs.audit_status || cur.status,
                    so_lan_retry: attrs.retries,
                    nhan_xet_chi_tiet: attrs.review || {},
                    tokens_su_dung: stepTokens
                };
                if (cur.status === "RETRY") {
                    caption = "Answer Review phát hiện điểm cần bổ sung → Kích hoạt RETRY quay về Deep Decide.";
                } else {
                    caption = "Answer Review xác nhận câu trả lời chuẩn xác (" + cur.status + ") → Chuyển sang Output Guardrail.";
                }
            } else if (cur.nodeId === "output_guardrail") {
                let outSource = "answer_review";
                if (prev) {
                    outSource = prev.nodeId + " (Bước " + prev.step + ")";
                }
                let citeCount = Number(summary.citation_count || 0);
                if (Array.isArray(attrs.citation_refs)) {
                    citeCount = attrs.citation_refs.length;
                }
                stepIn = {
                    tu_node: outSource,
                    du_thao_tra_loi: answerText
                };
                stepOut = {
                    ket_luan_an_toan: attrs.is_safe,
                    so_trich_dan: citeCount,
                    cau_tra_loi_cuoi: answerText
                };
                caption = "Output Guardrail duyệt an toàn & gắn trích dẫn hoàn tất → Trả kết quả cho người dùng (END).";
            } else {
                let genericSource = "graph";
                if (prev) {
                    genericSource = prev.nodeId + " (Bước " + prev.step + ")";
                }
                stepIn = { tu_node: genericSource };
                stepOut = attrs;
            }

            cur.stepInput = stepIn;
            cur.stepOutput = stepOut;
            cur.caption = caption;
            cur.stepTokens = stepTokens;
            cur.llmModels = llmModels;

            if (nodeMap[cur.nodeId]) {
                if (nodeMap[cur.nodeId].steps.length <= 1) {
                    nodeMap[cur.nodeId].input_summary = stepIn;
                } else {
                    nodeMap[cur.nodeId].input_summary = {
                        luot_cuoi_buoc: cur.step,
                        chi_tiet_dau_vao: stepIn
                    };
                }
            }
        }
    }

    function resolveDebugDagTelemetry(detail, snap, graphNodes) {
        if (snap && snap.dag_telemetry && typeof snap.dag_telemetry === "object" && snap.dag_telemetry.stages) {
            return snap.dag_telemetry;
        }
        const stages = {};
        for (let i = 0; i < LANGGRAPH_CANONICAL_NODES.length; i++) {
            const def = LANGGRAPH_CANONICAL_NODES[i];
            stages[def.id] = {
                stage_id: def.id,
                label: def.label,
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

        for (const gn of graphNodes) {
            let sid = gn.name;
            if (sid && sid.startsWith("graph.")) {
                sid = sid.slice(6);
            }
            if (!sid || !stages[sid]) {
                continue;
            }
            const st = stages[sid];
            st.executed = true;
            st.call_count = st.call_count + 1;
            st.latency_ms = Number((st.latency_ms + Number(gn.duration_ms || 0)).toFixed(2));
            st.nodes.push(gn.name);
            const attrs = gn.attributes || {};
            if (attrs.audit_status) {
                st.status = String(attrs.audit_status);
            } else if (gn.status === "error") {
                st.status = "ERROR";
            } else if (gn.status === "suspended") {
                st.status = "AWAITING_INPUT";
            } else {
                st.status = "PASS";
            }
            st.output_summary = attrs;
        }

        const summary = detail.summary || {};
        const tok = summary.token_usage || {};
        let resolvedModel = "";
        if (Array.isArray(summary.models_used) && summary.models_used.length > 0) {
            resolvedModel = summary.models_used[0];
        }
        return {
            total_latency_ms: Number(detail.duration_ms || 0),
            active_model: resolvedModel,
            active_profile: "STANDARD",
            token_usage: {
                input_tokens: Number(tok.input_tokens || 0),
                output_tokens: Number(tok.output_tokens || 0),
                total_tokens: Number(tok.total_tokens || 0),
                estimated: false
            },
            audit_status: String(snap.audit_status || summary.audit_status || "PASS"),
            retry_loops: Number(snap.hallucination_retries || 0),
            stages: stages
        };
    }

    const LANGGRAPH_SVG_BOXES = {
        START: { x: 16, y: 236, w: 66, h: 36, terminal: true, title: "START", sub: "Khởi tạo" },
        input_guardrail: { x: 112, y: 218, w: 146, h: 72, title: "Input Guardrail", sub: "Kiểm duyệt vào" },
        context_prepare: { x: 288, y: 218, w: 146, h: 72, title: "Context Prepare", sub: "Ngữ cảnh & Memory" },
        deep_decide: { x: 468, y: 212, w: 168, h: 84, title: "Deep Decide", sub: "Coordinator Hub" },
        react_action: { x: 244, y: 26, w: 152, h: 64, title: "ReAct Action", sub: "Tools & RAG" },
        skill_inspect: { x: 414, y: 26, w: 152, h: 64, title: "Skill Inspect", sub: "Kỹ năng chuyên sâu" },
        plan_update: { x: 584, y: 26, w: 152, h: 64, title: "Plan Update", sub: "Kế hoạch ReAct" },
        subagent: { x: 754, y: 26, w: 152, h: 64, title: "Subagent", sub: "Ủy quyền nhánh" },
        memory_update: { x: 924, y: 26, w: 152, h: 64, title: "Memory Update", sub: "Bộ nhớ dài hạn" },
        answer_review: { x: 696, y: 218, w: 156, h: 72, title: "Answer Review", sub: "Kiểm toán & Audit" },
        output_guardrail: { x: 912, y: 218, w: 156, h: 72, title: "Output Guardrail", sub: "Kiểm duyệt ra" },
        END: { x: 1100, y: 236, w: 62, h: 36, terminal: true, title: "END", sub: "Hoàn tất" },
        prepare_interaction: { x: 540, y: 386, w: 160, h: 64, title: "Prepare Interact", sub: "Tạo câu hỏi HITL" },
        await_interaction: { x: 748, y: 386, w: 160, h: 64, title: "Await Interact", sub: "Đợi user trả lời" }
    };

    const LANGGRAPH_SVG_EDGES = [
        { from: "START", to: "input_guardrail", d: "M 82 254 L 112 254", label: "", lx: 0, ly: 0 },
        { from: "input_guardrail", to: "context_prepare", d: "M 258 254 L 288 254", label: "safe", lx: 273, ly: 245 },
        { from: "input_guardrail", to: "output_guardrail", d: "M 185 290 L 185 478 Q 185 494 202 494 L 1018 494 Q 1034 494 1034 478 L 1034 290", label: "blocked / fast-exit", lx: 580, ly: 486 },
        { from: "context_prepare", to: "deep_decide", d: "M 434 254 L 468 254", label: "", lx: 0, ly: 0 },
        { from: "deep_decide", to: "react_action", d: "M 572 212 L 572 152 Q 572 142 562 142 L 346 142 Q 336 142 336 132 L 336 90", label: "", lx: 0, ly: 0 },
        { from: "react_action", to: "deep_decide", d: "M 304 90 L 304 162 Q 304 172 314 172 L 522 172 Q 532 172 532 182 L 532 212", label: "", lx: 0, ly: 0 },
        { from: "deep_decide", to: "skill_inspect", d: "M 572 212 L 572 152 Q 572 142 562 142 L 516 142 Q 506 142 506 132 L 506 90", label: "", lx: 0, ly: 0 },
        { from: "skill_inspect", to: "deep_decide", d: "M 474 90 L 474 162 Q 474 172 484 172 L 522 172 Q 532 172 532 182 L 532 212", label: "", lx: 0, ly: 0 },
        { from: "deep_decide", to: "plan_update", d: "M 572 212 L 572 152 Q 572 142 582 142 L 666 142 Q 676 142 676 132 L 676 90", label: "", lx: 0, ly: 0 },
        { from: "plan_update", to: "deep_decide", d: "M 644 90 L 644 162 Q 644 172 634 172 L 542 172 Q 532 172 532 182 L 532 212", label: "", lx: 0, ly: 0 },
        { from: "deep_decide", to: "subagent", d: "M 572 212 L 572 152 Q 572 142 582 142 L 836 142 Q 846 142 846 132 L 846 90", label: "", lx: 0, ly: 0 },
        { from: "subagent", to: "deep_decide", d: "M 814 90 L 814 162 Q 814 172 804 172 L 542 172 Q 532 172 532 182 L 532 212", label: "", lx: 0, ly: 0 },
        { from: "deep_decide", to: "memory_update", d: "M 572 212 L 572 152 Q 572 142 582 142 L 1006 142 Q 1016 142 1016 132 L 1016 90", label: "", lx: 0, ly: 0 },
        { from: "memory_update", to: "deep_decide", d: "M 984 90 L 984 162 Q 984 172 974 172 L 542 172 Q 532 172 532 182 L 532 212", label: "", lx: 0, ly: 0 },
        { from: "deep_decide", to: "answer_review", d: "M 636 254 L 696 254", label: "review", lx: 666, ly: 245 },
        { from: "answer_review", to: "deep_decide", d: "M 744 290 C 744 322 600 322 600 296", label: "RETRY", lx: 672, ly: 314, kind: "retry" },
        { from: "deep_decide", to: "output_guardrail", d: "M 556 296 C 556 350 936 350 936 290", label: "direct", lx: 874, ly: 332 },
        { from: "answer_review", to: "output_guardrail", d: "M 852 254 L 912 254", label: "PASS", lx: 882, ly: 245 },
        { from: "deep_decide", to: "prepare_interaction", d: "M 498 296 C 498 374 504 418 540 418", label: "HITL", lx: 486, ly: 365, kind: "hitl" },
        { from: "prepare_interaction", to: "await_interaction", d: "M 700 418 L 748 418", label: "", lx: 0, ly: 0, kind: "hitl" },
        { from: "await_interaction", to: "deep_decide", d: "M 796 386 C 796 362 526 362 526 296", label: "resume", lx: 660, ly: 356, kind: "hitl" },
        { from: "await_interaction", to: "output_guardrail", d: "M 908 418 C 966 418 972 356 972 290", label: "final", lx: 984, ly: 355 },
        { from: "prepare_interaction", to: "output_guardrail", d: "M 620 450 C 620 474 1002 474 1002 290", label: "", lx: 0, ly: 0 },
        { from: "output_guardrail", to: "END", d: "M 1068 254 L 1100 254", label: "", lx: 0, ly: 0 }
    ];

    function computeExecutedEdgeMap(flow, detail, activeTransitionKey) {
        const edgeMap = {};
        function markEdge(fromId, toId, stepTag) {
            const key = fromId + "->" + toId;
            if (!edgeMap[key]) {
                edgeMap[key] = { active: true, isCurrent: false, steps: [] };
            }
            if (activeTransitionKey && key === activeTransitionKey) {
                edgeMap[key].isCurrent = true;
            }
            if (stepTag) {
                edgeMap[key].steps.push(stepTag);
            }
        }

        if (flow && Array.isArray(flow.trail) && flow.trail.length > 0) {
            markEdge("START", flow.trail[0].nodeId, "#1");
            for (let i = 0; i < flow.trail.length - 1; i++) {
                const cur = flow.trail[i];
                const nxt = flow.trail[i + 1];
                markEdge(cur.nodeId, nxt.nodeId, "#" + cur.step + "→" + nxt.step);
            }
            const lastNode = flow.trail[flow.trail.length - 1].nodeId;
            const outcome = String((detail && detail.outcome) || "").toLowerCase();
            if (!activeTransitionKey && lastNode === "output_guardrail" && outcome !== "running") {
                markEdge("output_guardrail", "END", "done");
            }
        }
        return edgeMap;
    }

    function sliceFlowForReplayStep(fullFlow, stepIdx) {
        if (!fullFlow || !Array.isArray(fullFlow.trail) || stepIdx < 0 || stepIdx >= fullFlow.trail.length) {
            return { flow: fullFlow, replayInfo: null };
        }
        const slicedNodes = {};
        const nodeKeys = Object.keys(fullFlow.nodes);
        for (let i = 0; i < nodeKeys.length; i++) {
            const nid = nodeKeys[i];
            const orig = fullFlow.nodes[nid];
            slicedNodes[nid] = {
                id: orig.id,
                label: orig.label,
                groupId: orig.groupId,
                role: orig.role,
                desc: orig.desc,
                executed: false,
                call_count: 0,
                latency_ms: 0,
                status: "IDLE",
                steps: [],
                executions: [],
                input_summary: {},
                output_summary: {}
            };
        }

        const activeTrail = [];
        for (let i = 0; i <= stepIdx; i++) {
            const item = fullFlow.trail[i];
            activeTrail.push(item);
            const target = slicedNodes[item.nodeId];
            if (target) {
                target.executed = true;
                target.call_count = target.call_count + 1;
                target.latency_ms = Number((target.latency_ms + Number(item.duration_ms || 0)).toFixed(2));
                target.steps.push(item.step);
                target.status = item.status;
                target.input_summary = item.stepInput || {};
                target.output_summary = item.stepOutput || item.attributes || {};
                target.executions.push(item);
            }
        }

        let executedCount = 0;
        for (let i = 0; i < nodeKeys.length; i++) {
            if (slicedNodes[nodeKeys[i]].executed) {
                executedCount = executedCount + 1;
            }
        }

        const curItem = fullFlow.trail[stepIdx];
        let prevNodeId = "START";
        let prevLabel = "START";
        if (stepIdx > 0) {
            prevNodeId = fullFlow.trail[stepIdx - 1].nodeId;
            prevLabel = fullFlow.trail[stepIdx - 1].label;
        }

        return {
            flow: {
                nodes: slicedNodes,
                trail: activeTrail,
                executedCount: executedCount,
                totalCount: fullFlow.totalCount
            },
            replayInfo: {
                stepIndex: stepIdx,
                totalSteps: fullFlow.trail.length,
                stepItem: curItem,
                prevNodeId: prevNodeId,
                prevLabel: prevLabel,
                currentNodeId: curItem.nodeId,
                activeEdgeKey: prevNodeId + "->" + curItem.nodeId
            }
        };
    }

    function buildLangGraphSvgDiagram(flow, selectedState, detail, onSelectNode, replayInfo) {
        let activeTransitionKey = "";
        let currentReplayNodeId = "";
        if (replayInfo) {
            activeTransitionKey = replayInfo.activeEdgeKey || "";
            currentReplayNodeId = replayInfo.currentNodeId || "";
        }
        const edgeMap = computeExecutedEdgeMap(flow, detail, activeTransitionKey);
        const scrollWrap = el("div", "langgraph-svg-scroll");

        const svg = svgEl("svg", {
            viewBox: "0 0 1180 512",
            class: "langgraph-svg-canvas",
            role: "img",
            "aria-label": "Biểu đồ luồng đầy đủ LangGraph và đường chạy thực tế của bot"
        });

        const defs = svgEl("defs");

        function makeMarker(id, color) {
            const m = svgEl("marker", {
                id: id,
                viewBox: "0 0 10 10",
                refX: "8.5",
                refY: "5",
                markerWidth: "6",
                markerHeight: "6",
                orient: "auto-start-reverse"
            });
            m.appendChild(svgEl("path", { d: "M 0 1.2 L 9 5 L 0 8.8 z", fill: color }));
            return m;
        }
        defs.appendChild(makeMarker("lgArrowIdle", "#475569"));
        defs.appendChild(makeMarker("lgArrowPass", "#22c55e"));
        defs.appendChild(makeMarker("lgArrowWarn", "#f59e0b"));
        defs.appendChild(makeMarker("lgArrowDyn", "#38bdf8"));

        const glowFilter = svgEl("filter", { id: "lgGlowGreen", x: "-20%", y: "-20%", width: "140%", height: "140%" });
        glowFilter.appendChild(svgEl("feDropShadow", { dx: "0", dy: "0", stdDeviation: "3.5", "flood-color": "#22c55e", "flood-opacity": "0.55" }));
        defs.appendChild(glowFilter);

        const glowAmber = svgEl("filter", { id: "lgGlowAmber", x: "-20%", y: "-20%", width: "140%", height: "140%" });
        glowAmber.appendChild(svgEl("feDropShadow", { dx: "0", dy: "0", stdDeviation: "3.5", "flood-color": "#f59e0b", "flood-opacity": "0.55" }));
        defs.appendChild(glowAmber);

        const glowCyan = svgEl("filter", { id: "lgGlowCyan", x: "-25%", y: "-25%", width: "150%", height: "150%" });
        glowCyan.appendChild(svgEl("feDropShadow", { dx: "0", dy: "0", stdDeviation: "5", "flood-color": "#38bdf8", "flood-opacity": "0.8" }));
        defs.appendChild(glowCyan);

        svg.appendChild(defs);

        // Background architectural zones
        const zonesLayer = svgEl("g", { class: "lg-svg-zones" });
        zonesLayer.appendChild(svgEl("rect", {
            x: "230", y: "6", width: "860", height: "94", rx: "10",
            fill: "rgba(30, 41, 59, 0.35)", stroke: "rgba(56, 189, 248, 0.22)", "stroke-dasharray": "6 4"
        }));
        zonesLayer.appendChild(svgEl("text", {
            x: "244", y: "20", fill: "#64748b", "font-size": "9.5", "font-weight": "700", "letter-spacing": "0.06em"
        }, "VÒNG LẶP HÀNH ĐỘNG & CÔNG CỤ (ReAct ACTIONS LOOP ↔ DEEP DECIDE)"));

        zonesLayer.appendChild(svgEl("rect", {
            x: "524", y: "366", width: "400", height: "96", rx: "10",
            fill: "rgba(30, 41, 59, 0.35)", stroke: "rgba(251, 191, 36, 0.22)", "stroke-dasharray": "6 4"
        }));
        zonesLayer.appendChild(svgEl("text", {
            x: "538", y: "380", fill: "#64748b", "font-size": "9.5", "font-weight": "700", "letter-spacing": "0.06em"
        }, "TƯƠNG TÁC NGƯỜI DÙNG (HUMAN-IN-THE-LOOP / HITL)"));
        svg.appendChild(zonesLayer);

        // Render edges: idle edges first, active edges on top
        const idleEdgeGroup = svgEl("g", { class: "lg-edges-idle" });
        const activeEdgeGroup = svgEl("g", { class: "lg-edges-active" });
        const renderedKeys = {};

        for (let i = 0; i < LANGGRAPH_SVG_EDGES.length; i++) {
            const edge = LANGGRAPH_SVG_EDGES[i];
            const key = edge.from + "->" + edge.to;
            renderedKeys[key] = true;
            const execInfo = edgeMap[key];
            const isActive = Boolean(execInfo && execInfo.active);
            const isCurrentFrame = Boolean(execInfo && execInfo.isCurrent);

            if (isActive) {
                let strokeCol = "#22c55e";
                let markerUrl = "url(#lgArrowPass)";
                let strokeW = "2.6";
                let edgeClass = "lg-svg-edge-active-flow";
                if (edge.kind === "retry" || edge.kind === "hitl") {
                    strokeCol = "#f59e0b";
                    markerUrl = "url(#lgArrowWarn)";
                }
                let haloW = "6";
                let haloOpacity = "0.18";
                if (isCurrentFrame) {
                    strokeCol = "#38bdf8";
                    markerUrl = "url(#lgArrowDyn)";
                    strokeW = "3.4";
                    edgeClass = "lg-svg-edge-current-flow";
                    haloW = "8";
                    haloOpacity = "0.28";
                }
                activeEdgeGroup.appendChild(svgEl("path", {
                    d: edge.d,
                    fill: "none",
                    stroke: strokeCol,
                    "stroke-width": haloW,
                    opacity: haloOpacity,
                    "stroke-linecap": "round"
                }));
                activeEdgeGroup.appendChild(svgEl("path", {
                    d: edge.d,
                    fill: "none",
                    stroke: strokeCol,
                    "stroke-width": strokeW,
                    "marker-end": markerUrl,
                    class: edgeClass
                }));
                if (edge.label) {
                    activeEdgeGroup.appendChild(svgEl("text", {
                        x: edge.lx,
                        y: edge.ly,
                        "text-anchor": "middle",
                        fill: strokeCol,
                        "font-size": "9.5",
                        "font-weight": "700",
                        class: "lg-edge-label"
                    }, edge.label));
                }
            } else {
                idleEdgeGroup.appendChild(svgEl("path", {
                    d: edge.d,
                    fill: "none",
                    stroke: "#334155",
                    "stroke-width": "1.35",
                    "stroke-dasharray": "4 3",
                    opacity: "0.7",
                    "marker-end": "url(#lgArrowIdle)"
                }));
                if (edge.label) {
                    idleEdgeGroup.appendChild(svgEl("text", {
                        x: edge.lx,
                        y: edge.ly,
                        "text-anchor": "middle",
                        fill: "#475569",
                        "font-size": "8.8",
                        "font-weight": "600"
                    }, edge.label));
                }
            }
        }

        // Draw dynamic transition curves for any edge executed at runtime not covered statically
        const edgeKeys = Object.keys(edgeMap);
        for (let i = 0; i < edgeKeys.length; i++) {
            const ek = edgeKeys[i];
            if (renderedKeys[ek]) {
                continue;
            }
            const parts = ek.split("->");
            const bFrom = LANGGRAPH_SVG_BOXES[parts[0]];
            const bTo = LANGGRAPH_SVG_BOXES[parts[1]];
            if (bFrom && bTo) {
                const x1 = bFrom.x + bFrom.w / 2;
                const y1 = bFrom.y + bFrom.h / 2;
                const x2 = bTo.x + bTo.w / 2;
                const y2 = bTo.y + bTo.h / 2;
                const mx = (x1 + x2) / 2;
                const my = Math.min(y1, y2) - 42;
                activeEdgeGroup.appendChild(svgEl("path", {
                    d: "M " + x1 + " " + y1 + " Q " + mx + " " + my + " " + x2 + " " + y2,
                    fill: "none",
                    stroke: "#38bdf8",
                    "stroke-width": "2.4",
                    "stroke-dasharray": "5 3",
                    "marker-end": "url(#lgArrowDyn)"
                }));
            }
        }

        svg.appendChild(idleEdgeGroup);
        svg.appendChild(activeEdgeGroup);

        // Render Nodes
        const nodesLayer = svgEl("g", { class: "lg-nodes-layer" });
        const nodeIds = Object.keys(LANGGRAPH_SVG_BOXES);

        for (let i = 0; i < nodeIds.length; i++) {
            const nid = nodeIds[i];
            const box = LANGGRAPH_SVG_BOXES[nid];

            if (box.terminal) {
                let isTermActive = false;
                if (nid === "START" && flow.trail.length > 0) {
                    isTermActive = true;
                }
                if (nid === "END" && flow.nodes.output_guardrail && flow.nodes.output_guardrail.executed && !replayInfo) {
                    isTermActive = true;
                }
                if (nid === "END" && replayInfo && replayInfo.currentNodeId === "output_guardrail" && replayInfo.stepIndex === replayInfo.totalSteps - 1) {
                    isTermActive = true;
                }
                let termStroke = "#475569";
                let termFill = "#0f172a";
                let termTextCol = "#94a3b8";
                let termWidth = "1.2";
                if (isTermActive) {
                    termStroke = "#22c55e";
                    termFill = "#052e16";
                    termTextCol = "#86efac";
                    termWidth = "2";
                }
                const gTerm = svgEl("g", { class: "lg-svg-terminal" });
                gTerm.appendChild(svgEl("rect", {
                    x: box.x, y: box.y, width: box.w, height: box.h, rx: "18",
                    fill: termFill, stroke: termStroke, "stroke-width": termWidth
                }));
                gTerm.appendChild(svgEl("text", {
                    x: box.x + box.w / 2, y: box.y + 22,
                    "text-anchor": "middle", fill: termTextCol,
                    "font-size": "11", "font-weight": "800", "letter-spacing": "0.05em"
                }, box.title));
                nodesLayer.appendChild(gTerm);
                continue;
            }

            const nState = flow.nodes[nid] || { executed: false, status: "IDLE", steps: [], latency_ms: 0, call_count: 0 };
            const isReplayCurrent = currentReplayNodeId === nid;
            const isSelected = selectedState.id === nid || isReplayCurrent;

            let bgFill = "rgba(15, 23, 42, 0.72)";
            let borderCol = "#334155";
            let borderDash = "4 3";
            let borderWidth = "1.3";
            let titleCol = "#64748b";
            let subCol = "#475569";
            let badgeBg = "rgba(51, 65, 85, 0.35)";
            let badgeTextCol = "#64748b";
            let statusBadgeTxt = "— Chưa chạy";
            let filterAttr = "";

            if (replayInfo && !nState.executed) {
                statusBadgeTxt = "… Chờ tới";
            }

            if (nState.executed) {
                borderDash = "";
                borderWidth = "2";
                titleCol = "#f8fafc";
                subCol = "#cbd5e1";

                const stUpper = String(nState.status || "PASS").toUpperCase();
                if (stUpper === "RETRY" || stUpper === "AWAITING_INPUT" || stUpper === "NEEDS_INPUT") {
                    bgFill = "#1f1606";
                    borderCol = "#f59e0b";
                    badgeBg = "rgba(245, 158, 11, 0.22)";
                    badgeTextCol = "#fcd34d";
                    statusBadgeTxt = "↻ " + stUpper;
                    filterAttr = "url(#lgGlowAmber)";
                } else if (stUpper === "ERROR" || stUpper === "FAIL" || stUpper === "BLOCKED") {
                    bgFill = "#2a0e14";
                    borderCol = "#ef4444";
                    badgeBg = "rgba(239, 68, 68, 0.24)";
                    badgeTextCol = "#fca5a5";
                    statusBadgeTxt = "✕ " + stUpper;
                } else {
                    bgFill = "#06251c";
                    borderCol = "#22c55e";
                    badgeBg = "rgba(34, 197, 94, 0.22)";
                    badgeTextCol = "#86efac";
                    statusBadgeTxt = "✓ " + stUpper;
                    filterAttr = "url(#lgGlowGreen)";
                }
            }

            if (isReplayCurrent) {
                bgFill = "#082f49";
                borderCol = "#38bdf8";
                borderWidth = "3";
                filterAttr = "url(#lgGlowCyan)";
            } else if (isSelected) {
                borderCol = "#38bdf8";
                borderWidth = "2.8";
            }

            let gNodeClass = "lg-svg-node is-idle";
            if (nState.executed) {
                gNodeClass = "lg-svg-node is-executed";
            }
            if (isReplayCurrent) {
                gNodeClass = gNodeClass + " is-replay-current";
            }

            const gNode = svgEl("g", {
                class: gNodeClass,
                role: "button",
                tabindex: "0",
                "aria-label": box.title + " - " + statusBadgeTxt
            });

            const rectAttrs = {
                x: box.x,
                y: box.y,
                width: box.w,
                height: box.h,
                rx: "10",
                fill: bgFill,
                stroke: borderCol,
                "stroke-width": borderWidth
            };
            if (borderDash) {
                rectAttrs["stroke-dasharray"] = borderDash;
            }
            if (filterAttr && (!isSelected || isReplayCurrent)) {
                rectAttrs.filter = filterAttr;
            }
            gNode.appendChild(svgEl("rect", rectAttrs));

            if (nState.executed) {
                gNode.appendChild(svgEl("rect", {
                    x: box.x + 1,
                    y: box.y + 8,
                    width: "4",
                    height: box.h - 16,
                    rx: "2",
                    fill: borderCol
                }));
            }

            gNode.appendChild(svgEl("text", {
                x: box.x + 11,
                y: box.y + 20,
                fill: titleCol,
                "font-size": "11.5",
                "font-weight": "700"
            }, box.title));

            if (nState.executed && nState.steps && nState.steps.length > 0) {
                let stepLabel = "#" + nState.steps.join(",");
                if (isReplayCurrent && replayInfo && replayInfo.stepItem) {
                    stepLabel = "▶ #" + replayInfo.stepItem.step;
                } else if (stepLabel.length > 9) {
                    stepLabel = "#" + nState.steps[0] + " (" + nState.call_count + "x)";
                }
                const pillW = Math.max(26, stepLabel.length * 6.8 + 10);
                let pillFill = "rgba(56, 189, 248, 0.2)";
                let pillStroke = "rgba(56, 189, 248, 0.5)";
                let pillTxtCol = "#7dd3fc";
                if (isReplayCurrent) {
                    pillFill = "#0284c7";
                    pillStroke = "#7dd3fc";
                    pillTxtCol = "#ffffff";
                }
                gNode.appendChild(svgEl("rect", {
                    x: box.x + box.w - pillW - 6,
                    y: box.y + 6,
                    width: pillW,
                    height: "16",
                    rx: "8",
                    fill: pillFill,
                    stroke: pillStroke,
                    "stroke-width": "0.8"
                }));
                gNode.appendChild(svgEl("text", {
                    x: box.x + box.w - pillW / 2 - 6,
                    y: box.y + 17,
                    "text-anchor": "middle",
                    fill: pillTxtCol,
                    "font-size": "9",
                    "font-weight": "700"
                }, stepLabel));
            }

            gNode.appendChild(svgEl("text", {
                x: box.x + 11,
                y: box.y + 35,
                fill: subCol,
                "font-size": "9.5",
                "font-weight": "500"
            }, box.sub));

            const pillWBottom = Math.min(box.w - 22, 82);
            gNode.appendChild(svgEl("rect", {
                x: box.x + 10,
                y: box.y + box.h - 22,
                width: pillWBottom,
                height: "15",
                rx: "4",
                fill: badgeBg
            }));
            gNode.appendChild(svgEl("text", {
                x: box.x + 15,
                y: box.y + box.h - 11.5,
                fill: badgeTextCol,
                "font-size": "8.8",
                "font-weight": "700"
            }, statusBadgeTxt));

            if (nState.executed) {
                const durTxt = formatShortMs(Number(nState.latency_ms || 0));
                gNode.appendChild(svgEl("text", {
                    x: box.x + box.w - 8,
                    y: box.y + box.h - 11.5,
                    "text-anchor": "end",
                    fill: "#94a3b8",
                    "font-size": "9",
                    "font-family": "ui-monospace, monospace",
                    "font-weight": "600"
                }, durTxt));
            }

            gNode.addEventListener("click", () => {
                onSelectNode(nid);
            });
            gNode.addEventListener("keydown", (ev) => {
                if (ev.key === "Enter" || ev.key === " ") {
                    ev.preventDefault();
                    onSelectNode(nid);
                }
            });

            nodesLayer.appendChild(gNode);
        }

        svg.appendChild(nodesLayer);
        scrollWrap.appendChild(svg);
        return scrollWrap;
    }

    function stopReplayPlayback() {
        if (state.replayTimer !== null) {
            window.clearInterval(state.replayTimer);
            state.replayTimer = null;
        }
        state.replayPlaying = false;
    }

    function renderDagIntoContainer(dag, track, inspector, selectedState, detail) {
        if (!track || !inspector) {
            return;
        }
        const flow = resolveLangGraphExecutionState(detail, dag);
        const traceId = String((detail && detail.trace_id) || "");
        if (state.replayTraceId !== traceId) {
            stopReplayPlayback();
            state.replayTraceId = traceId;
            state.replayStepIndex = -1;
        }
        if (state.replayStepIndex >= flow.trail.length) {
            state.replayStepIndex = flow.trail.length - 1;
        }

        function getActiveReplayStepItem() {
            if (state.replayStepIndex >= 0 && state.replayStepIndex < flow.trail.length) {
                return flow.trail[state.replayStepIndex];
            }
            return null;
        }

        function renderInspectorBox(nodeId, explicitStepItem) {
            let targetNodeId = nodeId;
            if (explicitStepItem && explicitStepItem.nodeId) {
                targetNodeId = explicitStepItem.nodeId;
            }
            if (!targetNodeId || !flow.nodes[targetNodeId]) {
                inspector.hidden = true;
                inspector.replaceChildren();
                return;
            }
            const node = flow.nodes[targetNodeId];
            inspector.hidden = false;
            inspector.replaceChildren();

            const hdr = el("div", "dag-inspector-header");
            const titleGroup = el("div", "dag-inspector-title");
            let headerTitle = node.label;
            let statusLabel = "CHƯA CHẠY";
            let latMs = Number(node.latency_ms || 0);
            let inputObj = node.input_summary || {};
            let outObj = { ...(node.output_summary || {}) };
            let stepsTxt = "Không";
            let callsTxt = String(Number(node.call_count || 0));

            if (explicitStepItem) {
                headerTitle = node.label + " · Bước #" + explicitStepItem.step + "/" + flow.trail.length;
                statusLabel = String(explicitStepItem.status || "PASS");
                latMs = Number(explicitStepItem.duration_ms || 0);
                inputObj = explicitStepItem.stepInput || {};
                outObj = { ...(explicitStepItem.stepOutput || explicitStepItem.attributes || {}) };
                stepsTxt = "Bước " + explicitStepItem.step;
                callsTxt = "Lượt tại frame này";
            } else {
                if (node.executed) {
                    statusLabel = String(node.status || "PASS");
                }
                if (node.steps.length > 0) {
                    stepsTxt = "Bước " + node.steps.join(", ");
                }
                if (Array.isArray(node.executions) && node.executions.length > 0) {
                    outObj.executions = node.executions;
                }
                if (!node.executed) {
                    outObj.ghi_chu = "Node này thuộc sơ đồ chuẩn của LangGraph nhưng không phát sinh thực thi trong lượt trace này.";
                }
            }

            titleGroup.append(
                el("span", "", headerTitle),
                el("span", "dag-node-status", statusLabel)
            );

            let execChip = null;
            if (explicitStepItem || node.executed) {
                execChip = el("span", "dag-node-profile is-standard", "✓ ĐÃ CHẠY");
            } else {
                execChip = el("span", "dag-node-profile is-rule", "— BỎ QUA");
            }
            titleGroup.appendChild(execChip);

            const closeBtn = document.createElement("button");
            closeBtn.type = "button";
            closeBtn.className = "dag-inspector-close";
            closeBtn.textContent = "×";
            closeBtn.setAttribute("aria-label", "Đóng chi tiết node");
            closeBtn.addEventListener("click", () => {
                selectedState.id = null;
                if (state.replayStepIndex >= 0) {
                    stopReplayPlayback();
                    state.replayStepIndex = -1;
                }
                renderFlowView();
                renderInspectorBox(null, null);
            });
            hdr.appendChild(titleGroup);
            const hdrActions = el("div", "dag-inspector-actions");
            const copyFullNodeBtn = buildCopyIconBtn(() => {
                return JSON.stringify({
                    node: targetNodeId,
                    status: statusLabel,
                    duration_ms: latMs,
                    input: inputObj,
                    output: outObj
                }, null, 2);
            }, "Copy JSON Input & Output");
            hdrActions.append(copyFullNodeBtn, closeBtn);
            hdr.appendChild(hdrActions);
            inspector.appendChild(hdr);

            let descText = node.desc || node.role;
            if (explicitStepItem && explicitStepItem.caption) {
                descText = explicitStepItem.caption;
            }
            const descBox = el("div", "dag-inspector-desc", descText);
            inspector.appendChild(descBox);

            const stats = el("div", "dag-inspector-stats");
            let latTxt = "—";
            if (explicitStepItem || node.executed) {
                latTxt = formatShortMs(latMs);
            }
            stats.append(
                el("span", "dag-inspector-stat", "Thời gian: " + latTxt),
                el("span", "dag-inspector-stat", "Số lượt chạy: " + callsTxt),
                el("span", "dag-inspector-stat", "Thứ tự: " + stepsTxt),
                el("span", "dag-inspector-stat", "Nghiệp vụ: " + node.role)
            );
            inspector.appendChild(stats);

            const ioGrid = el("div", "dag-io-grid");
            const inJsonStr = JSON.stringify(inputObj, null, 2);
            const inBox = el("div", "dag-io-box");
            const inHdr = el("div", "dag-io-header");
            inHdr.append(
                el("div", "dag-io-label", "Input"),
                buildCopyIconBtn(() => { return inJsonStr; }, "Copy Input JSON")
            );
            inBox.append(inHdr, el("pre", "dag-io-pre", inJsonStr));

            const outJsonStr = JSON.stringify(outObj, null, 2);
            const outBox = el("div", "dag-io-box");
            const outHdr = el("div", "dag-io-header");
            outHdr.append(
                el("div", "dag-io-label", "Output"),
                buildCopyIconBtn(() => { return outJsonStr; }, "Copy Output JSON")
            );
            outBox.append(outHdr, el("pre", "dag-io-pre", outJsonStr));

            ioGrid.append(inBox, outBox);
            inspector.appendChild(ioGrid);
        }

        function syncInspectorWithState() {
            const activeItem = getActiveReplayStepItem();
            if (activeItem) {
                selectedState.id = activeItem.nodeId;
                renderInspectorBox(activeItem.nodeId, activeItem);
            } else {
                renderInspectorBox(selectedState.id, null);
            }
        }

        function jumpToReplayStep(nextIdx, autoOpenInspector) {
            if (nextIdx < 0) {
                stopReplayPlayback();
                state.replayStepIndex = -1;
                renderFlowView();
                renderInspectorBox(selectedState.id, null);
                return;
            }
            let clamped = nextIdx;
            if (clamped >= flow.trail.length) {
                clamped = flow.trail.length - 1;
            }
            state.replayStepIndex = clamped;
            renderFlowView();
            syncInspectorWithState();
            if (autoOpenInspector && inspector && !inspector.hidden) {
                inspector.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }
        }

        function startReplayPlayback() {
            if (flow.trail.length === 0) {
                return;
            }
            stopReplayPlayback();
            state.replayPlaying = true;
            if (state.replayStepIndex < 0 || state.replayStepIndex >= flow.trail.length - 1) {
                state.replayStepIndex = 0;
            }
            renderFlowView();
            syncInspectorWithState();
            state.replayTimer = window.setInterval(() => {
                if (state.replayStepIndex + 1 >= flow.trail.length) {
                    stopReplayPlayback();
                    renderFlowView();
                    return;
                }
                state.replayStepIndex = state.replayStepIndex + 1;
                renderFlowView();
                syncInspectorWithState();
            }, state.replaySpeedMs);
        }

        function handleNodeSelect(nodeId) {
            stopReplayPlayback();
            state.replayStepIndex = -1;
            if (selectedState.id === nodeId) {
                selectedState.id = null;
            } else {
                selectedState.id = nodeId;
            }
            renderFlowView();
            renderInspectorBox(selectedState.id, null);
            if (selectedState.id && inspector && !inspector.hidden) {
                inspector.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }
        }

        function renderFlowView() {
            track.replaceChildren();

            const flowContainer = el("div", "langgraph-flow-container");
            const sliced = sliceFlowForReplayStep(flow, state.replayStepIndex);
            const activeFlow = sliced.flow;
            const replayInfo = sliced.replayInfo;

            // 1. Video-style Trace Replay Player Bar
            if (flow.trail.length > 0) {
                const replayBar = el("div", "lg-replay-bar");
                const btnGroup = el("div", "lg-replay-controls");

                let playLabel = "▶ Phát lại luồng";
                if (state.replayPlaying) {
                    playLabel = "⏸ Tạm dừng";
                }
                const playBtn = el("button", "lg-replay-btn is-primary", playLabel);
                playBtn.type = "button";
                if (state.replayPlaying) {
                    playBtn.classList.add("is-playing");
                }
                playBtn.addEventListener("click", () => {
                    if (state.replayPlaying) {
                        stopReplayPlayback();
                        renderFlowView();
                    } else {
                        startReplayPlayback();
                    }
                });

                const firstBtn = el("button", "lg-replay-btn", "⏮");
                firstBtn.type = "button";
                firstBtn.title = "Về Bước 1";
                firstBtn.addEventListener("click", () => {
                    stopReplayPlayback();
                    jumpToReplayStep(0, true);
                });

                const prevBtn = el("button", "lg-replay-btn", "◀");
                prevBtn.type = "button";
                prevBtn.title = "Bước trước";
                prevBtn.addEventListener("click", () => {
                    stopReplayPlayback();
                    if (state.replayStepIndex <= 0) {
                        jumpToReplayStep(0, true);
                    } else {
                        jumpToReplayStep(state.replayStepIndex - 1, true);
                    }
                });

                const nextBtn = el("button", "lg-replay-btn", "▶|");
                nextBtn.type = "button";
                nextBtn.title = "Bước tiếp theo";
                nextBtn.addEventListener("click", () => {
                    stopReplayPlayback();
                    if (state.replayStepIndex < 0) {
                        jumpToReplayStep(0, true);
                    } else {
                        jumpToReplayStep(state.replayStepIndex + 1, true);
                    }
                });

                const fullViewBtn = el("button", "lg-replay-btn", "⏹ Toàn cảnh");
                fullViewBtn.type = "button";
                fullViewBtn.title = "Thoát chế độ phát lại, xem toàn bộ luồng";
                if (state.replayStepIndex < 0) {
                    fullViewBtn.classList.add("is-active");
                }
                fullViewBtn.addEventListener("click", () => {
                    jumpToReplayStep(-1, false);
                });

                btnGroup.append(playBtn, firstBtn, prevBtn, nextBtn, fullViewBtn);

                const scrubberWrap = el("div", "lg-replay-scrubber-wrap");
                const scrubber = document.createElement("input");
                scrubber.type = "range";
                scrubber.className = "lg-replay-scrubber";
                scrubber.min = "0";
                scrubber.max = String(flow.trail.length);
                if (state.replayStepIndex < 0) {
                    scrubber.value = "0";
                } else {
                    scrubber.value = String(state.replayStepIndex + 1);
                }
                scrubber.setAttribute("aria-label", "Thanh tua các bước LangGraph");
                scrubber.addEventListener("input", () => {
                    stopReplayPlayback();
                    const val = Number(scrubber.value);
                    if (val <= 0) {
                        jumpToReplayStep(-1, false);
                    } else {
                        jumpToReplayStep(val - 1, false);
                    }
                });

                let stepStatusTxt = "Toàn cảnh · " + flow.trail.length + "b";
                if (replayInfo) {
                    stepStatusTxt = "Bước " + (replayInfo.stepIndex + 1) + "/" + replayInfo.totalSteps;
                }
                const framePill = el("span", "lg-replay-frame-pill", stepStatusTxt);
                scrubberWrap.append(scrubber, framePill);

                const speedGroup = el("div", "lg-replay-speeds");
                const speeds = [
                    { ms: 2400, label: "0.5x" },
                    { ms: 1400, label: "1x" },
                    { ms: 700, label: "2x" }
                ];
                for (let sIdx = 0; sIdx < speeds.length; sIdx++) {
                    const sp = speeds[sIdx];
                    const sBtn = el("button", "lg-replay-speed-btn", sp.label);
                    sBtn.type = "button";
                    if (state.replaySpeedMs === sp.ms) {
                        sBtn.classList.add("is-active");
                    }
                    sBtn.addEventListener("click", () => {
                        state.replaySpeedMs = sp.ms;
                        if (state.replayPlaying) {
                            startReplayPlayback();
                        } else {
                            renderFlowView();
                        }
                    });
                    speedGroup.appendChild(sBtn);
                }

                replayBar.append(btnGroup, scrubberWrap, speedGroup);
                flowContainer.appendChild(replayBar);
            }

            // 2. LangGraph SVG Flowchart
            const svgGraph = buildLangGraphSvgDiagram(activeFlow, selectedState, detail, handleNodeSelect, replayInfo);
            flowContainer.appendChild(svgGraph);

            track.appendChild(flowContainer);
        }

        renderFlowView();
        syncInspectorWithState();
    }

    function buildDebugDagVisualizer(detail, snap, graphNodes) {
        const dag = resolveDebugDagTelemetry(detail, snap, graphNodes);
        const section = el("div", "debug-dag-section");

        const summaryRow = el("div", "debug-section-summary");
        const tok = dag.token_usage || {};
        let modelText = "—";
        if (dag.active_model) {
            modelText = dag.active_model;
        }
        let tokLabel = String(Number(tok.total_tokens || 0)) + " tok";
        if (tok.estimated && Number(tok.total_tokens || 0) > 0) {
            tokLabel = "~" + tokLabel;
        }
        summaryRow.append(
            el("span", "debug-summary-chip", "Latency: " + formatShortMs(Number(dag.total_latency_ms || 0))),
            el("span", "debug-summary-chip", "Profile: " + String(dag.active_profile || "STANDARD")),
            el("span", "debug-summary-chip", "Model: " + modelText),
            el("span", "debug-summary-chip", "Tokens: " + tokLabel),
            el("span", "debug-summary-chip", "Retry Loops: " + String(dag.retry_loops || 0))
        );
        section.appendChild(summaryRow);

        const track = el("div", "dag-graph-track");
        const inspector = el("div", "dag-node-inspector");
        inspector.hidden = true;
        renderDagIntoContainer(dag, track, inspector, { id: null }, detail);
        section.append(track, inspector);
        return section;
    }

    function renderHeaderDagVisualizer(detail) {
        if (!detail) {
            if (dom.dagGraphTrack) {
                dom.dagGraphTrack.replaceChildren();
            }
            if (dom.dagNodeInspector) {
                dom.dagNodeInspector.hidden = true;
                dom.dagNodeInspector.replaceChildren();
            }
            return;
        }

        const snap = detail.turn_snapshot || {};
        let graphNodes = [];
        if (Array.isArray(detail.graph_nodes)) {
            graphNodes = detail.graph_nodes;
        }
        const dag = resolveDebugDagTelemetry(detail, snap, graphNodes);

        renderDagIntoContainer(dag, dom.dagGraphTrack, dom.dagNodeInspector, { id: null }, detail);
    }

    function renderAgentTab(detail) {
        const panel = dom.panelAgentInner;
        if (!panel) {
            return;
        }
        panel.replaceChildren();

        const snap = detail.turn_snapshot || {};
        let tools = [];
        if (Array.isArray(detail.tool_calls)) {
            tools = detail.tool_calls;
        }
        let graphNodes = [];
        if (Array.isArray(detail.graph_nodes)) {
            graphNodes = detail.graph_nodes;
        }

        // 1. Guardrails & Response Profile
        panel.appendChild(el("h4", "debug-section-heading", "1. Guardrails, Persona & Audit Status"));
        const inGuard = snap.input_guard || {};
        const outGuard = snap.output_guard || {};
        const verification = snap.verification || {};
        const guardCard = el("div", "debug-detail-card");
        guardCard.appendChild(buildKeyValueTable([
            ["User Query", snap.query],
            ["Persona / Tone / Format", [snap.persona, snap.tone, snap.format_style].filter(Boolean).join(" · ")],
            ["Intent", snap.intent || (detail.summary && detail.summary.intent)],
            ["Input Guardrail", "is_safe=" + String(inGuard.is_safe) + " · flags=" + JSON.stringify(inGuard.flags || [])],
            ["Output Guardrail", "is_safe=" + String(outGuard.is_safe) + " · flags=" + JSON.stringify(outGuard.flags || [])],
            ["Audit Status", snap.audit_status || (detail.summary && detail.summary.audit_status)],
            ["Hallucination Retries / Search Retries", (snap.hallucination_retries || 0) + " / " + (snap.retry_count || 0)],
            ["Review Outcomes", renderPillsGroup(snap.review_outcomes)],
            ["Verification", verification],
            ["Audit Feedback", snap.audit_feedback],
            ["Error Category", snap.error_category]
        ]));
        panel.appendChild(guardCard);

        // 2. Plan & Steps
        let plan = [];
        if (Array.isArray(snap.plan)) {
            plan = snap.plan;
        }
        let steps = [];
        if (Array.isArray(snap.execution_steps)) {
            steps = snap.execution_steps;
        }
        if (plan.length > 0 || steps.length > 0) {
            panel.appendChild(el("h4", "debug-section-heading", "2. Kế hoạch Agent (Rev " + (snap.plan_revision || 0) + ") & Execution Steps"));
            const planCard = el("div", "debug-detail-card");
            if (plan.length > 0) {
                const ol = document.createElement("ol");
                ol.className = "debug-plan-list";
                for (const item of plan) {
                    const li = document.createElement("li");
                    li.append(
                        el("span", "debug-status-badge " + statusBadgeClass(item.status), item.status || "pending"),
                        el("span", "", " " + (item.task || ""))
                    );
                    ol.appendChild(li);
                }
                planCard.appendChild(ol);
            }
            if (steps.length > 0) {
                planCard.appendChild(
                    buildAccordionItem(
                        "Execution Steps (" + steps.length + " bước)",
                        el("pre", "debug-code-pre", steps.join("\n")),
                        true
                    )
                );
            }
            panel.appendChild(planCard);
        }

        // 3. Tool & Skill Calls
        if (tools.length > 0 || (Array.isArray(snap.tool_calls) && snap.tool_calls.length > 0)) {
            panel.appendChild(el("h4", "debug-section-heading", "3. Tool & Skill Executions (" + tools.length + ")"));
            if (tools.length > 0) {
                for (const tc of tools) {
                    const a = tc.attributes || {};
                    const card = el("div", "debug-detail-card");
                    const hdr = el("div", "debug-detail-card-header");
                    hdr.append(
                        el("strong", "", tc.name),
                        el("div", "debug-detail-card-meta", "")
                    );
                    const meta = hdr.lastChild;
                    meta.append(
                        el("span", "debug-status-badge " + statusBadgeClass(a.status || tc.status), a.status || tc.status),
                        el("span", "debug-mono-chip", formatMs(tc.duration_ms))
                    );
                    if (a.simulation) {
                        meta.appendChild(el("span", "debug-pill debug-pill-warn", "SIMULATION"));
                    }
                    if (a.retried) {
                        meta.appendChild(el("span", "debug-pill debug-pill-warn", "RETRIED"));
                    }
                    card.appendChild(hdr);
                    card.appendChild(buildKeyValueTable([
                        ["Kind / Tool", (a.kind || "—") + " / " + (a.tool_name || "—")],
                        ["Error Code", a.error_code],
                        ["Retrieval Documents", a.documents_count],
                        ["Params", a.params],
                        ["Output Keys", renderPillsGroup(a.output_keys)],
                        ["Output Preview", a.output_preview]
                    ]));
                    panel.appendChild(card);
                }
            } else {
                panel.appendChild(el("pre", "debug-code-pre", JSON.stringify(snap.tool_calls, null, 2)));
            }
        }

        // 4. Graph Nodes & Memory / Delegations
        panel.appendChild(el("h4", "debug-section-heading", "4. Graph Nodes, Memory & Delegations"));
        if (graphNodes.length > 0) {
            const gCard = el("div", "debug-detail-card");
            const acc = el("div", "debug-accordion-group");
            for (const gn of graphNodes) {
                const title = gn.name + " · " + formatShortMs(gn.duration_ms) + " · " + gn.status;
                acc.appendChild(
                    buildAccordionItem(
                        title,
                        buildKeyValueTable(Object.entries(gn.attributes || {})),
                        false
                    )
                );
            }
            gCard.appendChild(acc);
            panel.appendChild(gCard);
        }

        const extraCard = el("div", "debug-detail-card");
        extraCard.appendChild(buildKeyValueTable([
            ["Delegations", snap.delegations],
            ["Memory Status", snap.memory],
            ["Pipeline Node Summary", snap.nodes],
            ["Citations / Images / Sources", (snap.citations_count || 0) + " citations · " + (snap.images_count || 0) + " images · " + JSON.stringify(snap.sources || [])]
        ]));
        panel.appendChild(extraCard);

        if (snap.interaction && typeof snap.interaction === "object") {
            panel.appendChild(el("h4", "debug-section-heading", "5. Interaction Prompt (Awaiting User Input)"));
            const interCard = el("div", "debug-detail-card");
            interCard.appendChild(buildKeyValueTable([
                ["Interaction ID", snap.interaction.id],
                ["Kind", snap.interaction.kind],
                ["Multi-Select", String(Boolean(snap.interaction.multi_select))],
                ["Prompt / Question", snap.interaction.question || snap.interaction.proposal || snap.interaction.prompt],
                ["Options", snap.interaction.options]
            ]));
            panel.appendChild(interCard);
        }

        if (Array.isArray(snap.human_replies) && snap.human_replies.length > 0) {
            panel.appendChild(el("h4", "debug-section-heading", "6. User Selected Replies (HITL)"));
            const replyCard = el("div", "debug-detail-card");
            const replyRows = [];
            for (let rIdx = 0; rIdx < snap.human_replies.length; rIdx++) {
                const rep = snap.human_replies[rIdx];
                if (rep && typeof rep === "object") {
                    replyRows.push([rep.question || ("Reply " + (rIdx + 1)), rep.response || JSON.stringify(rep)]);
                } else {
                    replyRows.push(["Reply " + (rIdx + 1), String(rep)]);
                }
            }
            replyCard.appendChild(buildKeyValueTable(replyRows));
            panel.appendChild(replyCard);
        }
    }

    function renderConfigTab(detail) {
        const panel = dom.panelConfigInner;
        if (!panel) {
            return;
        }
        panel.replaceChildren();

        const cfg = detail.system_config || state.systemConfig || {};
        panel.appendChild(el("h4", "debug-section-heading", "Cấu hình Hệ thống & Pipeline"));
        const sysCard = el("div", "debug-detail-card");
        sysCard.appendChild(buildKeyValueTable([
            ["Agent LLM Model", cfg.default_model],
            ["Agent Mode", cfg.agent_mode],
            ["Embedding Model", cfg.embedding_model],
            ["Reranker Model", cfg.reranker_model],
            ["Qdrant Collection", cfg.qdrant_collection],
            ["Indexed Points Count", cfg.points_count],
            ["Tracing Provider", cfg.tracing_provider || detail.provider],
            ["Tracing Enabled", String(cfg.tracing_enabled)],
            ["LangSmith Remote Active", String(cfg.langsmith_active)],
            ["LangSmith Project", cfg.langsmith_project || detail.project_name],
            ["LangSmith Base URL", cfg.langsmith_url],
            ["Local JSONL Export", String(cfg.local_export_enabled)]
        ]));
        panel.appendChild(sysCard);

        panel.appendChild(el("h4", "debug-section-heading", "Thông tin định danh Trace hiện tại"));
        const traceMetaCard = el("div", "debug-detail-card");
        traceMetaCard.appendChild(buildKeyValueTable([
            ["Schema Version", detail.schema_version],
            ["Trace ID", detail.trace_id],
            ["Run ID", detail.run_id],
            ["Turn ID", detail.turn_id],
            ["Session Ref (SHA-256)", detail.session_ref],
            ["Entrypoint", detail.entrypoint],
            ["Outcome", detail.outcome],
            ["LangSmith Run URL", detail.langsmith_run_url]
        ]));
        panel.appendChild(traceMetaCard);
    }

    function buildFullTraceResultPayload(detail) {
        if (!detail) {
            return {};
        }
        const snap = detail.turn_snapshot || {};
        let graphNodes = [];
        if (Array.isArray(detail.graph_nodes)) {
            graphNodes = detail.graph_nodes;
        }
        const dag = resolveDebugDagTelemetry(detail, snap, graphNodes);
        const flow = resolveLangGraphExecutionState(detail, dag);
        const executedNodes = [];
        const skippedNodes = [];
        for (let i = 0; i < LANGGRAPH_CANONICAL_NODES.length; i++) {
            const def = LANGGRAPH_CANONICAL_NODES[i];
            const st = flow.nodes[def.id];
            if (st && st.executed) {
                executedNodes.push({
                    id: def.id,
                    label: def.label,
                    status: st.status,
                    steps: st.steps,
                    call_count: st.call_count,
                    latency_ms: st.latency_ms
                });
            } else {
                skippedNodes.push(def.id);
            }
        }
        const edgeMap = computeExecutedEdgeMap(flow, detail);
        return {
            ...detail,
            langgraph_flow_summary: {
                executed_node_count: flow.executedCount,
                total_canonical_nodes: flow.totalCount,
                executed_nodes: executedNodes,
                skipped_nodes: skippedNodes,
                edges_traversed: Object.keys(edgeMap)
            }
        };
    }

    function renderJsonTab(detail) {
        const panel = dom.panelJsonInner;
        if (!panel) {
            return;
        }
        panel.replaceChildren();

        const fullPayload = buildFullTraceResultPayload(detail);
        const fullJson = JSON.stringify(fullPayload, null, 2);
        const bar = el("div", "debug-json-toolbar");

        const filterInput = document.createElement("input");
        filterInput.type = "search";
        filterInput.className = "debug-span-search";
        filterInput.placeholder = "Lọc các dòng chứa từ khóa trong JSON...";
        filterInput.value = state.jsonFilterQuery;

        const copyBtn = el("button", "debug-mini-btn", "Copy Full JSON");
        copyBtn.type = "button";
        copyBtn.addEventListener("click", () => {
            copyText(fullJson, copyBtn, "Copy Full JSON");
        });

        const dlBtn = el("button", "debug-mini-btn", "Tải file .json");
        dlBtn.type = "button";
        dlBtn.addEventListener("click", () => {
            downloadTraceJson(detail);
        });

        bar.append(filterInput, copyBtn, dlBtn);
        panel.appendChild(bar);

        const pre = el("pre", "debug-json-pre");
        function applyJsonFilter() {
            const q = state.jsonFilterQuery.trim().toLowerCase();
            if (!q) {
                pre.textContent = fullJson;
                return;
            }
            const lines = fullJson.split("\n");
            const matched = [];
            for (const line of lines) {
                if (line.toLowerCase().includes(q)) {
                    matched.push(line);
                }
            }
            if (matched.length === 0) {
                pre.textContent = "// Không tìm thấy dòng nào khớp với: " + state.jsonFilterQuery;
            } else {
                pre.textContent = matched.join("\n");
            }
        }
        filterInput.addEventListener("input", () => {
            state.jsonFilterQuery = filterInput.value;
            applyJsonFilter();
        });
        applyJsonFilter();
        panel.appendChild(pre);
    }

    function renderTraceDetail(detail) {
        if (!dom.emptyState || !dom.detailContainer) {
            return;
        }
        if (!detail) {
            dom.emptyState.hidden = false;
            dom.detailContainer.hidden = true;
            syncActionButtons(null);
            renderHeaderDagVisualizer(null);
            return;
        }

        dom.emptyState.hidden = true;
        dom.detailContainer.hidden = false;
        syncActionButtons(detail);

        renderHeaderDagVisualizer(detail);
        renderKpiAndIdentity(detail);
        renderWaterfallTab(detail);
        renderLlmTab(detail);
        renderRagTab(detail);
        renderAgentTab(detail);
        renderConfigTab(detail);
        renderJsonTab(detail);
    }

    function downloadTraceJson(detail) {
        if (!detail) {
            return;
        }
        const fullPayload = buildFullTraceResultPayload(detail);
        const jsonText = JSON.stringify(fullPayload, null, 2);
        const blob = new Blob([jsonText], { type: "application/json;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "trace-" + (detail.trace_id || "debug") + ".json";
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
    }

    async function fetchTraceList(autoSelect) {
        if (state.loadingList) {
            state.pendingReload = true;
            if (autoSelect) {
                state.pendingAutoSelect = true;
            }
            return;
        }
        state.loadingList = true;
        try {
            const params = new URLSearchParams();
            if (state.sessionId) {
                params.set("session_id", state.sessionId);
            }
            params.set("scope", state.scope);
            params.set("limit", "100");
            params.set("include_active", "true");

            const resp = await fetch("/api/debug/traces?" + params.toString());
            if (!resp.ok) {
                return;
            }
            const data = await resp.json();
            if (Array.isArray(data.traces)) {
                state.traces = data.traces;
            } else {
                state.traces = [];
            }
            state.activeCount = Number(data.active_count || 0);
            state.totalStored = Number(data.total_stored || state.traces.length);
            if (data.system_config) {
                state.systemConfig = data.system_config;
            }
            syncHeaderIndicators();
            syncTopSystemBadges();
            renderTraceList();

            if (autoSelect && state.traces.length > 0) {
                const stillExists = state.traces.some((item) => item.trace_id === state.selectedTraceId);
                if (!state.selectedTraceId || !stillExists) {
                    await selectTrace(state.traces[0].trace_id);
                } else if (dom.dialog && dom.dialog.open) {
                    await selectTrace(state.selectedTraceId);
                }
            } else if (state.traces.length === 0) {
                state.selectedTraceId = null;
                state.selectedDetail = null;
                renderTraceDetail(null);
            }
        } catch (err) {
            console.warn("Fetch debug trace list failed:", err);
        } finally {
            state.loadingList = false;
            syncPollingState();
            if (state.pendingReload) {
                const nextAuto = state.pendingAutoSelect;
                state.pendingReload = false;
                state.pendingAutoSelect = false;
                fetchTraceList(nextAuto);
            }
        }
    }

    async function selectTrace(traceId) {
        if (!traceId) {
            return;
        }
        state.selectedTraceId = traceId;
        renderTraceList();
        state.loadingDetail = true;
        try {
            const resp = await fetch("/api/debug/traces/" + encodeURIComponent(traceId));
            if (!resp.ok) {
                if (resp.status === 404) {
                    state.selectedDetail = null;
                    renderTraceDetail(null);
                }
                return;
            }
            const detail = await resp.json();
            state.selectedDetail = detail;
            if (detail.system_config) {
                state.systemConfig = detail.system_config;
                syncTopSystemBadges();
            }
            renderTraceDetail(detail);
        } catch (err) {
            console.warn("Fetch debug trace detail failed:", err);
        } finally {
            state.loadingDetail = false;
            syncPollingState();
        }
    }

    function stopPolling() {
        if (state.pollTimer !== null) {
            window.clearInterval(state.pollTimer);
            state.pollTimer = null;
        }
    }

    function syncPollingState() {
        const dialogOpen = Boolean(dom.dialog && dom.dialog.open);
        const shouldPoll = dialogOpen && (
            Boolean(state.liveTraceId)
            || state.activeCount > 0
            || Boolean(state.selectedDetail && state.selectedDetail.is_active)
        );
        if (shouldPoll && state.pollTimer === null) {
            state.pollTimer = window.setInterval(() => {
                fetchTraceList(false).then(() => {
                    if (state.selectedTraceId && dom.dialog && dom.dialog.open) {
                        selectTrace(state.selectedTraceId);
                    }
                });
            }, 1200);
        } else if (!shouldPoll) {
            stopPolling();
        }
    }

    function openDialogModal() {
        if (!dom.dialog) {
            return;
        }
        if (!dom.dialog.open) {
            dom.dialog.showModal();
        }
    }

    async function openLatest() {
        openDialogModal();
        if (state.liveTraceId) {
            state.selectedTraceId = state.liveTraceId;
        } else {
            state.selectedTraceId = null;
        }
        await fetchTraceList(true);
        syncPollingState();
    }

    async function openTrace(traceId) {
        openDialogModal();
        if (traceId) {
            state.selectedTraceId = traceId;
        }
        await fetchTraceList(false);
        if (traceId) {
            await selectTrace(traceId);
        } else if (state.traces.length > 0) {
            await selectTrace(state.traces[0].trace_id);
        }
        syncPollingState();
    }

    function onLiveTraceStart(traceId) {
        if (traceId) {
            state.liveTraceId = traceId;
            state.selectedTraceId = traceId;
        }
        syncHeaderIndicators();
        if (dom.dialog && dom.dialog.open) {
            fetchTraceList(false).then(() => {
                if (traceId) {
                    selectTrace(traceId);
                }
            });
            syncPollingState();
        }
    }

    function onTraceCompleted(traceId) {
        if (traceId && state.liveTraceId === traceId) {
            state.liveTraceId = null;
        } else if (!traceId) {
            state.liveTraceId = null;
        }
        if (traceId) {
            state.selectedTraceId = traceId;
        }
        syncHeaderIndicators();
        fetchTraceList(false).then(() => {
            if (dom.dialog && dom.dialog.open && traceId && state.selectedTraceId === traceId) {
                selectTrace(traceId);
            }
        });
    }

    function setSessionId(sessionId) {
        stopReplayPlayback();
        state.replayStepIndex = -1;
        state.replayTraceId = null;
        state.sessionId = sessionId || "";
        state.selectedTraceId = null;
        state.selectedDetail = null;
        state.selectedSpanId = null;
        state.liveTraceId = null;
        if (dom.dialog && dom.dialog.open) {
            fetchTraceList(true);
        } else {
            fetchTraceList(false);
        }
    }

    function bindEvents() {
        if (dom.debugBtn) {
            dom.debugBtn.addEventListener("click", () => {
                openLatest();
            });
        }

        if (dom.dialog) {
            dom.dialog.addEventListener("close", () => {
                dom.dialog.returnValue = "";
                stopPolling();
                stopReplayPlayback();
            });

            if (!("closedBy" in HTMLDialogElement.prototype)) {
                let pressedOutside = false;
                const isOutside = (event) => {
                    const r = dom.dialog.getBoundingClientRect();
                    return event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom;
                };
                dom.dialog.addEventListener("pointerdown", (event) => {
                    pressedOutside = isOutside(event);
                });
                dom.dialog.addEventListener("pointerup", (event) => {
                    if (pressedOutside && isOutside(event)) {
                        dom.dialog.close();
                    }
                });
            }
        }

        if (dom.scopeSessionBtn) {
            dom.scopeSessionBtn.addEventListener("click", () => {
                state.scope = "session";
                syncScopeButtons();
                fetchTraceList(true);
            });
        }

        if (dom.scopeAllBtn) {
            dom.scopeAllBtn.addEventListener("click", () => {
                state.scope = "all";
                syncScopeButtons();
                fetchTraceList(true);
            });
        }

        if (dom.refreshBtn) {
            dom.refreshBtn.addEventListener("click", () => {
                fetchTraceList(true);
            });
        }

        if (dom.copyJsonBtn) {
            dom.copyJsonBtn.addEventListener("click", () => {
                if (state.selectedDetail) {
                    const fullPayload = buildFullTraceResultPayload(state.selectedDetail);
                    copyText(JSON.stringify(fullPayload, null, 2), dom.copyJsonBtn, "Copy Full JSON");
                }
            });
        }

        if (dom.exportJsonBtn) {
            dom.exportJsonBtn.addEventListener("click", () => {
                if (state.selectedDetail) {
                    downloadTraceJson(state.selectedDetail);
                }
            });
        }

        if (dom.fullscreenBtn && dom.dialog) {
            dom.fullscreenBtn.addEventListener("click", () => {
                const isFull = dom.dialog.classList.toggle("is-fullscreen");
                dom.fullscreenBtn.setAttribute("aria-pressed", String(isFull));
                if (isFull) {
                    dom.fullscreenBtn.textContent = "⛶ Thu gọn";
                } else {
                    dom.fullscreenBtn.textContent = "⛶ Toàn màn hình";
                }
            });
        }

        if (dom.clearBtn) {
            dom.clearBtn.addEventListener("click", async () => {
                const params = new URLSearchParams();
                if (state.sessionId) {
                    params.set("session_id", state.sessionId);
                }
                params.set("scope", state.scope);
                try {
                    const resp = await fetch("/api/debug/traces?" + params.toString(), { method: "DELETE" });
                    if (resp.ok) {
                        state.selectedTraceId = null;
                        state.selectedDetail = null;
                        state.selectedSpanId = null;
                        await fetchTraceList(false);
                    }
                } catch (err) {
                    console.warn("Clear debug traces failed:", err);
                }
            });
        }

        if (dom.searchInput) {
            dom.searchInput.addEventListener("input", () => {
                state.searchQuery = dom.searchInput.value;
                renderTraceList();
            });
        }

        if (dom.tablist && dom.tabs) {
            for (let idx = 0; idx < dom.tabs.length; idx++) {
                const tab = dom.tabs[idx];
                tab.addEventListener("click", () => {
                    activateTab(tab.id, false);
                });
                tab.addEventListener("keydown", (event) => {
                    let targetIdx = -1;
                    if (event.key === "ArrowRight") {
                        targetIdx = (idx + 1) % dom.tabs.length;
                    } else if (event.key === "ArrowLeft") {
                        targetIdx = (idx - 1 + dom.tabs.length) % dom.tabs.length;
                    } else if (event.key === "Home") {
                        targetIdx = 0;
                    } else if (event.key === "End") {
                        targetIdx = dom.tabs.length - 1;
                    }
                    if (targetIdx >= 0) {
                        event.preventDefault();
                        activateTab(dom.tabs[targetIdx].id, true);
                    }
                });
            }
        }

        if (dom.panels) {
            for (const panel of dom.panels) {
                panel.addEventListener("beforematch", () => {
                    const ownerId = panel.getAttribute("aria-labelledby");
                    if (ownerId) {
                        activateTab(ownerId, false);
                    }
                });
            }
        }

        document.addEventListener("keydown", (event) => {
            if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === "d") {
                event.preventDefault();
                if (dom.dialog && dom.dialog.open) {
                    dom.dialog.close();
                } else {
                    openLatest();
                }
            }
        });
    }

    function init(options) {
        if (state.initialized) {
            return;
        }
        state.initialized = true;
        if (options && options.sessionId) {
            state.sessionId = String(options.sessionId);
        }

        dom.debugBtn = document.getElementById("debugTraceBtn");
        dom.liveDot = document.getElementById("debugLiveDot");
        dom.traceCountBadge = document.getElementById("debugTraceCount");
        dom.dialog = document.getElementById("debugTraceDialog");
        dom.langsmithBadge = document.getElementById("debugLangSmithBadge");
        dom.projectBadge = document.getElementById("debugProjectBadge");
        dom.scopeSessionBtn = document.getElementById("debugScopeSessionBtn");
        dom.scopeAllBtn = document.getElementById("debugScopeAllBtn");
        dom.refreshBtn = document.getElementById("debugRefreshBtn");
        dom.langsmithLink = document.getElementById("debugLangSmithLink");
        dom.copyJsonBtn = document.getElementById("debugCopyJsonBtn");
        dom.exportJsonBtn = document.getElementById("debugExportJsonBtn");
        dom.fullscreenBtn = document.getElementById("debugFullscreenBtn");
        dom.clearBtn = document.getElementById("debugClearBtn");
        dom.searchInput = document.getElementById("debugTraceSearch");
        dom.sidebarMeta = document.getElementById("debugSidebarMeta");
        dom.traceList = document.getElementById("debugTraceList");
        dom.emptyState = document.getElementById("debugEmptyState");
        dom.detailContainer = document.getElementById("debugTraceDetail");
        dom.kpiStrip = document.getElementById("debugKpiStrip");
        dom.identityBar = document.getElementById("debugIdentityBar");
        dom.tablist = document.querySelector("#debugTraceDialog .debug-tablist");
        dom.tabs = Array.from(document.querySelectorAll('#debugTraceDialog [role="tab"]'));
        dom.panels = Array.from(document.querySelectorAll('#debugTraceDialog [role="tabpanel"]'));
        dom.panelWaterfallInner = document.getElementById("debugPanelWaterfallInner");
        dom.panelLlmInner = document.getElementById("debugPanelLlmInner");
        dom.panelRagInner = document.getElementById("debugPanelRagInner");
        dom.panelAgentInner = document.getElementById("debugPanelAgentInner");
        dom.panelConfigInner = document.getElementById("debugPanelConfigInner");
        dom.panelJsonInner = document.getElementById("debugPanelJsonInner");
        dom.dagGraphTrack = document.getElementById("dagGraphTrack");
        dom.dagNodeInspector = document.getElementById("dagNodeInspector");

        syncScopeButtons();
        syncHeaderIndicators();
        bindEvents();
        fetchTraceList(false);
    }

    window.ABankDebugInspector = {
        init: init,
        setSessionId: setSessionId,
        openLatest: openLatest,
        openTrace: openTrace,
        onLiveTraceStart: onLiveTraceStart,
        onTraceCompleted: onTraceCompleted,
        refresh: fetchTraceList
    };
})();
