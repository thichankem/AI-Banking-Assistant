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

    function copyText(text, btn, defaultLabel) {
        if (!text) {
            return;
        }
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(String(text)).then(() => {
                if (btn) {
                    btn.textContent = "Đã sao chép";
                    window.setTimeout(() => {
                        btn.textContent = defaultLabel;
                    }, 1500);
                }
            }).catch(() => {});
        }
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
            if (it.snippet) {
                docCell.appendChild(
                    buildAccordionItem(
                        "Đoạn trích (" + (it.chunk_id || "chunk") + ")",
                        el("div", "debug-snippet-text", it.snippet),
                        false
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

        if (parsers.length === 0 && retrievals.length === 0 && rerankers.length === 0 && embeddings.length === 0 && finalChunks.length === 0) {
            panel.appendChild(el("div", "debug-muted-note", "Turn này không thực hiện bước truy xuất RAG nào."));
            return;
        }

        if (parsers.length > 0) {
            panel.appendChild(el("h4", "debug-section-heading", "1. Query Parser & Chuẩn hóa truy vấn (" + parsers.length + ")"));
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
            panel.appendChild(el("h4", "debug-section-heading", "2. Hybrid Retrieval — Dense Vector + BM25 + RRF (" + retrievals.length + ")"));
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
            panel.appendChild(el("h4", "debug-section-heading", "3. Cross-Encoder Reranker (" + rerankers.length + ")"));
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
            panel.appendChild(el("h4", "debug-section-heading", "4. Dense Embeddings (" + embeddings.length + ")"));
            for (const em of embeddings) {
                const card = el("div", "debug-detail-card");
                card.appendChild(buildKeyValueTable(Object.entries(em.attributes || {})));
                panel.appendChild(card);
            }
        }

        if (finalChunks.length > 0) {
            panel.appendChild(el("h4", "debug-section-heading", "5. Evidence Chunks cuối cùng đưa vào Agent (" + finalChunks.length + ")"));
            const acc = el("div", "debug-accordion-group");
            for (const ch of finalChunks) {
                const docName = String(ch.source || "—").split("\\").join("/").split("/").pop();
                let title = "[" + ch.index + "] " + docName;
                if (ch.section_title) {
                    title += " · " + ch.section_title;
                }
                if (Array.isArray(ch.page_numbers) && ch.page_numbers.length > 0) {
                    title += " (Tr. " + ch.page_numbers.join(", ") + ")";
                }
                const body = el("div", "");
                body.appendChild(buildKeyValueTable([
                    ["Chunk ID", ch.chunk_id],
                    ["Source", ch.source],
                    ["Breadcrumb", ch.breadcrumb],
                    ["Type", ch.chunk_type],
                    ["Excerpt", ch.excerpt_preview]
                ]));
                acc.appendChild(buildAccordionItem(title, body, false));
            }
            panel.appendChild(acc);
        }
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

    function renderJsonTab(detail) {
        const panel = dom.panelJsonInner;
        if (!panel) {
            return;
        }
        panel.replaceChildren();

        const fullJson = JSON.stringify(detail, null, 2);
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
            return;
        }

        dom.emptyState.hidden = true;
        dom.detailContainer.hidden = false;
        syncActionButtons(detail);

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
        const jsonText = JSON.stringify(detail, null, 2);
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
                    copyText(JSON.stringify(state.selectedDetail, null, 2), dom.copyJsonBtn, "Copy JSON");
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
