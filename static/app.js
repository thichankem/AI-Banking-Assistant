/**
 * AI Banking Assistant - Modern Client v4.5 (Vietcombank Specialized Agentic RAG)
 * Tuân thủ 100%: 0% Regex Law & Zero Ternary Law.
 */

document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const chatMessages = document.getElementById("chatMessages");
    const clearChatBtn = document.getElementById("clearChatBtn");

    // Stepper DOM Elements
    const nodeStepper = document.getElementById("nodeStepper");
    const stepperDesc = document.getElementById("stepperDesc");
    const stepperBadge = document.getElementById("stepperBadge");

    const stepNode0 = document.getElementById("step-node0");
    const stepNode1 = document.getElementById("step-node1");
    const stepNode2 = document.getElementById("step-node2");
    const stepNode3 = document.getElementById("step-node3");
    const stepNode6 = document.getElementById("step-node6");
    const stepNode7 = document.getElementById("step-node7");
    const stepNode8 = document.getElementById("step-node8");

    // Session ID cố định cho phiên hội thoại (hỗ trợ thread_id bộ nhớ)
    let currentSessionId = "vcb_session_" + Math.floor(Math.random() * 1000000);

    const initialWelcomeHtml = chatMessages.innerHTML;

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

    // Xử lý in đậm và code inline (0% Regex, Zero Ternary)
    function renderInlineFormatting(str) {
        let safe = escapeHtml(str);

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

        return safe;
    }

    // Parse Markdown (0% Regex, Zero Ternary)
    function parseMarkdown(text) {
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
                    tblHtml.push("<th>" + renderInlineFormatting(c) + "</th>");
                }
                tblHtml.push('</tr></thead>');
                startIdx = 2;
            }

            tblHtml.push('<tbody>');
            for (let r = startIdx; r < tableRows.length; r++) {
                tblHtml.push('<tr>');
                for (let c of tableRows[r]) {
                    tblHtml.push("<td>" + renderInlineFormatting(c) + "</td>");
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
                htmlParts.push("<h4>" + renderInlineFormatting(trimmed.substring(4)) + "</h4>");
                continue;
            }
            if (trimmed.startsWith("## ")) {
                if (inList) {
                    htmlParts.push("</ul>");
                    inList = false;
                }
                htmlParts.push("<h3>" + renderInlineFormatting(trimmed.substring(3)) + "</h3>");
                continue;
            }

            if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
                if (!inList) {
                    htmlParts.push("<ul>");
                    inList = true;
                }
                htmlParts.push("<li>" + renderInlineFormatting(trimmed.substring(2)) + "</li>");
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

            htmlParts.push("<p>" + renderInlineFormatting(trimmed) + "</p>");
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

    function getSubAgentBadgeClass(role) {
        if (role === "CREDIT_CARD_AGENT") {
            return "badge-credit-card";
        } else if (role === "LOAN_MORTGAGE_AGENT") {
            return "badge-loan-mortgage";
        } else if (role === "COMPLIANCE_AUDIT_AGENT") {
            return "badge-compliance-audit";
        } else {
            return "badge-general-advisor";
        }
    }

    function resetStepper() {
        const steps = [stepNode0, stepNode1, stepNode2, stepNode3, stepNode6, stepNode7, stepNode8];
        for (let s of steps) {
            if (s) {
                s.className = "step-item";
            }
        }
        if (stepperDesc) {
            stepperDesc.textContent = "Khởi tạo luồng xử lý Agent...";
        }
        if (stepperBadge) {
            stepperBadge.className = "sub-agent-badge badge-general-advisor";
            stepperBadge.textContent = "[SUPERVISOR]";
        }
    }

    function updateStepperProgress(nodeName, description, subAgent) {
        if (nodeStepper) {
            nodeStepper.style.display = "block";
        }
        if (stepperDesc && description) {
            stepperDesc.textContent = description;
        }
        if (stepperBadge && subAgent) {
            stepperBadge.className = "sub-agent-badge " + getSubAgentBadgeClass(subAgent);
            stepperBadge.textContent = "[" + subAgent + "]";
        }

        if (nodeName === "input_guardrail") {
            if (stepNode0) {
                stepNode0.className = "step-item active";
            }
        } else if (nodeName === "router") {
            if (stepNode0) {
                stepNode0.className = "step-item completed";
            }
            if (stepNode1) {
                stepNode1.className = "step-item active";
            }
        } else if (nodeName === "retrieve" || nodeName === "banking_tool") {
            if (stepNode1) {
                stepNode1.className = "step-item completed";
            }
            if (stepNode2) {
                stepNode2.className = "step-item active";
            }
        } else if (nodeName === "grade_documents" || nodeName === "rewrite_query") {
            if (stepNode2) {
                stepNode2.className = "step-item completed";
            }
            if (stepNode3) {
                stepNode3.className = "step-item active";
            }
        } else if (nodeName === "generate") {
            if (stepNode3) {
                stepNode3.className = "step-item completed";
            }
            if (stepNode6) {
                stepNode6.className = "step-item active";
            }
        } else if (nodeName === "audit_generation") {
            if (stepNode6) {
                stepNode6.className = "step-item completed";
            }
            if (stepNode7) {
                stepNode7.className = "step-item active";
            }
        } else if (nodeName === "output_guardrail") {
            if (stepNode7) {
                stepNode7.className = "step-item completed";
            }
            if (stepNode8) {
                stepNode8.className = "step-item active";
            }
        }
    }

    function completeAllStepper() {
        const steps = [stepNode0, stepNode1, stepNode2, stepNode3, stepNode6, stepNode7, stepNode8];
        for (let s of steps) {
            if (s) {
                s.className = "step-item completed";
            }
        }
        if (stepperDesc) {
            stepperDesc.textContent = "Hoàn tất phản hồi an toàn!";
        }
    }

    function appendMessage(sender, content, meta = null) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message " + sender + "-message";

        if (sender === "user") {
            msgDiv.innerHTML = `
                <div class="bubble">
                    <p>${escapeHtml(content)}</p>
                </div>
            `;
        } else {
            let roleBadgeHtml = "";
            if (meta && meta.sub_agent) {
                const badgeClass = getSubAgentBadgeClass(meta.sub_agent);
                roleBadgeHtml = `<div style="margin-bottom: 8px;"><span class="sub-agent-badge ${badgeClass}">[${escapeHtml(meta.sub_agent)}]</span></div>`;
            }

            let metricsHtml = "";
            if (meta && meta.evaluation_metrics) {
                const em = meta.evaluation_metrics;
                let cpVal = 0;
                if (em.context_precision) {
                    cpVal = em.context_precision;
                }
                let faithVal = 0;
                if (em.faithfulness) {
                    faithVal = em.faithfulness;
                }
                let relVal = 0;
                if (em.answer_relevance) {
                    relVal = em.answer_relevance;
                }
                let ovVal = 0;
                if (em.overall_score) {
                    ovVal = em.overall_score;
                }

                const cp = Math.round(cpVal * 100);
                const faith = Math.round(faithVal * 100);
                const rel = Math.round(relVal * 100);
                const ov = Math.round(ovVal * 100);

                metricsHtml = `
                    <div class="eval-metrics-card">
                        <div class="metric-pill"><span class="metric-name">Context Precision:</span> <span class="metric-val">${cp}%</span></div>
                        <div class="metric-pill"><span class="metric-name">Faithfulness:</span> <span class="metric-val">${faith}%</span></div>
                        <div class="metric-pill"><span class="metric-name">Relevance:</span> <span class="metric-val">${rel}%</span></div>
                        <div class="metric-pill"><span class="metric-name">Overall:</span> <span class="metric-val">${ov}%</span></div>
                    </div>
                `;
            }

            let sourcesHtml = "";
            if (meta) {
                const combined = [];
                if (meta.citations) {
                    for (let c of meta.citations) {
                        if (c && !combined.includes(c)) {
                            combined.push(c);
                        }
                    }
                }
                if (meta.sources) {
                    for (let s of meta.sources) {
                        if (s && !combined.includes(s)) {
                            combined.push(s);
                        }
                    }
                }

                if (combined.length > 0) {
                    let pills = [];
                    for (let item of combined.slice(0, 4)) {
                        pills.push(`<span class="source-tag">${escapeHtml(item)}</span>`);
                    }
                    sourcesHtml = `
                        <div class="sources-card">
                            <div class="sources-header">Căn cứ quy chuẩn Vietcombank:</div>
                            <div class="sources-pills">${pills.join("")}</div>
                        </div>
                    `;
                }
            }

            msgDiv.innerHTML = `
                <div class="bubble">
                    ${roleBadgeHtml}
                    ${parseMarkdown(content)}
                    ${metricsHtml}
                    ${sourcesHtml}
                </div>
            `;
        }

        chatMessages.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    async function submitQueryFallback(cleanQuery) {
        try {
            const res = await fetch("/api/agent/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: cleanQuery, session_id: currentSessionId })
            });
            if (!res.ok) {
                const errData = await res.json().catch(() => ({ detail: "Lỗi kết nối máy chủ dịch vụ" }));
                let detailMsg = "Không thể xử lý yêu cầu lúc này.";
                if (errData && errData.detail) {
                    detailMsg = errData.detail;
                }
                appendMessage("bot", "**Rất tiếc:** " + detailMsg);
            } else {
                const data = await res.json();
                completeAllStepper();
                let ans = "Không nhận được phản hồi từ hệ thống.";
                if (data.answer) {
                    ans = data.answer;
                }
                appendMessage("bot", ans, {
                    sub_agent: data.sub_agent_role,
                    evaluation_metrics: data.evaluation_metrics,
                    citations: data.citations,
                    sources: data.sources
                });
            }
        } catch (err) {
            appendMessage("bot", "**Lỗi kết nối:** Không thể liên lạc với máy chủ (" + err.message + ").");
        }
    }

    async function submitQuery(queryText) {
        let cleanQuery = "";
        if (queryText) {
            cleanQuery = queryText.trim();
        }
        if (cleanQuery === "") {
            return;
        }

        appendMessage("user", cleanQuery);
        userInput.value = "";
        userInput.disabled = true;
        sendBtn.disabled = true;

        resetStepper();
        if (nodeStepper) {
            nodeStepper.style.display = "block";
        }

        // Tạo bubble tạm thời để stream tokens vào
        const botMsgDiv = document.createElement("div");
        botMsgDiv.className = "message bot-message";
        botMsgDiv.innerHTML = `
            <div class="bubble" id="streamingBubble">
                <span id="streamingText" style="color: #94a3b8; font-style: italic;">Đang kết nối hệ thống Multi-Agent...</span>
            </div>
        `;
        chatMessages.appendChild(botMsgDiv);
        scrollToBottom();

        const streamingBubble = botMsgDiv.querySelector("#streamingBubble");
        let accumulatedText = "";
        let finalDoneMeta = null;
        let streamSuccess = false;

        try {
            const response = await fetch("/api/agent/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: cleanQuery, session_id: currentSessionId })
            });

            if (!response.ok || !response.body) {
                botMsgDiv.remove();
                await submitQueryFallback(cleanQuery);
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    break;
                }

                buffer = buffer + decoder.decode(value, { stream: true });
                const parts = buffer.split("\n\n");
                buffer = parts.pop();

                for (let part of parts) {
                    const trimmedPart = part.trim();
                    if (trimmedPart.startsWith("data: ")) {
                        const jsonStr = trimmedPart.substring(6).trim();
                        try {
                            const eventData = JSON.parse(jsonStr);
                            const eventType = eventData.event_type;

                            if (eventType === "node_progress") {
                                updateStepperProgress(eventData.node, eventData.description, eventData.sub_agent);
                            } else if (eventType === "token") {
                                streamSuccess = true;
                                accumulatedText = accumulatedText + eventData.content;
                                if (streamingBubble) {
                                    streamingBubble.innerHTML = parseMarkdown(accumulatedText);
                                    scrollToBottom();
                                }
                            } else if (eventType === "done") {
                                streamSuccess = true;
                                finalDoneMeta = eventData;
                                completeAllStepper();
                            }
                        } catch (e) {
                            console.warn("Parse SSE event error:", e);
                        }
                    }
                }
            }

            // Nếu stream thành công và có full_answer, cập nhật bong bóng hoàn chỉnh
            if (streamSuccess) {
                let finalText = accumulatedText;
                if (finalDoneMeta && finalDoneMeta.full_answer) {
                    finalText = finalDoneMeta.full_answer;
                }

                let metaSubAgent = "GENERAL_ADVISOR";
                let metaEvalMetrics = null;
                if (finalDoneMeta) {
                    if (finalDoneMeta.sub_agent) {
                        metaSubAgent = finalDoneMeta.sub_agent;
                    }
                    if (finalDoneMeta.evaluation_metrics) {
                        metaEvalMetrics = finalDoneMeta.evaluation_metrics;
                    }
                }

                botMsgDiv.remove();
                appendMessage("bot", finalText, {
                    sub_agent: metaSubAgent,
                    evaluation_metrics: metaEvalMetrics
                });
            } else {
                botMsgDiv.remove();
                await submitQueryFallback(cleanQuery);
            }

        } catch (err) {
            console.warn("SSE stream failed, falling back to chat endpoint:", err);
            botMsgDiv.remove();
            await submitQueryFallback(cleanQuery);
        } finally {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
            scrollToBottom();
        }
    }

    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        submitQuery(userInput.value);
    });

    if (clearChatBtn) {
        clearChatBtn.addEventListener("click", () => {
            currentSessionId = "vcb_session_" + Math.floor(Math.random() * 1000000);
            chatMessages.innerHTML = initialWelcomeHtml;
            if (nodeStepper) {
                nodeStepper.style.display = "none";
            }
            userInput.value = "";
            userInput.focus();
        });
    }

    if (userInput) {
        userInput.focus();
    }
});
