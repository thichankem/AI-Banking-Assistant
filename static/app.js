/**
 * A-Bank web chat
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
    const agentActions = document.getElementById("agentActions");
    const agentPlan = document.getElementById("agentPlan");

    function newSessionId() {
        return Array.from(window.crypto.getRandomValues(new Uint32Array(4)), value => value.toString(16).padStart(8, "0")).join("");
    }

    const sessionKey = "banking-agent-session";
    let currentSessionId = sessionStorage.getItem(sessionKey) || newSessionId();
    sessionStorage.setItem(sessionKey, currentSessionId);
    let pendingCard = null;
    let busy = false;

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

    function renderPlan(plan) {
        if (!Array.isArray(plan)) {
            return;
        }
        const labels = {
            pending: "Chờ xử lý", in_progress: "Đang xử lý",
            done: "Hoàn tất", blocked: "Cần bổ sung"
        };
        const items = plan.map(step => {
            const item = document.createElement("li");
            item.textContent = step.task + " · " + (labels[step.status] || step.status);
            return item;
        });
        agentPlan.replaceChildren(...items);
    }

    function resetStepper() {
        renderPlan([]);
        agentActions.replaceChildren();
        stepperDesc.textContent = "Đang nhận yêu cầu...";
    }

    function updateStepperProgress(description, action) {
        stepperDesc.textContent = description;
        if (action) {
            const item = document.createElement("li");
            item.textContent = description;
            agentActions.appendChild(item);
        }
    }

    function finishProgress(message = "Đã hoàn tất phản hồi") {
        stepperDesc.textContent = message;
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
            let sourcesHtml = "";
            if (meta) {
                const combined = new Set();
                for (const citation of meta.citations || []) {
                    if (typeof citation === "string") {
                        combined.add(citation);
                    } else if (citation && citation.source) {
                        combined.add(citation.label + " " + citation.source);
                    }
                }
                if (combined.size === 0) {
                    for (const source of meta.sources || []) {
                        combined.add(source);
                    }
                }

                if (combined.size > 0) {
                    let pills = [];
                    for (let item of Array.from(combined).slice(0, 6)) {
                        pills.push(`<span class="source-tag">${escapeHtml(item)}</span>`);
                    }
                    sourcesHtml = `
                        <div class="sources-card">
                            <div class="sources-header">Tài liệu đã trích dẫn</div>
                            <div class="sources-pills">${pills.join("")}</div>
                        </div>
                    `;
                }
            }

            msgDiv.innerHTML = `
                <div class="bubble">
                    ${parseMarkdown(content)}
                    ${sourcesHtml}
                </div>
            `;
        }

        chatMessages.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    function setBusy(value) {
        busy = value;
        userInput.disabled = busy || Boolean(pendingCard);
        sendBtn.disabled = userInput.disabled;
        if (clearChatBtn) {
            clearChatBtn.disabled = busy;
        }
    }

    function renderInteraction(data) {
        const card = data.interaction;
        if (pendingCard && pendingCard.id === card.id) {
            return;
        }
        if (pendingCard) {
            pendingCard.element.remove();
        }
        renderPlan(data.plan);
        finishProgress("Đang chờ lựa chọn của bạn");
        const element = document.createElement("div");
        element.className = "message bot-message";
        const bubble = document.createElement("div");
        bubble.className = "bubble interaction-card";
        const message = document.createElement("p");
        message.textContent = card.message;
        bubble.appendChild(message);
        if (card.scope.length) {
            const list = document.createElement("ol");
            for (const item of card.scope) {
                const entry = document.createElement("li");
                entry.textContent = item.task;
                list.appendChild(entry);
            }
            bubble.appendChild(list);
        }

        const choices = document.createElement("div");
        choices.className = "interaction-options";
        const other = document.createElement("form");
        other.className = "interaction-options";
        other.hidden = true;
        const field = document.createElement("input");
        field.maxLength = 500;
        field.required = true;
        field.setAttribute("aria-label", "Câu trả lời khác");
        field.placeholder = "Nhập lựa chọn của bạn";
        const submit = document.createElement("button");
        submit.textContent = "Gửi lựa chọn";
        const feedback = document.createElement("p");
        feedback.setAttribute("role", "status");
        other.append(field, submit);
        other.addEventListener("submit", event => {
            event.preventDefault();
            const value = field.value.trim();
            if (value) {
                sendChoice(card, "other", value, bubble, feedback);
            }
        });
        for (const option of card.options) {
            const button = document.createElement("button");
            button.type = "button";
            button.textContent = option.label;
            button.addEventListener("click", () => {
                if (option.id === "other") {
                    other.hidden = false;
                    field.focus();
                } else {
                    sendChoice(card, option.id, option.label, bubble, feedback);
                }
            });
            choices.appendChild(button);
        }
        bubble.append(choices, other, feedback);
        element.appendChild(bubble);
        pendingCard = { id: card.id, element };
        chatMessages.appendChild(element);
        setBusy(busy);
        scrollToBottom();
    }

    function renderAgentResult(data) {
        if (data.status === "awaiting_input") {
            renderInteraction(data);
            return;
        }
        renderPlan(data.plan);
        finishProgress();
        appendMessage("bot", data.answer || "Không nhận được phản hồi từ hệ thống.", data);
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

    async function sendChoice(card, choiceId, label, bubble, feedback) {
        if (busy || !pendingCard || pendingCard.id !== card.id) {
            return;
        }
        setBusy(true);
        const controls = bubble.querySelectorAll("button, input");
        controls.forEach(control => { control.disabled = true; });
        feedback.textContent = "Đang gửi lựa chọn…";
        try {
            const payload = { interaction_id: card.id, choice_id: choiceId };
            if (choiceId === "other") {
                payload.text = label;
            }
            const response = await postAgent("/api/agent/resume", payload);
            if (response.status === 422) {
                const data = await response.json();
                feedback.textContent = "Vui lòng kiểm tra lại lựa chọn.";
                if (typeof data.detail === "string") {
                    feedback.textContent = data.detail;
                }
                return;
            }
            if (!response.ok) {
                throw new Error("Không nhận được kết quả tiếp tục.");
            }
            const data = await response.json();
            pendingCard = null;
            bubble.querySelectorAll(".interaction-options").forEach(group => group.remove());
            feedback.remove();
            if (choiceId === "other") {
                appendMessage("user", "Đã gửi câu trả lời khác.");
            } else {
                appendMessage("user", label);
            }
            renderAgentResult(data);
        } catch {
            if (!await recoverInteraction(card.id)) {
                feedback.textContent = "Chưa nhận được kết quả. Bạn có thể thử lại.";
            }
        } finally {
            if (pendingCard && pendingCard.id === card.id) {
                controls.forEach(control => { control.disabled = false; });
            }
            setBusy(false);
        }
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
                renderAgentResult(data);
            }
        } catch (err) {
            appendMessage("bot", "**Lỗi kết nối:** Không thể liên lạc với máy chủ (" + err.message + ").");
        }
    }

    async function submitQuery(queryText) {
        if (busy || pendingCard) {
            return;
        }
        let cleanQuery = "";
        if (queryText) {
            cleanQuery = queryText.trim();
        }
        if (cleanQuery === "") {
            return;
        }

        appendMessage("user", cleanQuery);
        userInput.value = "";
        setBusy(true);

        resetStepper();
        if (nodeStepper) {
            nodeStepper.style.display = "block";
        }

        // Tạo bubble tạm thời để stream tokens vào
        const botMsgDiv = document.createElement("div");
        botMsgDiv.className = "message bot-message";
        botMsgDiv.innerHTML = `
            <div class="bubble" id="streamingBubble">
                <span id="streamingText" style="color: #94a3b8; font-style: italic;">Đang kết nối agent...</span>
            </div>
        `;
        chatMessages.appendChild(botMsgDiv);
        scrollToBottom();

        const streamingBubble = botMsgDiv.querySelector("#streamingBubble");
        let accumulatedText = "";
        let finalDoneMeta = null;
        let streamStarted = false;

        try {
            const response = await fetch("/api/agent/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: cleanQuery, session_id: currentSessionId })
            });

            if (!response.ok || !response.body) {
                botMsgDiv.remove();
                if (!await recoverInteraction() && response.status < 500) {
                    await submitQueryFallback(cleanQuery);
                } else if (!pendingCard) {
                    appendMessage("bot", "Không nhận được phản hồi từ hệ thống.");
                }
                return;
            }

            streamStarted = true;
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
                                updateStepperProgress(eventData.description, eventData.action);
                                renderPlan(eventData.plan);
                            } else if (eventType === "awaiting_input") {
                                finalDoneMeta = eventData;
                            } else if (eventType === "token") {
                                accumulatedText = accumulatedText + eventData.content;
                                if (streamingBubble) {
                                    streamingBubble.innerHTML = parseMarkdown(accumulatedText);
                                    scrollToBottom();
                                }
                            } else if (eventType === "done") {
                                finalDoneMeta = eventData;
                                renderPlan(eventData.plan);
                                finishProgress();
                            }
                        } catch (e) {
                            console.warn("Parse SSE event error:", e);
                        }
                    }
                }
            }

            if (finalDoneMeta) {
                botMsgDiv.remove();
                if (finalDoneMeta.status === "awaiting_input") {
                    renderInteraction(finalDoneMeta);
                    return;
                }
                let finalText = accumulatedText;
                if (finalDoneMeta.full_answer) {
                    finalText = finalDoneMeta.full_answer;
                }

                appendMessage("bot", finalText, {
                    citations: finalDoneMeta.citations,
                    sources: finalDoneMeta.sources
                });
            } else {
                botMsgDiv.remove();
                if (!await recoverInteraction()) {
                    finishProgress("Kết nối bị gián đoạn");
                    appendMessage("bot", "Kết nối bị gián đoạn trước khi nhận được phản hồi.");
                }
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
                await submitQueryFallback(cleanQuery);
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

    if (clearChatBtn) {
        clearChatBtn.addEventListener("click", () => {
            currentSessionId = newSessionId();
            sessionStorage.setItem(sessionKey, currentSessionId);
            pendingCard = null;
            setBusy(false);
            chatMessages.innerHTML = initialWelcomeHtml;
            if (nodeStepper) {
                nodeStepper.style.display = "none";
            }
            userInput.value = "";
            userInput.focus();
        });
    }

    setBusy(true);
    recoverInteraction(null, true).finally(() => {
        setBusy(false);
        userInput.focus();
    });
});
